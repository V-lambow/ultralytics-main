import json
import logging
import threading
from pathlib import Path

import numpy as np
import torch
import zmq
from ultralytics import YOLO
from ultralytics.data.augment import BaseTransform, Compose, LetterBox, Format
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import unwrap_model

logger = logging.getLogger("train_service")


class ROICrop(BaseTransform):
    """Crop images to ROI region and adjust bounding boxes."""

    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def apply_image(self, labels, params=None):
        img = labels["img"]
        h, w = img.shape[:2]
        x1 = max(0, min(self.x, w))
        y1 = max(0, min(self.y, h))
        x2 = max(0, min(self.x + self.w, w))
        y2 = max(0, min(self.y + self.h, h))
        labels["img"] = img[y1:y2, x1:x2]
        return labels

    def apply_instances(self, labels, params=None):
        instances = labels.get("instances")
        if instances is None or len(instances) == 0:
            return labels
        instances.denormalize(labels["ori_shape"][1], labels["ori_shape"][0])
        instances.convert_bbox("xyxy")
        bboxes = instances.bboxes
        x1, y1, x2, y2 = self.x, self.y, self.x + self.w, self.y + self.h
        bw = x2 - x1
        bh = y2 - y1
        mask = (bboxes[:, 0] < x2) & (bboxes[:, 2] > x1) & (bboxes[:, 1] < y2) & (bboxes[:, 3] > y1)
        bboxes = bboxes[mask]
        if len(bboxes) > 0:
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clip(x1, x2) - x1
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clip(y1, y2) - y1
        instances.update(bboxes, instances.segments[mask] if len(instances.segments) else instances.segments)
        instances.convert_bbox("xywh")
        instances.normalize(bw, bh)
        labels["instances"] = instances
        return labels


class ROIDataset(YOLODataset):
    """YOLODataset with ROI crop transform inserted before LetterBox."""

    _roi = None

    @classmethod
    def set_roi(cls, x, y, w, h):
        cls._roi = (x, y, w, h)

    def build_transforms(self, hyp=None):
        transforms = super().build_transforms(hyp)
        if self._roi and self.augment:
            x, y, w, h = self._roi
            transforms.insert(0, ROICrop(x, y, w, h))
        return transforms


class ROITrainer(DetectionTrainer):
    """DetectionTrainer that uses ROIDataset."""

    _roi = None

    @classmethod
    def set_roi(cls, x, y, w, h):
        cls._roi = (x, y, w, h)

    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        ROIDataset.set_roi(*self._roi) if self._roi else None
        dataset = ROIDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or mode == "val",
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=gs,
            pad=0.0 if mode == "train" else 0.5,
            prefix=f"{mode}: ",
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )
        return dataset


class TrainService:
    def __init__(self, config: dict, config_path: str = ""):
        self.train_cfg = config.get("train", {})
        self.port = self.train_cfg.get("port", 9257)
        self.output_dir = str(Path(config_path).parent / "output") if config_path else "output"

        self.trainer = None
        self.training = False
        self.paused = False
        self.save_dir = None
        self.last_ckpt = None

        self._resume = False
        self._saved_project = None
        self._saved_name = None
        self._train_thread = None
        self._train_done = threading.Event()

        self.ctx = zmq.Context()
        self.pub = None
        self.rep = None
        self._running = False

    def start_server(self):
        self._running = True

        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.setsockopt(zmq.LINGER, 0)
        self.pub.bind(f"tcp://*:{self.port}")
        logger.info(f"PUB bound to tcp://*:{self.port}")

        self.rep = self.ctx.socket(zmq.REP)
        self.rep.setsockopt(zmq.LINGER, 0)
        self.rep.bind(f"tcp://*:{self.port + 1}")
        logger.info(f"REP bound to tcp://*:{self.port + 1}")

        poller = zmq.Poller()
        poller.register(self.rep, zmq.POLLIN)

        logger.info("Train service started. Waiting for commands...")

        while self._running:
            try:
                socks = dict(poller.poll(timeout=100))
                if self.rep in socks:
                    msg = self.rep.recv_json()
                    logger.info(f"Received: {msg.get('cmd', '?')}")
                    resp = self._handle_command(msg)
                    self.rep.send_json(resp)
            except zmq.ZMQError as e:
                if self._running:
                    logger.error(f"ZMQ error: {e}")
                break

        self._cleanup()

    def stop_server(self):
        self._running = False

    def _cleanup(self):
        for s in (self.pub, self.rep):
            if s and not s.closed:
                s.close()
        self.ctx.term()
        logger.info("Service stopped.")

    def _handle_command(self, msg: dict) -> dict:
        cmd = msg.get("cmd", "")
        handlers = {
            "train": self.start_train,
            "pause": self.pause_train,
            "holdon": self.holdon_train,
            "stop": self.stop_train,
            "status": self.get_status,
        }
        handler = handlers.get(cmd)
        if not handler:
            return {"status": "error", "message": f"Unknown command: {cmd}"}
        try:
            return handler()
        except Exception as e:
            logger.error(f"Command '{cmd}' failed: {e}")
            return {"status": "error", "message": str(e)}

    def start_train(self) -> dict:
        if self.training:
            return {"status": "error", "message": "Already training"}
        self._resume = False
        self._train_done.clear()
        self.training = True
        self.paused = False
        self._train_thread = threading.Thread(target=self._train_loop, daemon=True)
        self._train_thread.start()
        return {"status": "ok", "message": "Training started"}

    def _train_loop(self):
        try:
            model_path = self.train_cfg.get("modelPath", "yolo26n.pt")
            model = YOLO(model_path)
            model.add_callback("on_train_start", self._on_train_start)
            model.add_callback("on_fit_epoch_end", self._on_fit_epoch_end)
            model.add_callback("on_train_end", self._on_train_end)

            resume = self._resume
            self._resume = False

            cfg_device = self.train_cfg.get("device", "auto")
            if cfg_device == "auto":
                cfg_device = "0" if torch.cuda.is_available() else "cpu"

            kw = dict(
                model=model_path,
                data=self.train_cfg.get("dataYaml", "coco8.yaml"),
                epochs=self.train_cfg.get("epochs", 100),
                batch=self.train_cfg.get("batch", 16),
                imgsz=self.train_cfg.get("imgsz", 640),
                device=cfg_device,
                amp=self.train_cfg.get("amp", True),
                cos_lr=self.train_cfg.get("cos_lr", False),
                optimizer=self.train_cfg.get("optimizer", "auto"),
                resume=resume,
                task=self.train_cfg.get("task", "detect"),
                project=self.output_dir,
                plots=True,
                exist_ok=True,
            )

            alias = self.train_cfg.get("alias", "")
            if alias:
                kw["name"] = alias

            use_roi = self.train_cfg.get("useRoi", False)
            roi_cfg = self.train_cfg.get("roi", {})
            if use_roi and all(k in roi_cfg for k in ("x", "y", "width", "height")):
                rx, ry, rw, rh = roi_cfg["x"], roi_cfg["y"], roi_cfg["width"], roi_cfg["height"]
                if rw > 0 and rh > 0:
                    ROITrainer.set_roi(rx, ry, rw, rh)
                    kw["trainer"] = ROITrainer
                    logger.info(f"ROI enabled: ({rx}, {ry}, {rw}, {rh})")

            if resume and self._saved_project and self._saved_name:
                kw["project"] = self._saved_project
                kw["name"] = self._saved_name

            model.train(**kw)
        except Exception as e:
            logger.error(f"Training error: {e}")
            self._publish("error", {"message": str(e)})
        finally:
            self.training = False
            self._train_done.set()
            logger.info("Training thread finished.")

    def _on_train_start(self, trainer):
        self.trainer = trainer
        self.save_dir = str(trainer.save_dir)
        self.last_ckpt = str(Path(self.save_dir) / "weights" / "last.pt")
        self._saved_project = str(trainer.args.project)
        self._saved_name = str(trainer.args.name)
        logger.info(f"Save dir: {self.save_dir}")

    def _on_fit_epoch_end(self, trainer):
        epoch = trainer.epoch + 1
        msg = {
            "epoch": epoch,
            "total_epochs": trainer.epochs,
            "loss": float(trainer.loss) if trainer.loss is not None else 0,
            "box_loss": float(trainer.tloss[0]) if trainer.tloss is not None else 0,
            "cls_loss": float(trainer.tloss[1]) if trainer.tloss is not None else 0,
            "dfl_loss": float(trainer.tloss[2]) if trainer.tloss is not None else 0,
            "precision": float(trainer.metrics.get("metrics/precision(B)", 0)),
            "recall": float(trainer.metrics.get("metrics/recall(B)", 0)),
            "mAP50": float(trainer.metrics.get("metrics/mAP50(B)", 0)),
            "mAP50_95": float(trainer.metrics.get("metrics/mAP50-95(B)", 0)),
        }
        self._publish("metrics", msg)
        logger.info(
            f"Epoch {epoch}/{trainer.epochs} | loss: {msg['loss']:.4f} | "
            f"P: {msg['precision']:.4f} R: {msg['recall']:.4f} | "
            f"mAP50: {msg['mAP50']:.4f} mAP50-95: {msg['mAP50_95']:.4f}"
        )

    def _on_train_end(self, trainer):
        self._publish("train_end", {"save_dir": str(trainer.save_dir)})
        logger.info(f"Training complete. Results: {trainer.save_dir}")

    def _publish(self, topic: str, msg: dict):
        if self.pub and not self.pub.closed:
            self.pub.send_multipart([topic.encode(), json.dumps(msg).encode()])

    def pause_train(self) -> dict:
        if not self.training:
            return {"status": "error", "message": "Not training"}
        if self.paused:
            return {"status": "error", "message": "Already paused"}
        if self.trainer:
            self.trainer.stop = True
        self.paused = True
        logger.info("Training paused.")
        return {"status": "ok", "message": "Training paused"}

    def holdon_train(self) -> dict:
        if not self.paused:
            return {"status": "error", "message": "Not paused"}
        if self._train_thread and self._train_thread.is_alive():
            self._train_thread.join(timeout=30)
        self.paused = False
        self._resume = True
        self._train_done.clear()
        self.training = True
        self._train_thread = threading.Thread(target=self._train_loop, daemon=True)
        self._train_thread.start()
        logger.info("Training resumed.")
        return {"status": "ok", "message": "Training resumed"}

    def stop_train(self) -> dict:
        if not self.training:
            return {"status": "error", "message": "Not training"}
        if self.trainer:
            self.trainer.stop = True
        self.paused = False
        logger.info("Training stopped.")
        return {"status": "ok", "message": "Training stopped"}

    def get_status(self) -> dict:
        return {
            "status": "ok",
            "training": self.training,
            "paused": self.paused,
            "save_dir": self.save_dir,
            "last_ckpt": self.last_ckpt,
        }
