import json
import logging
import math
import shutil
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
import zmq
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.obb.train import OBBTrainer
from ultralytics.models.yolo.pose.train import PoseTrainer
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.patches import imread
from ultralytics.utils.torch_utils import unwrap_model

logger = logging.getLogger("train_service")


def _clip_rect(pts, x1, y1, x2, y2):
    for a, b, c in ((-1, 0, -x1), (1, 0, x2), (0, -1, -y1), (0, 1, y2)):
        if pts is None or not len(pts):
            return None
        out = []
        for i in range(len(pts)):
            p, q = pts[i - 1], pts[i]
            dp, dq = a * p[0] + b * p[1] - c, a * q[0] + b * q[1] - c
            if dq <= 0:
                if dp > 0:
                    out.append(p + (q - p) * (dp / (dp - dq)))
                out.append(q)
            elif dp <= 0:
                out.append(p + (q - p) * (dp / (dp - dq)))
        pts = np.asarray(out, dtype=np.float32).reshape(-1, 2)
    return pts if len(pts) else None


def _crop_label(label, roi):
    x1, y1, x2, y2 = roi
    h0, w0 = label["shape"]
    ix1, iy1, ix2, iy2 = max(x1, 0), max(y1, 0), min(x2, w0), min(y2, h0)
    if ix2 <= ix1 or iy2 <= iy1:
        raise ValueError(f"ROI {roi} does not intersect image {label['im_file']} of shape {(h0, w0)}")
    sx, sy = ix2 - ix1, iy2 - iy1
    cls = label["cls"]
    bboxes = label["bboxes"]
    segments = label["segments"]
    keypoints = label.get("keypoints")
    n = len(cls)
    keep = np.ones(n, dtype=bool)
    if len(bboxes):
        px1 = bboxes[:, 0] * w0 - bboxes[:, 2] * w0 / 2
        py1 = bboxes[:, 1] * h0 - bboxes[:, 3] * h0 / 2
        px2 = px1 + bboxes[:, 2] * w0
        py2 = py1 + bboxes[:, 3] * h0
        px1, py1 = np.maximum(px1, ix1), np.maximum(py1, iy1)
        px2, py2 = np.minimum(px2, ix2), np.minimum(py2, iy2)
        keep &= (px2 > px1) & (py2 > py1)
        bboxes = np.stack([(px1 + px2) / 2 - ix1, (py1 + py2) / 2 - iy1, px2 - px1, py2 - py1], 1)
        bboxes /= np.array([sx, sy, sx, sy], dtype=np.float32)
    else:
        bboxes = np.zeros((0, 4), dtype=np.float32)
    if len(segments):
        clipped = []
        for i, seg in enumerate(segments):
            if not keep[i]:
                continue
            pts = _clip_rect(np.asarray(seg, dtype=np.float32) * [w0, h0], ix1, iy1, ix2, iy2)
            if pts is not None and len(pts) >= 3:
                clipped.append((pts - [ix1, iy1]) / [sx, sy])
            else:
                keep[i] = False
        label["segments"] = clipped
    else:
        label["segments"] = []
    label["cls"] = cls[keep]
    label["bboxes"] = bboxes[keep]
    if keypoints is not None:
        kpts = keypoints.copy()
        kpts[..., 0] = np.clip(kpts[..., 0] * w0, ix1, ix2) - ix1
        kpts[..., 1] = np.clip(kpts[..., 1] * h0, iy1, iy2) - iy1
        kpts[..., 0] /= sx
        kpts[..., 1] /= sy
        outside = (
            (keypoints[..., 0] * w0 < ix1)
            | (keypoints[..., 0] * w0 > ix2)
            | (keypoints[..., 1] * h0 < iy1)
            | (keypoints[..., 1] * h0 > iy2)
        )
        kpts[..., 2][outside] = 0.0
        label["keypoints"] = kpts[keep]
    label["shape"] = (sy, sx)
    label["crop_shape"] = (sy, sx)
    label["full_shape"] = (h0, w0)
    label["roi_origin"] = (ix1, iy1)
    return label


def _canvas_layout(h0, w0, lb_h, lb_w):
    r = min(lb_h / h0, lb_w / w0)
    cw, ch = round(w0 * r), round(h0 * r)
    pad_x = round((lb_w - cw) / 2 - 0.1)
    pad_y = round((lb_h - ch) / 2 - 0.1)
    return r, (pad_x, pad_y), (ch, cw)


def _sample_geometry(batch, si):
    sy, sx = batch["crop_shape"][si]
    h0, w0 = batch["full_shape"][si]
    x1, y1 = (int(v) for v in batch["roi_origin"][si])
    gh, gw = batch["ratio_pad"][si][0]
    left, top = batch["ratio_pad"][si][1]
    ch, cw = batch["img"].shape[2:]
    r = min(ch / (sy * gh), cw / (sx * gw), 1.0)  # val LetterBox uses scaleup=False
    kx, ky = gw * r, gh * r
    rr, (pad_x, pad_y), _ = _canvas_layout(h0, w0, ch, cw)
    ox, oy = x1 - left / kx, y1 - top / ky
    return {
        "canvas": (ch, cw),
        "full": (h0, w0),
        "pad": (left, top),
        "origin": (x1, y1),
        "gain": (kx, ky),
        "norm_affine": (rr / kx, (ox * rr + pad_x) / cw, rr / ky, (oy * rr + pad_y) / ch),
        "px_affine": (1.0 / kx, ox, 1.0 / ky, oy),
        "plot_affine": (rr / kx, ox * rr + pad_x, rr / ky, oy * rr + pad_y),
    }


def _mask_affine_image(geo, mh, mw):
    ch, cw = geo["canvas"]
    gx, ox, gy, oy = geo["px_affine"]
    return np.array([[cw / mw * gx, 0.0, ox], [0.0, ch / mh * gy, oy]], dtype=np.float32)


def _mask_affine_canvas(geo, mh, mw):
    ch, cw = geo["canvas"]
    gx, ox, gy, oy = geo["px_affine"]
    rr, (pad_x, pad_y), _ = _canvas_layout(*geo["full"], ch, cw)
    return np.array(
        [[rr * cw / mw * gx, 0.0, ox * rr + pad_x], [0.0, rr * ch / mh * gy, oy * rr + pad_y]],
        dtype=np.float32,
    )


def _warp_masks(masks, mat, dsize):
    arr = masks.detach().cpu().numpy()
    if not len(arr):
        return masks
    out = torch.from_numpy(np.stack([cv2.warpAffine(p, mat, dsize, flags=cv2.INTER_NEAREST) for p in arr]))
    return out.to(device=masks.device, dtype=masks.dtype)


def _full_canvas(batch):
    ch, cw = batch["img"].shape[2:]
    flags = cv2.IMREAD_GRAYSCALE if batch["img"].shape[1] == 1 else cv2.IMREAD_COLOR
    imgs = []
    for si, im_file in enumerate(batch["im_file"]):
        h0, w0 = batch["full_shape"][si]
        _, (pad_x, pad_y), (rh_img, rw_img) = _canvas_layout(h0, w0, ch, cw)
        img = imread(im_file, flags=flags)
        if img is None:
            raise ValueError(f"Failed to read image {im_file} for ROI plot canvas")
        img = cv2.resize(img, (rw_img, rh_img), interpolation=cv2.INTER_LINEAR)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] > 3:
            img = np.ascontiguousarray(img[..., :3])
        img = cv2.copyMakeBorder(
            img, pad_y, ch - rh_img - pad_y, pad_x, cw - rw_img - pad_x, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        imgs.append(torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)))
    return torch.stack(imgs, 0).float()


class ROIDataset(YOLODataset):
    """YOLODataset that crops images and labels to a fixed ROI at the read layer."""

    def __init__(self, *args, roi=None, **kwargs):
        self._roi = roi
        super().__init__(*args, **kwargs)

    def get_labels(self):
        labels = super().get_labels()
        if self._roi is None:
            return labels
        return [_crop_label(label, self._roi) for label in labels]

    def _load_full_image(self, i):
        fn, f = self.npy_files[i], self.im_files[i]
        if fn.exists():
            try:
                im = np.load(fn)
                npy_channels = im.shape[-1] if im.ndim >= 3 else 1
                if npy_channels != self.channels:
                    LOGGER.warning(
                        f"{self.prefix}Removing stale *.npy image file {fn} with {npy_channels} channels, "
                        f"expected {self.channels}"
                    )
                    fn.unlink(missing_ok=True)
                    im = imread(f, flags=self.cv2_flag)
            except Exception as e:
                LOGGER.warning(f"{self.prefix}Removing corrupt *.npy image file {fn} due to: {e}")
                fn.unlink(missing_ok=True)
                im = imread(f, flags=self.cv2_flag)
        else:
            im = imread(f, flags=self.cv2_flag)
        if im is None:
            raise FileNotFoundError(f"Image Not Found {f}")
        return im

    def load_image(self, i, rect_mode=True, resize_short=False):
        if self._roi is None:
            return super().load_image(i, rect_mode, resize_short)
        im, hw0, hw = self.ims[i], self.im_hw0[i], self.im_hw[i]
        if im is not None:
            return im, hw0, hw
        im = self._load_full_image(i)
        x1, y1, x2, y2 = self._roi
        h0, w0 = im.shape[:2]
        ix1, iy1 = max(x1, 0), max(y1, 0)
        ix2, iy2 = min(x2, w0), min(y2, h0)
        if ix2 <= ix1 or iy2 <= iy1:
            raise ValueError(f"ROI {self._roi} does not intersect image {self.im_files[i]} of shape {(h0, w0)}")
        im = im[iy1:iy2, ix1:ix2]
        h0, w0 = im.shape[:2]
        if rect_mode:
            if resize_short:
                r = self.imgsz / min(h0, w0)
                w, h = (math.ceil(w0 * r), self.imgsz) if h0 < w0 else (self.imgsz, math.ceil(h0 * r))
            else:
                r = self.imgsz / max(h0, w0)
                w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
            if r != 1:
                im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        elif not (h0 == w0 == self.imgsz):
            im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        if im.ndim == 2:
            im = im[..., None]
        if self.augment:
            self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]
            self.buffer.append(i)
            if 1 < len(self.buffer) >= self.max_buffer_length:
                j = self.buffer.pop(0)
                if self.cache != "ram":
                    self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None
        return im, (h0, w0), im.shape[:2]


class ROIValidatorMixin:
    """Remap ROI-canvas validation batches and predictions to full-image coordinates."""

    def _prepare_batch(self, si, batch):
        pbatch = super()._prepare_batch(si, batch)
        geo = _sample_geometry(batch, si)
        gx, ox, gy, oy = geo["px_affine"]
        pbatch["ori_shape"] = geo["full"]
        pbatch["ratio_pad"] = ((1.0 / gx, 1.0 / gy), (-ox / gx, -oy / gy))
        pbatch["roi_geometry"] = geo
        return pbatch

    def scale_preds(self, predn, pbatch):
        geo = pbatch["roi_geometry"]
        gx, ox, gy, oy = geo["px_affine"]
        h0, w0 = geo["full"]
        out = {**predn}
        bboxes = predn["bboxes"].clone()
        if bboxes.shape[1] == 5:
            bboxes[:, 0] = bboxes[:, 0] * gx + ox
            bboxes[:, 1] = bboxes[:, 1] * gy + oy
            bboxes[:, 2] *= gx
            bboxes[:, 3] *= gy
        else:
            bboxes[:, 0::2] = bboxes[:, 0::2] * gx + ox
            bboxes[:, 1::2] = bboxes[:, 1::2] * gy + oy
            bboxes[:, 0::2] = bboxes[:, 0::2].clamp_(0, w0)
            bboxes[:, 1::2] = bboxes[:, 1::2].clamp_(0, h0)
        out["bboxes"] = bboxes
        keypoints = predn.get("keypoints")
        if keypoints is not None and keypoints.numel():
            kpts = keypoints.clone()
            kpts[..., 0] = kpts[..., 0] * gx + ox
            kpts[..., 1] = kpts[..., 1] * gy + oy
            out["keypoints"] = kpts
        masks = predn.get("masks")
        if masks is not None and masks.numel():
            mat = _mask_affine_image(geo, *masks.shape[2:])
            out["masks"] = _warp_masks(masks, mat, (w0, h0)).byte()
        return out

    def plot_val_samples(self, batch, ni):
        self._map_gt(batch)
        batch["img"] = _full_canvas(batch)
        super().plot_val_samples(batch, ni)

    def plot_predictions(self, batch, preds, ni, *args, **kwargs):
        self._map_preds(preds, batch)
        batch["img"] = _full_canvas(batch)
        super().plot_predictions(batch, preds, ni, *args, **kwargs)

    def _map_gt(self, batch):
        masks = batch.get("masks")
        per_instance = bool(
            masks is not None and masks.numel() and masks.shape[0] == batch["batch_idx"].shape[0] and masks.max() <= 1
        )
        for si in range(len(batch["im_file"])):
            geo = _sample_geometry(batch, si)
            ax, bx, ay, by = geo["norm_affine"]
            idx = batch["batch_idx"] == si
            bboxes = batch["bboxes"]
            if len(bboxes):
                if bboxes.shape[1] == 5:
                    bboxes[idx, 0] = bboxes[idx, 0] * ax + bx
                    bboxes[idx, 1] = bboxes[idx, 1] * ay + by
                    bboxes[idx, 2] *= ax
                    bboxes[idx, 3] *= ay
                else:
                    bboxes[idx, 0::2] = bboxes[idx, 0::2] * ax + bx
                    bboxes[idx, 1::2] = bboxes[idx, 1::2] * ay + by
            keypoints = batch.get("keypoints")
            if keypoints is not None and len(keypoints):
                keypoints[idx, :, 0] = keypoints[idx, :, 0] * ax + bx
                keypoints[idx, :, 1] = keypoints[idx, :, 1] * ay + by
            if masks is not None and masks.numel():
                mat = _mask_affine_canvas(geo, *masks.shape[-2:])
                if per_instance:
                    if idx.any():
                        masks[idx] = _warp_masks(masks[idx], mat, geo["canvas"][::-1])
                else:
                    masks[si : si + 1] = _warp_masks(masks[si : si + 1], mat, geo["canvas"][::-1])

    def _map_preds(self, preds, batch):
        for si, pred in enumerate(preds):
            geo = _sample_geometry(batch, si)
            gx, ox, gy, oy = geo["plot_affine"]
            bboxes = pred["bboxes"]
            if bboxes.shape[1] == 5:
                bboxes[:, 0] = bboxes[:, 0] * gx + ox
                bboxes[:, 1] = bboxes[:, 1] * gy + oy
                bboxes[:, 2] *= gx
                bboxes[:, 3] *= gy
            else:
                bboxes[:, 0::2] = bboxes[:, 0::2] * gx + ox
                bboxes[:, 1::2] = bboxes[:, 1::2] * gy + oy
            if "keypoints" in pred:
                pred["keypoints"][..., 0] = pred["keypoints"][..., 0] * gx + ox
                pred["keypoints"][..., 1] = pred["keypoints"][..., 1] * gy + oy
            if "masks" in pred and pred["masks"].numel():
                mat = _mask_affine_canvas(geo, *pred["masks"].shape[2:])
                pred["masks"] = _warp_masks(pred["masks"], mat, geo["canvas"][::-1])


class ROITrainerMixin:
    """Build ROI-cropped datasets for both splits and rebind the validator for full-image remapping."""

    def build_dataset(self, img_path, mode="train", batch=None):
        cfg = self.args
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return ROIDataset(
            img_path=img_path,
            imgsz=cfg.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=cfg,
            rect=cfg.rect or mode == "val",
            cache=cfg.cache or None,
            single_cls=cfg.single_cls or False,
            stride=gs,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=cfg.task,
            classes=cfg.classes,
            data=self.data,
            fraction=cfg.fraction if mode == "train" else 1.0,
            roi=self._roi,
        )

    def get_validator(self):
        validator = super().get_validator()
        base = type(validator)
        validator.__class__ = type(base.__name__, (ROIValidatorMixin, base), {})
        return validator


def build_roi_trainer(task, roi):
    bases = {
        "detect": DetectionTrainer,
        "segment": SegmentationTrainer,
        "pose": PoseTrainer,
        "obb": OBBTrainer,
    }
    if task not in bases:
        raise ValueError(f"ROI training is not supported for task '{task}', expected one of {sorted(bases)}")
    base = bases[task]
    return type(f"ROI{base.__name__}", (ROITrainerMixin, base), {"_roi": roi})


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
        self.resume_ckpt = None

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
            model.add_callback("on_model_save", self._on_model_save)
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
                resume=self.resume_ckpt if resume else False,
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
                    kw["trainer"] = build_roi_trainer(kw["task"], (rx, ry, rw, rh))
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

    def _on_model_save(self, trainer):
        # final_eval() strips the optimizer from last.pt after a stop, so keep a resumable copy on pause
        if self.paused:
            self.resume_ckpt = str(Path(trainer.wdir) / "pause.pt")
            shutil.copy2(trainer.last, self.resume_ckpt)

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
            if self._train_thread.is_alive():
                return {"status": "error", "message": "Previous training is still stopping, retry later"}
        if not self.resume_ckpt or not Path(self.resume_ckpt).exists():
            return {"status": "error", "message": "No resumable checkpoint saved for this pause"}
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
