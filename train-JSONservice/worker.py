"""训练子进程入口：读 payload、校验环境、调用 ultralytics、经 stdout 回传事件。"""

import sys
import traceback

import matplotlib

from protocol import EventLogStream, emit, load_payload

matplotlib.use("Agg")


def _check_tensor(device):
    import torch

    tensor = torch.tensor([1.0], device=device) + 1
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    if tensor.item() != 2.0:
        raise RuntimeError(f"Tensor validation failed on {device}")


def resolve_device(device):
    import torch

    if device in (None, "", "auto"):
        device = "0" if torch.cuda.is_available() else "cpu"

    if isinstance(device, list):
        devices = [str(item) for item in device]
    else:
        devices = [item for item in str(device).split(",") if item]

    for item in devices:
        if item == "cpu":
            _check_tensor("cpu")
        elif item == "mps":
            if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
                raise RuntimeError("MPS is not available")
            _check_tensor("mps")
        else:
            index = int(item)
            if not torch.cuda.is_available() or index >= torch.cuda.device_count():
                raise RuntimeError(f"CUDA device {index} is not available")
            _check_tensor(f"cuda:{index}")
    return device


def build_roi_trainer(train_args):
    use_roi = train_args.pop("useRoi", False)
    roi_cfg = train_args.pop("roi", None) or {}
    if not use_roi or not any(roi_cfg.get(key) for key in ("x", "y", "width", "height")):
        return None

    required = ("x", "y", "width", "height")
    missing = [key for key in required if key not in roi_cfg]
    if missing:
        raise ValueError(f"ROI is missing fields: {missing}")

    roi = tuple(int(round(float(roi_cfg[key]))) for key in required)
    width, height = roi[2], roi[3]
    if width <= 0 or height <= 0:
        raise ValueError(f"ROI width/height must be > 0, got {roi_cfg['width']}x{roi_cfg['height']}")

    from roi import build_roi_trainer as build

    task = train_args.get("task", "detect")
    return build(task, roi)


def install_gray_shim():
    import cv2
    import numpy as np
    import ultralytics.data.base as base
    import ultralytics.data.loaders as loaders
    import ultralytics.utils.patches as patches

    original = getattr(patches, "imread", None)
    if original is None:
        raise RuntimeError("Grayscale shim failed: ultralytics.utils.patches.imread not found")
    if getattr(original, "is_gray_shim", False):
        return

    def imread(filename, flags=cv2.IMREAD_COLOR):
        im = original(filename, flags)
        if flags != cv2.IMREAD_GRAYSCALE and im is not None and im.ndim == 3 and im.shape[2] == 1:
            return np.repeat(im, 3, axis=2)
        return im

    imread.is_gray_shim = True
    patches.imread = imread
    for module in (base, loaders):
        if getattr(module, "imread", None) is original:
            module.imread = imread


def emit_started(total_epochs):
    def on_train_start(trainer):
        emit("training_started", total_epochs=total_epochs, save_dir=str(trainer.save_dir))

    return on_train_start


def emit_metrics(trainer):
    tloss = trainer.tloss
    metrics = trainer.metrics or {}
    task = getattr(trainer.args, "task", "detect")

    def loss_at(index):
        if tloss is None or index >= len(tloss):
            return None
        return float(tloss[index])

    payload = {
        "epoch": trainer.epoch + 1,
        "total_epochs": trainer.epochs,
        "loss": float(trainer.loss) if trainer.loss is not None else None,
        "box_loss": loss_at(0),
        "cls_loss": None,
        "dfl_loss": None,
        "precision": float(metrics.get("metrics/precision(B)", 0)),
        "recall": float(metrics.get("metrics/recall(B)", 0)),
        "mAP50": float(metrics.get("metrics/mAP50(B)", 0)),
        "mAP50_95": float(metrics.get("metrics/mAP50-95(B)", 0)),
        "task": task,
    }
    if task in ("detect", "obb"):
        payload["cls_loss"] = loss_at(1)
        payload["dfl_loss"] = loss_at(2)
    emit("training_metrics", **payload)


def run_train(payload):
    train_args = dict(payload["args"])
    train_args["device"] = resolve_device(train_args.get("device"))
    total_epochs = train_args.get("epochs", 100)

    from ultralytics import YOLO

    log_stream = EventLogStream()
    original_stdout, original_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout = log_stream
        sys.stderr = log_stream
        model = YOLO(train_args.pop("model"))
        model.add_callback("on_train_start", emit_started(total_epochs))
        model.add_callback("on_fit_epoch_end", emit_metrics)
        trainer = build_roi_trainer(train_args)
        if trainer is not None:
            train_args["trainer"] = trainer
        model.train(**train_args)
    finally:
        log_stream.flush()
        sys.stdout, sys.stderr = original_stdout, original_stderr

    emit("training_completed", results="Training completed successfully")


def main(payload_path):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

    try:
        payload = load_payload(payload_path)
        if payload["action"] != "train":
            raise ValueError(f"Unsupported action: {payload['action']}")
        run_train(payload)
    except Exception as exc:
        emit(
            "training_error",
            error=str(exc),
            exception_type=type(exc).__name__,
            traceback=traceback.format_exc(),
        )
        raise SystemExit(1) from exc


install_gray_shim()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: worker.py <payload.json>")
    main(sys.argv[1])
