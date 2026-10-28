# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""CURB (Curriculum Resampling Under Budget): difficulty-driven resampling of the training set.

Every ``curb_interval`` epochs the trainer re-runs inference over the unaugmented training split, scores how
badly the current model misses / confuses / mislocalizes each image, and rebuilds the train sampler's
weights while keeping the draw budget fixed at ``len(dataset)`` per epoch so LR schedules stay untouched.
Per-image weights are median-normalized and clipped, per-class expected exposure is capped at ``curb_kappa``
times uniform exposure, and classes whose mean difficulty stops improving lose their boost (stop-loss) and
are listed in ``save_dir/curb_review.txt`` for manual inspection.

Usage:
    >>> from ultralytics import YOLO
    >>> YOLO("yolo26n.yaml").train(data="data.yaml", curb=True, curb_interval=25)
"""

from __future__ import annotations

import numpy as np
import torch

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.utils import LOGGER, nms, ops
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.torch_utils import autocast, smart_inference_mode, unwrap_model

# Fixed CURB design constants, deliberately not configurable
CLIP = (0.3, 6.0)  # per-image weight clip bounds
POW = 0.5  # difficulty ratio exponent
MATCH_IOU = 0.5  # IoU threshold for TP matching in the difficulty pass
FP_PENALTY = 0.5  # false positive weight relative to a missed ground truth box
STALL_ABS = 0.005  # min absolute mean-difficulty drop for a class to count as improving
STALL_REL = 0.01  # min relative mean-difficulty drop for a class to count as improving
QUOTA_PASSES = 5  # class-quota multiplier fixed-point iterations
DIFFICULTY_EPS = 1e-6  # median difficulty below which all weights stay uniform


class CurbSampler(torch.utils.data.WeightedRandomSampler):
    """Weighted replacement sampler with a fixed per-epoch draw budget of ``num_samples``.

    CURB swaps only the weight tensor in place, never the budget, so the learning-rate schedule paired with
    ``len(dataset)`` draws per epoch stays invariant across refreshes.
    """

    def __init__(self, num_samples: int) -> None:
        """Initialize uniform weights with a fixed draw budget of ``num_samples`` per epoch."""
        super().__init__(torch.ones(num_samples, dtype=torch.double), num_samples, replacement=True)

    def update(self, weights: torch.Tensor) -> None:
        """Replace the sampling weights in place, keeping the draw budget unchanged."""
        self.weights.copy_(weights.to(dtype=self.weights.dtype))


def _match(iou: torch.Tensor, gt_cls: torch.Tensor, pred_cls: torch.Tensor) -> tuple[list[float], int, int]:
    """Greedily match predictions to ground truth one-to-one, requiring class agreement and IoU >= MATCH_IOU.

    Args:
        iou (torch.Tensor): Pairwise IoU matrix (N, M) between N ground truth and M predicted boxes.
        gt_cls (torch.Tensor): Ground truth class indices of shape (N,).
        pred_cls (torch.Tensor): Predicted class indices of shape (M,).

    Returns:
        (tuple): Matched IoU values, missed ground truth count, and false positive count.
    """
    n_gt, n_pred = iou.shape
    pairs = torch.nonzero((iou >= MATCH_IOU) & (gt_cls[:, None] == pred_cls[None, :]), as_tuple=False)
    pairs = pairs[iou[pairs[:, 0], pairs[:, 1]].argsort(descending=True)]
    used_gt, used_pred, tp_ious = set(), set(), []
    for gi, pi in pairs.tolist():
        if gi not in used_gt and pi not in used_pred:
            used_gt.add(gi)
            used_pred.add(pi)
            tp_ious.append(float(iou[gi, pi]))
    return tp_ious, n_gt - len(tp_ious), n_pred - len(tp_ious)


def _image_difficulty(gt_cls: torch.Tensor, gt_box: torch.Tensor, pred: dict[str, torch.Tensor]) -> float:
    """Score one image's detection difficulty in [0, 1].

    Difficulty is ``(FN + 0.5 * FP + sum(1 - IoU) over matches) / max(1, n_gt + n_pred)``: a fully missed
    image reaches 1.0, an image holding only false positives saturates at 0.5, a clean image scores 0.

    Args:
        gt_cls (torch.Tensor): Ground truth class indices of shape (N,) in letterboxed pixel space.
        gt_box (torch.Tensor): Ground truth boxes of shape (N, 4), xyxy in letterboxed pixel space.
        pred (dict[str, torch.Tensor]): Predicted ``bboxes`` (M, 4) xyxy and ``cls`` (M,) after NMS.

    Returns:
        (float): Per-image difficulty clipped to [0, 1].
    """
    n_gt, n_pred = int(gt_cls.numel()), int(pred["cls"].numel())
    if n_gt and n_pred:
        tp_ious, fn, fp = _match(box_iou(gt_box, pred["bboxes"]), gt_cls, pred["cls"])
    else:
        tp_ious, fn, fp = [], n_gt, n_pred
    score = (fn + FP_PENALTY * fp + sum(1.0 - i for i in tp_ious)) / max(1, n_gt + n_pred)
    return float(np.clip(score, 0.0, 1.0))


@smart_inference_mode()
def _evaluate(trainer) -> dict[str, float]:
    """Run one inference pass over the unaugmented training split and return per-image difficulty.

    Mirrors DetectionValidator: EMA weights, val-mode (no augmentation) boxes, identical NMS and ground
    truth preparation, so difficulty is measured exactly where validation measures it.

    Args:
        trainer: The DetectionTrainer currently training.

    Returns:
        (dict[str, float]): Mapping of training image path to difficulty.
    """
    args = trainer.args
    batch_size = trainer.batch_size // max(trainer.world_size, 1)
    gs = max(int(unwrap_model(trainer.model).stride.max()), 32)
    dataset = build_yolo_dataset(
        args,
        trainer.data["train"],
        batch_size,
        trainer.data,
        mode="val",
        rect=True,
        stride=gs,
        fraction=args.fraction,
    )
    loader = build_dataloader(dataset, batch=batch_size, workers=args.workers, shuffle=False, rank=-1)
    model = trainer.ema.ema or trainer.model
    if args.compile and hasattr(model, "_orig_mod"):
        model = model._orig_mod
    model = model.float().eval()
    conf = 0.25 if args.conf is None else args.conf
    results = {}
    try:
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(trainer.device, non_blocking=trainer.device.type == "cuda")
            img = batch["img"].float() / 255
            with autocast(trainer.amp and trainer.device.type != "cpu", trainer.device.type):
                preds = model(img)
            outputs = nms.non_max_suppression(
                preds,
                conf,
                args.iou,
                nc=0,
                multi_label=True,
                agnostic=args.single_cls or args.agnostic_nms,
                max_det=args.max_det,
                end2end=getattr(model, "end2end", False),
            )
            scale = torch.tensor(img.shape[2:], device=trainer.device)[[1, 0, 1, 0]]
            for si, x in enumerate(outputs):
                pred = {"bboxes": x[:, :4], "cls": x[:, 5]}  # same slicing as DetectionValidator.postprocess
                idx = batch["batch_idx"] == si
                gt_cls = batch["cls"][idx].squeeze(-1)
                gt_box = batch["bboxes"][idx]
                if gt_cls.numel():
                    gt_box = ops.xywh2xyxy(gt_box) * scale  # normalized xywh -> letterboxed xyxy pixels
                results[batch["im_file"][si]] = _image_difficulty(gt_cls, gt_box, pred)
    finally:
        loader.close()
    return results


def _refresh(trainer) -> None:
    """Rebuild CURB weights from a fresh difficulty pass and swap them into the live train sampler.

    Pipeline: EMA-smoothed difficulty -> median-normalized clipped weights -> stagnation stop-loss ->
    per-class exposure quota -> in-place sampler update and prefetch reset.

    Args:
        trainer: The DetectionTrainer currently training.
    """
    args = trainer.args
    state = trainer._curb
    train_ds = trainer.train_loader.dataset
    results = _evaluate(trainer)
    missing = [f for f in train_ds.im_files if f not in results]
    assert not missing, f"CURB difficulty pass missed {len(missing)} training images, e.g. {missing[0]}"
    d_new = np.fromiter((results[f] for f in train_ds.im_files), dtype=np.float64, count=len(train_ds.im_files))
    d = d_new if state["d"] is None else (1.0 - args.curb_ema) * state["d"] + args.curb_ema * d_new
    state["d"] = d

    # Per-class image membership and mean difficulty
    nc = trainer.data["nc"]
    names = trainer.data["names"]
    img_cls = [np.unique(lb["cls"].reshape(-1).astype(int)) for lb in train_ds.labels]
    members: list[np.ndarray] = []
    for c in range(nc):
        members.append(np.asarray([i for i, cs in enumerate(img_cls) if c in cs], dtype=int))
    sizes = np.array([m.size for m in members])
    mu = np.array([d[m].mean() if m.size else 0.0 for m in members])

    # Stagnation counters: a class improves only if its mean difficulty drops by a meaningful margin.
    # Classes with no training images carry no observations, so they never accumulate stall counts.
    if state["mu"] is None:
        state["stall"] = np.zeros(nc, dtype=int)
    else:
        drop = state["mu"] - mu
        improved = drop >= np.maximum(STALL_ABS, STALL_REL * np.abs(state["mu"]))
        state["stall"] = np.where(sizes > 0, np.where(improved, 0, state["stall"] + 1), 0)
    state["mu"] = mu
    stalled = state["stall"] >= args.curb_stall

    # Per-image weights: median-normalized difficulty, clipped
    med = float(np.median(d))
    w = np.ones_like(d) if med <= DIFFICULTY_EPS else np.clip((d / med) ** POW, CLIP[0], CLIP[1])

    # Stop-loss: images whose labels are all stalled lose their boost but keep any down-weighting
    if stalled.any():
        boost_stopped = np.array([bool(stalled[cs].all()) if cs.size else False for cs in img_cls])
        w[boost_stopped] = np.minimum(w[boost_stopped], 1.0)

    # Per-class exposure quota: expected draws of class-c images per epoch capped at curb_kappa * n_c
    n = len(d)
    for _ in range(QUOTA_PASSES):
        p = w / w.sum()
        g = np.ones(nc)
        for c, m in enumerate(members):
            if m.size:
                exposure = n * p[m].sum()
                cap = args.curb_kappa * sizes[c]
                if exposure > cap:
                    g[c] = cap / exposure
        mult = np.ones(n)
        for i, cs in enumerate(img_cls):
            if cs.size:
                mult[i] = g[cs].min()
        if np.all(mult == 1.0):
            break
        w = w * mult

    p = w / w.sum()
    exposures = np.array([n * p[m].sum() if m.size else 0.0 for m in members])
    state["sampler"].update(torch.from_numpy(w))
    trainer.train_loader.reset()  # drop prefetched indices drawn under the previous weights

    epoch = trainer.epoch + 1
    stalled_names = [str(names[int(c)]) for c in np.flatnonzero(stalled)]
    LOGGER.info(
        f"CURB: refreshed at epoch {epoch} | median difficulty {med:.3f} | weight range "
        f"[{w.min():.2f}, {w.max():.2f}] | stalled classes: {', '.join(stalled_names) or 'none'}"
    )
    _log_csv(trainer, epoch, names, mu, exposures, args.curb_kappa * sizes, state["stall"], stalled)
    if stalled.any():
        _log_review(trainer, epoch, names, d, members, stalled)


def _log_csv(trainer, epoch: int, names, mu, exposures, caps, stall, stalled) -> None:
    """Append one row per class to save_dir/curb.csv, creating it with a header on first refresh.

    Args:
        trainer: The DetectionTrainer currently training.
        epoch (int): Current 1-based epoch number.
        names: Class names (list or dict keyed by class index).
        mu (np.ndarray): Per-class mean difficulty.
        exposures (np.ndarray): Per-class expected draws per epoch under current weights.
        caps (np.ndarray): Per-class exposure caps (curb_kappa * class size).
        stall (np.ndarray): Per-class non-improving refresh counts.
        stalled (np.ndarray): Per-class boolean stalled flags.
    """
    path = trainer.save_dir / "curb.csv"
    new = not path.exists()
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write("epoch,class,mean_diff,exposure,cap,stall_count,stalled\n")
        f.writelines(
            f"{epoch},{names[int(c)]},{mu[c]:.6f},{exposures[c]:.3f},{caps[c]:.3f},{stall[c]},{int(stalled[c])}\n"
            for c in range(mu.size)
        )


def _log_review(trainer, epoch: int, names, d, members, stalled) -> None:
    """Append each stalled class with its hardest images to save_dir/curb_review.txt for inspection.

    Args:
        trainer: The DetectionTrainer currently training.
        epoch (int): Current 1-based epoch number.
        names: Class names (list or dict keyed by class index).
        d (np.ndarray): EMA-smoothed per-image difficulty.
        members (list[np.ndarray]): Indices of training images containing each class.
        stalled (np.ndarray): Per-class boolean stalled flags.
    """
    im_files = trainer.train_loader.dataset.im_files
    path = trainer.save_dir / "curb_review.txt"
    with open(path, "a", encoding="utf-8") as f:
        for c in np.flatnonzero(stalled):
            m = members[c]
            top = m[np.argsort(-d[m])[:5]]
            f.write(f"epoch {epoch} stalled class {int(c)}:{names[int(c)]} mean_diff={d[m].mean():.4f}\n")
            f.writelines(f"    {im_files[i]} ({d[i]:.3f})\n" for i in top)


def curb_on_fit_epoch_end(trainer) -> None:
    """Refresh CURB sampling weights every ``curb_interval`` epochs unless training is ending.

    Args:
        trainer: The DetectionTrainer whose training epoch just finished.
    """
    epoch = trainer.epoch + 1
    if trainer.stop or epoch % trainer.args.curb_interval:
        return
    _refresh(trainer)
