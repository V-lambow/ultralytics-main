import argparse
import json
from pathlib import Path

import torch
import yaml
from PIL import Image
from sahi.slicing import slice_coco
from sahi.utils.coco import Coco

from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset, img2label_paths


def parseArgs():
    use_cuda = torch.cuda.is_available()
    default_device = "0" if use_cuda else "cpu"

    parser = argparse.ArgumentParser(description="YOLO26 SAHI Sliced Training Script (detect bbox labels only)")
    parser.add_argument("--model", type=str, default="./models/yolov8m.pt", help="model name e.g. yolo26n.pt, yolo26s.pt, yolo26m.pt")
    parser.add_argument("--data", type=str, default="coco8.yaml", help="YOLO dataset config e.g. coco8.yaml, path/to/data.yaml")
    parser.add_argument("--output", type=str, default="runs/sahi_data", help="sliced dataset output directory")
    parser.add_argument("--slice-height", type=int, default=512, help="slice height")
    parser.add_argument("--slice-width", type=int, default=512, help="slice width")
    parser.add_argument("--overlap-height-ratio", type=float, default=0.2, help="slice overlap ratio in height")
    parser.add_argument("--overlap-width-ratio", type=float, default=0.2, help="slice overlap ratio in width")
    parser.add_argument("--min-area-ratio", type=float, default=0.1, help="drop annotations smaller than this ratio of their original area")
    parser.add_argument("--skip-slice", action="store_true", help="reuse existing sliced dataset in --output")
    parser.add_argument("--symlink", action="store_true", help="symlink images instead of copying them into the YOLO dataset")
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="image size")
    parser.add_argument("--device", type=str, default=default_device, help="device: 0, cpu, or 0,1,2,3")
    parser.add_argument("--workers", type=int, default=8, help="dataloader workers")
    parser.add_argument("--project", type=str, default="", help="save results to project/name")
    parser.add_argument("--name", type=str, default="train", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--pretrained", type=str, default="True", help="use pretrained weights")
    parser.add_argument("--optimizer", type=str, default="auto", help="optimizer: SGD, Adam, AdamW, auto")
    parser.add_argument("--lr0", type=float, default=0.01, help="initial learning rate")
    parser.add_argument("--patience", type=int, default=100, help="early stopping patience")
    parser.add_argument("--resume", action="store_true", help="resume training from last checkpoint")
    parser.add_argument("--amp", action="store_true", default=use_cuda, help="Automatic Mixed Precision (auto-off on CPU)")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="disable AMP")
    parser.add_argument("--cache", type=str, default="False", help="cache images: True, ram, disk, or False")
    parser.add_argument("--plots", action="store_true", default=True, help="save plots and images")
    parser.add_argument("--save-period", type=int, default=-1, help="save checkpoint every N epochs")
    parser.add_argument("--cfg", type=str, default=None, help="path to custom config.yaml")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--cos-lr", action="store_true", help="cosine learning rate scheduler")
    parser.add_argument("--close-mosaic", type=int, default=10, help="disable mosaic for final N epochs")
    parser.add_argument("--curb", action="store_true", help="enable CURB difficulty-based train resampling (detect only)")
    parser.add_argument("--curb-interval", type=int, default=25, help="epochs between CURB difficulty refreshes")
    parser.add_argument("--curb-kappa", type=float, default=2.0, help="CURB per-class exposure cap multiple")
    parser.add_argument("--curb-ema", type=float, default=0.3, help="CURB EMA weight on the newest difficulty observation")
    parser.add_argument("--curb-stall", type=int, default=3, help="CURB refreshes without improvement before stop-loss")
    return parser.parse_args()


def collectImages(split_path):
    files = []
    for p in split_path if isinstance(split_path, list) else [split_path]:
        p = Path(p)
        if p.is_dir():
            files.extend(f for f in p.rglob("*") if f.is_file() and f.suffix[1:].lower() in IMG_FORMATS)
        elif p.is_file() and p.suffix.lower() == ".txt":
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                q = Path(line)
                if not q.is_absolute():
                    q = p.parent / q
                if q.is_dir():
                    files.extend(f for f in q.rglob("*") if f.is_file() and f.suffix[1:].lower() in IMG_FORMATS)
                elif q.is_file() and q.suffix[1:].lower() in IMG_FORMATS:
                    files.append(q)
    return sorted(files)


def normalizeNames(names):
    if isinstance(names, dict):
        mapping = {int(k): str(v) for k, v in names.items()}
        ids = sorted(mapping)
        if ids != list(range(len(ids))):
            raise ValueError(f"Class ids must be contiguous from 0, got: {ids}")
        return [mapping[i] for i in ids]
    return [str(n) for n in names]


def yoloToCoco(image_paths, names, out_json):
    images, annotations = [], []
    missing = 0
    degenerate = 0
    ann_id = 0
    for img_id, img_path in enumerate(image_paths, start=1):
        with Image.open(img_path) as im:
            width, height = im.size
        images.append({"id": img_id, "file_name": str(img_path), "width": width, "height": height})
        label_path = Path(img2label_paths([str(img_path)])[0])
        if not label_path.is_file():
            missing += 1
            continue
        for line_no, line in enumerate(label_path.read_text().splitlines(), start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 5:
                raise ValueError(f"Only xywh bbox labels are supported, got {len(parts) - 1} values: {label_path}:{line_no}")
            cls_id = int(float(parts[0]))
            if not 0 <= cls_id < len(names):
                raise ValueError(f"Class id {cls_id} out of range [0, {len(names) - 1}]: {label_path}:{line_no}")
            cx, cy, bw, bh = (float(v) for v in parts[1:])
            bw_px, bh_px = bw * width, bh * height
            if bw_px <= 0 or bh_px <= 0:
                degenerate += 1
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls_id,
                    "bbox": [cx * width - bw_px / 2, cy * height - bh_px / 2, bw_px, bh_px],
                    "area": bw_px * bh_px,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": n} for i, n in enumerate(names)],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(coco))
    print(
        f"Converted {len(images)} images, {len(annotations)} annotations "
        f"({missing} without labels, {degenerate} degenerate skipped) -> {out_json}"
    )
    return out_json


def sliceDataset(args, work):
    data = check_det_dataset(args.data)
    names = normalizeNames(data["names"])
    print(f"Dataset: {args.data}, classes: {len(names)}, train: {data['train']}, val: {data['val']}")

    for split in ("train", "val"):
        images = collectImages(data[split])
        if not images:
            raise FileNotFoundError(f"No images found for {split} split: {data[split]}")
        split_json = yoloToCoco(images, names, work / f"{split}_source.json")
        sliced_dir = work / split
        _, sliced_json = slice_coco(
            coco_annotation_file_path=str(split_json),
            image_dir="",
            output_coco_annotation_file_name=split,
            output_dir=str(sliced_dir),
            slice_height=args.slice_height,
            slice_width=args.slice_width,
            overlap_height_ratio=args.overlap_height_ratio,
            overlap_width_ratio=args.overlap_width_ratio,
            min_area_ratio=args.min_area_ratio,
            ignore_negative_samples=False,
            verbose=False,
        )
        coco = Coco.from_coco_dict_or_path(coco_dict_or_path=sliced_json, image_dir=str(sliced_dir))
        coco.export_as_yolo(
            output_dir=str(work / "yolo"),
            train_split_rate=1.0 if split == "train" else 0.0,
            numpy_seed=args.seed,
            disable_symlink=not args.symlink,
        )
        print(f"[{split}] {len(images)} images sliced -> {work / 'yolo' / split}")

    data_yml = work / "data.yml"
    data_yml.write_text(
        yaml.safe_dump(
            {
                "train": str((work / "yolo" / "train").resolve()),
                "val": str((work / "yolo" / "val").resolve()),
                "nc": len(names),
                "names": names,
            },
            sort_keys=False,
            allow_unicode=True,
        )
    )
    print(f"Sliced dataset config: {data_yml}")
    return data_yml


def main():
    args = parseArgs()
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        args.workers = min(args.workers, 4)
    pretrained = args.pretrained.lower() not in ("false", "0", "no")
    print(f"Using device: {args.device} (CUDA available: {use_cuda})")

    work = Path(args.output)
    data_yml = work / "data.yml"
    if args.skip_slice:
        if not data_yml.is_file():
            raise FileNotFoundError(f"No sliced dataset at {data_yml}, run once without --skip-slice first")
        print(f"Reusing sliced dataset: {data_yml}")
    else:
        if work.exists() and any(work.iterdir()):
            raise FileExistsError(f"{work} is not empty, use --skip-slice to reuse it or choose another --output")
        data_yml = sliceDataset(args, work)

    model = YOLO(args.model)
    results = model.train(
        deterministic=False,
        data=str(data_yml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        pretrained=pretrained,
        optimizer=args.optimizer,
        lr0=args.lr0,
        patience=args.patience,
        resume=args.resume,
        amp=args.amp,
        cache=args.cache if args.cache.lower() not in ("false", "0", "no") else False,
        plots=args.plots,
        save_period=args.save_period,
        cfg=args.cfg,
        seed=args.seed,
        cos_lr=args.cos_lr,
        close_mosaic=args.close_mosaic,
        curb=args.curb,
        curb_interval=args.curb_interval,
        curb_kappa=args.curb_kappa,
        curb_ema=args.curb_ema,
        curb_stall=args.curb_stall,
    )
    print(f"Training complete. Results saved to: {results.save_dir}")


if __name__ == "__main__":
    main()
