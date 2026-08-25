from ultralytics import YOLO
import argparse
import torch


def main():
    use_cuda = torch.cuda.is_available()
    default_device = "0" if use_cuda else "cpu"

    parser = argparse.ArgumentParser(description="YOLO26 Training Script")
    parser.add_argument("--model", type=str, default="./models/yolo26m.pt", help="model name e.g. yolo26n.pt, yolo26s.pt, yolo26m.pt, yolo26l.pt, yolo26x.pt")
    parser.add_argument("--data", type=str, default="coco8.yaml", help="dataset config e.g. coco8.yaml, coco.yaml, visdrone.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="image size")
    parser.add_argument("--device", type=str, default=default_device, help="device: 0, cpu, or 0,1,2,3")
    parser.add_argument("--workers", type=int, default=8, help="dataloader workers")
    parser.add_argument("--project", type=str, default="runs/detect", help="save results to project/name")
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
    parser.add_argument("--task", type=str, default="detect", choices=["detect", "segment", "classify", "pose", "obb"], help="task type")
    parser.add_argument("--plots", action="store_true", default=True, help="save plots and images")
    parser.add_argument("--save-period", type=int, default=-1, help="save checkpoint every N epochs")
    parser.add_argument("--cfg", type=str, default=None, help="path to custom config.yaml")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--cos-lr", action="store_true", help="cosine learning rate scheduler")
    parser.add_argument("--close-mosaic", type=int, default=10, help="disable mosaic for final N epochs")
    args = parser.parse_args()

    if not use_cuda:
        args.workers = min(args.workers, 4)

    pretrained = args.pretrained.lower() not in ("false", "0", "no")
    print(f"Using device: {args.device} (CUDA available: {use_cuda})")

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
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
        task=args.task,
        plots=args.plots,
        save_period=args.save_period,
        cfg=args.cfg,
        seed=args.seed,
        cos_lr=args.cos_lr,
        close_mosaic=args.close_mosaic,
    )
    print(f"Training complete. Results saved to: {results.save_dir}")


if __name__ == "__main__":
    main()
