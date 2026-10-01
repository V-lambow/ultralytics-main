import argparse

import torch

from ultralytics import YOLO


def main():
    use_cuda = torch.cuda.is_available()
    default_device = "0" if use_cuda else "cpu"

    parser = argparse.ArgumentParser(description="YOLO26 Prediction Script")
    parser.add_argument(
        "--model", type=str, default="yolo26n.pt", help="model path e.g. yolo26n.pt, runs/detect/train/weights/best.pt"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="ultralytics/assets/bus.jpg",
        help="image/video/dir/URL/stream e.g. 0 for webcam, path/to/image.jpg, folder/",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS")
    parser.add_argument("--device", type=str, default=default_device, help="device: 0, cpu, or 0,1,2,3")
    parser.add_argument("--show", action="store_true", help="show results in window")
    parser.add_argument("--save", action="store_true", default=True, help="save results")
    parser.add_argument("--save-txt", action="store_true", help="save results as .txt files")
    parser.add_argument("--save-conf", action="store_true", help="save confidence scores")
    parser.add_argument("--save-crop", action="store_true", help="save cropped predictions")
    parser.add_argument("--project", type=str, default="runs/detect", help="save results to project/name")
    parser.add_argument("--name", type=str, default="predict", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok")
    parser.add_argument("--classes", type=int, nargs="+", default=None, help="filter by class: 0 or 0 2 3")
    parser.add_argument("--augment", action="store_true", help="test-time augmentation")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--max-det", type=int, default=300, help="max detections per image")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame stride")
    parser.add_argument("--line-width", type=int, default=None, help="box line width")
    parser.add_argument("--show-labels", action="store_true", default=True, help="show labels")
    parser.add_argument("--show-conf", action="store_true", default=True, help="show confidence")
    parser.add_argument("--show-boxes", action="store_true", default=True, help="show boxes")
    parser.add_argument("--retina-masks", action="store_true", help="high-res segmentation masks")
    parser.add_argument(
        "--task", type=str, default="detect", choices=["detect", "segment", "classify", "pose", "obb"], help="task type"
    )
    args = parser.parse_args()

    model = YOLO(args.model)
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        show=args.show,
        save=args.save,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        save_crop=args.save_crop,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        classes=args.classes,
        augment=args.augment,
        agnostic_nms=args.agnostic_nms,
        max_det=args.max_det,
        vid_stride=args.vid_stride,
        line_width=args.line_width,
        show_labels=args.show_labels,
        show_conf=args.show_conf,
        show_boxes=args.show_boxes,
        retina_masks=args.retina_masks,
        task=args.task,
    )

    for r in results:
        print(f"Image: {r.path}, Detections: {len(r.boxes) if r.boxes is not None else 0}")
        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                cls_name = r.names[cls_id] if r.names else str(cls_id)
                print(f"  {cls_name}: {conf:.2f} {xyxy}")
    print(f"\nResults saved to: {results[0].save_dir if results else 'N/A'}")


if __name__ == "__main__":
    main()
