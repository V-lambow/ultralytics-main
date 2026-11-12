import argparse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import colors

URL_PREFIXES = ("http://", "https://", "rtsp://", "rtmp://")


def parseArgs():
    use_cuda = torch.cuda.is_available()
    default_device = "0" if use_cuda else "cpu"

    parser = argparse.ArgumentParser(description="YOLO26 SAHI Sliced Prediction Script")
    parser.add_argument("--model", type=str, default="yolo26n.pt", help="model path e.g. yolo26n.pt, runs/detect/train/weights/best.pt")
    parser.add_argument("--source", type=str, default="ultralytics/assets/bus.jpg", help="image/video/dir/URL e.g. 0 for webcam, path/to/image.jpg, folder/")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--device", type=str, default=default_device, help="device: 0, cpu, or 0,1,2,3")
    parser.add_argument("--slice-height", type=int, default=512, help="slice height for sliced inference")
    parser.add_argument("--slice-width", type=int, default=512, help="slice width for sliced inference")
    parser.add_argument("--overlap-height-ratio", type=float, default=0.2, help="slice overlap ratio in height")
    parser.add_argument("--overlap-width-ratio", type=float, default=0.2, help="slice overlap ratio in width")
    parser.add_argument("--postprocess-type", type=str, default="GREEDYNMM", choices=["GREEDYNMM", "NMM", "NMS", "LSNMS"], help="slice merge postprocess type")
    parser.add_argument("--postprocess-match-metric", type=str, default="IOS", choices=["IOS", "IOU"], help="slice merge match metric")
    parser.add_argument("--postprocess-match-threshold", type=float, default=0.5, help="slice merge match threshold")
    parser.add_argument("--no-standard-pred", action="store_true", help="disable full image prediction on top of sliced predictions")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size for slice inference")
    parser.add_argument("--classes", type=int, nargs="+", default=None, help="filter by class: 0 or 0 2 3")
    parser.add_argument("--show", action="store_true", help="show results in window")
    parser.add_argument("--save", action="store_true", default=True, help="save results")
    parser.add_argument("--no-save", dest="save", action="store_false", help="do not save results")
    parser.add_argument("--save-txt", action="store_true", help="save detections as txt (cls conf x1 y1 x2 y2)")
    parser.add_argument("--project", type=str, default="runs/detect", help="save results to project/name")
    parser.add_argument("--name", type=str, default="predict_sahi", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok")
    parser.add_argument("--line-width", type=int, default=None, help="box line width")
    parser.add_argument("--hide-conf", action="store_true", help="hide confidence in results")
    return parser.parse_args()


def slicedPredict(model, image, args):
    results = get_sliced_prediction(
        image,
        model,
        slice_height=args.slice_height,
        slice_width=args.slice_width,
        overlap_height_ratio=args.overlap_height_ratio,
        overlap_width_ratio=args.overlap_width_ratio,
        perform_standard_pred=not args.no_standard_pred,
        postprocess_type=args.postprocess_type,
        postprocess_match_metric=args.postprocess_match_metric,
        postprocess_match_threshold=args.postprocess_match_threshold,
        confidence_threshold=args.conf,
        batch_size=args.batch_size,
        verbose=0,
    )
    if args.classes is not None:
        keep = set(args.classes)
        results.object_prediction_list = [p for p in results.object_prediction_list if int(p.category.id) in keep]
    return results


def writeTxt(path, results):
    lines = []
    for p in results.object_prediction_list:
        x1, y1, x2, y2 = p.bbox.to_xyxy()
        lines.append(f"{int(p.category.id)} {p.score.value:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}")
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""))


def printSummary(label, results):
    print(f"Image: {label}, Detections: {len(results.object_prediction_list)}")
    for p in results.object_prediction_list:
        xyxy = [round(v, 1) for v in p.bbox.to_xyxy()]
        print(f"  {p.category.name}: {p.score.value:.2f} {xyxy}")


def drawPredictions(frame, results, args):
    line_width = args.line_width or 2
    for p in results.object_prediction_list:
        x1, y1, x2, y2 = (int(v) for v in p.bbox.to_xyxy())
        color = tuple(int(c) for c in colors(int(p.category.id), bgr=True))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_width)
        text = p.category.name if args.hide_conf else f"{p.category.name} {p.score.value:.2f}"
        cv2.putText(frame, text, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame


def downloadImage(url):
    with urllib.request.urlopen(url) as r:
        data = np.frombuffer(r.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode image from URL: {url}")
    return img[..., ::-1]


def processImage(source, label, stem, model, args, save_dir):
    results = slicedPredict(model, source, args)
    printSummary(label, results)
    if save_dir is None:
        return
    if args.save:
        results.export_visuals(export_dir=str(save_dir), file_name=stem, hide_conf=args.hide_conf, rect_th=args.line_width)
    if args.save_txt:
        writeTxt(save_dir / f"{stem}.txt", results)


def processVideo(source, label, stem, model, args, save_dir):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video source: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if save_dir is not None and args.save:
        writer = cv2.VideoWriter(str(save_dir / f"{stem}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    txt_path = save_dir / f"{stem}.txt" if save_dir is not None and args.save_txt else None
    txt_lines = []
    frame_idx = total = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = slicedPredict(model, frame[..., ::-1], args)
        total += len(results.object_prediction_list)
        if writer is not None or args.show:
            drawPredictions(frame, results, args)
        if writer is not None:
            writer.write(frame)
        if txt_path is not None:
            for p in results.object_prediction_list:
                x1, y1, x2, y2 = p.bbox.to_xyxy()
                txt_lines.append(f"{frame_idx} {int(p.category.id)} {p.score.value:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}")
        if args.show:
            cv2.imshow("YOLO26 SAHI", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if txt_path is not None:
        txt_path.write_text("\n".join(txt_lines) + ("\n" if txt_lines else ""))
    if args.show:
        cv2.destroyAllWindows()
    print(f"Video: {label}, Frames: {frame_idx}, Detections: {total}")


def main():
    args = parseArgs()
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=args.model,
        device=args.device,
        confidence_threshold=args.conf,
        image_size=args.imgsz,
    )

    save_dir = None
    if args.save or args.save_txt:
        save_dir = increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok)
        save_dir.mkdir(parents=True, exist_ok=True)

    source = args.source
    if source.isdigit():
        processVideo(int(source), "webcam", "webcam", model, args, save_dir)
    elif source.lower().startswith(URL_PREFIXES):
        url_path = Path(source.split("?")[0])
        if url_path.suffix[1:].lower() in IMG_FORMATS:
            processImage(downloadImage(source), source, url_path.stem or "url_image", model, args, save_dir)
        else:
            processVideo(source, source, url_path.stem or "url_video", model, args, save_dir)
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Source not found: {source}")
        if path.is_dir():
            for f in sorted(p for p in path.iterdir() if p.is_file()):
                ext = f.suffix[1:].lower()
                if ext in IMG_FORMATS:
                    processImage(str(f), str(f), f.stem, model, args, save_dir)
                elif ext in VID_FORMATS:
                    processVideo(str(f), str(f), f.stem, model, args, save_dir)
        elif path.suffix[1:].lower() in VID_FORMATS:
            processVideo(str(path), str(path), path.stem, model, args, save_dir)
        elif path.suffix[1:].lower() in IMG_FORMATS:
            processImage(str(path), str(path), path.stem, model, args, save_dir)
        else:
            raise ValueError(f"Unsupported source type: {source}")

    if save_dir is not None:
        print(f"\nResults saved to: {save_dir}")


if __name__ == "__main__":
    main()
