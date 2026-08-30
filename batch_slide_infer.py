import importlib.metadata
_orig_read_text = importlib.metadata.PathDistribution.read_text
def _safe_read_text(self, name):
    try:
        return _orig_read_text(self, name)
    except UnicodeDecodeError:
        return None
importlib.metadata.PathDistribution.read_text = _safe_read_text

import argparse
import os
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ─── 滑窗核心函数（复用自 detect_server_slidewind.py） ───────────────────────

def compute_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def nms_merge(detections, iou_thresh=0.5):
    if not detections:
        return []
    detections = sorted(detections, key=lambda x: x.get("confidence", 0), reverse=True)
    keep = []
    while detections:
        current = detections.pop(0)
        merged_box = current["box"][:]
        remaining = []
        for det in detections:
            if det.get("code") != current.get("code"):
                remaining.append(det)
                continue
            iou = compute_iou(merged_box, det["box"])
            if iou >= iou_thresh:
                merged_box[0] = min(merged_box[0], det["box"][0])
                merged_box[1] = min(merged_box[1], det["box"][1])
                merged_box[2] = max(merged_box[2], det["box"][2])
                merged_box[3] = max(merged_box[3], det["box"][3])
                if det.get("confidence", 0) > current.get("confidence", 0):
                    current["confidence"] = det["confidence"]
            else:
                remaining.append(det)
        current["box"] = merged_box
        current["area"] = (merged_box[2] - merged_box[0]) * (merged_box[3] - merged_box[1])
        current["length"] = max(merged_box[2] - merged_box[0], merged_box[3] - merged_box[1])
        keep.append(current)
        detections = remaining
    return keep


def calculate_slide_positions(img_height, img_width, slide_rows, slide_cols, overlap_pixels):
    window_height = (img_height + (slide_rows - 1) * overlap_pixels) // slide_rows
    window_width = (img_width + (slide_cols - 1) * overlap_pixels) // slide_cols
    positions = []
    for r in range(slide_rows):
        for c in range(slide_cols):
            y1 = r * (window_height - overlap_pixels)
            x1 = c * (window_width - overlap_pixels)
            y2 = y1 + window_height
            x2 = x1 + window_width
            y1 = max(0, y1)
            x1 = max(0, x1)
            y2 = min(img_height, y2)
            x2 = min(img_width, x2)
            positions.append({"row": r, "col": c, "roi": (y1, x1, y2, x2)})
    return positions


def extract_window(img, roi, border_expand):
    y1, x1, y2, x2 = roi
    img_h, img_w = img.shape[:2]
    ext_y1 = y1 - border_expand
    ext_x1 = x1 - border_expand
    ext_y2 = y2 + border_expand
    ext_x2 = x2 + border_expand
    valid_y1 = max(0, ext_y1)
    valid_x1 = max(0, ext_x1)
    valid_y2 = min(img_h, ext_y2)
    valid_x2 = min(img_w, ext_x2)
    ext_h = ext_y2 - ext_y1
    ext_w = ext_x2 - ext_x1
    if len(img.shape) == 3:
        window = np.zeros((ext_h, ext_w, img.shape[2]), dtype=img.dtype)
    else:
        window = np.zeros((ext_h, ext_w), dtype=img.dtype)
    dst_y = valid_y1 - ext_y1
    dst_x = valid_x1 - ext_x1
    src_h = valid_y2 - valid_y1
    src_w = valid_x2 - valid_x1
    if src_h > 0 and src_w > 0:
        window[dst_y:dst_y + src_h, dst_x:dst_x + src_w] = img[valid_y1:valid_y2, valid_x1:valid_x2]
    return window, (ext_y1, ext_x1)


def map_boxes_to_original(results, offset, roi, border_expand, img_shape):
    offset_y, offset_x = offset
    img_h, img_w = img_shape[:2]
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = result.names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            orig_x1 = x1 + offset_x
            orig_y1 = y1 + offset_y
            orig_x2 = x2 + offset_x
            orig_y2 = y2 + offset_y
            clip_x1 = max(orig_x1, roi[1])
            clip_y1 = max(orig_y1, roi[0])
            clip_x2 = min(orig_x2, roi[3])
            clip_y2 = min(orig_y2, roi[2])
            if clip_x1 >= clip_x2 or clip_y1 >= clip_y2:
                continue
            detections.append({
                "code": class_name,
                "box": [int(orig_x1), int(orig_y1), int(orig_x2), int(orig_y2)],
                "confidence": confidence,
            })
    return detections


# ─── 单张大图滑窗推理 ──────────────────────────────────────────────────────

def slide_window_inference(model, img, slide_rows, slide_cols, overlap_pixels,
                           border_expand, imgsz, conf):
    img_height, img_width = img.shape[:2]
    positions = calculate_slide_positions(img_height, img_width, slide_rows, slide_cols, overlap_pixels)
    all_detections = []
    for pos in positions:
        roi = pos["roi"]
        window, offset = extract_window(img, roi, border_expand)
        results = model(window, conf=conf, imgsz=imgsz, verbose=False)
        detections = map_boxes_to_original(results, offset, roi, border_expand, img.shape)
        all_detections.extend(detections)
    merged = nms_merge(all_detections, iou_thresh=0.5)
    return merged


# ─── 绘制 HBB ──────────────────────────────────────────────────────────────

def draw_hbb(img, detections):
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        code = det["code"]
        conf = det["confidence"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{code} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 0, 255), -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return img


# ─── 柱状图 ────────────────────────────────────────────────────────────────

def plot_bar_chart(class_counts, save_path):
    if not class_counts:
        print("未检测到任何目标，跳过柱状图生成")
        return
    names = sorted(class_counts.keys())
    counts = [class_counts[n] for n in names]
    colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 6))
    bars = ax.bar(names, counts, color=colors, edgecolor="black", linewidth=0.5)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(c), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Detection Class Distribution", fontsize=14)
    ax.set_ylim(0, max(counts) * 1.15 if counts else 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"柱状图已保存: {save_path}")


# ─── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="滑动窗口批量 YOLO 推理")
    parser.add_argument("--data_dir", type=str, required=True, help="图片文件夹路径")
    parser.add_argument("--model", type=str, required=True, help="YOLO 模型权重路径")
    parser.add_argument("--slide_rows", type=int, default=2, help="滑窗行数")
    parser.add_argument("--slide_cols", type=int, default=3, help="滑窗列数")
    parser.add_argument("--overlap_pixels", type=int, default=32, help="相邻窗口重叠像素")
    parser.add_argument("--border_expand", type=int, default=0, help="窗口边界外扩像素")
    parser.add_argument("--imgsz", type=int, default=640, help="模型推理图像尺寸")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    args = parser.parse_args()

    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        print(f"错误: 文件夹不存在 - {data_dir}")
        sys.exit(1)

    # 收集图片
    images = sorted([f for f in os.listdir(data_dir)
                     if Path(f).suffix.lower() in IMAGE_EXTS])
    if not images:
        print("文件夹内没有找到图片文件")
        sys.exit(1)

    print(f"共找到 {len(images)} 张图片")
    print(f"滑窗参数: {args.slide_rows}x{args.slide_cols}, "
          f"overlap={args.overlap_pixels}px, border_expand={args.border_expand}px")
    print(f"模型: {args.model}, imgsz={args.imgsz}, conf={args.conf}")
    print("-" * 60)

    # 输出目录
    os.makedirs("assert", exist_ok=True)
    os.makedirs("assert/detail", exist_ok=True)

    # 加载模型
    model = YOLO(args.model, task="detect")
    class_counts = defaultdict(int)
    ok_count = 0

    for idx, img_name in enumerate(images, 1):
        img_path = os.path.join(data_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[{idx}/{len(images)}] {img_name} -> 无法读取，跳过")
            continue

        detections = slide_window_inference(
            model, img, args.slide_rows, args.slide_cols,
            args.overlap_pixels, args.border_expand, args.imgsz, args.conf
        )

        for det in detections:
            class_counts[det["code"]] += 1

        # 画 HBB
        vis = img.copy()
        draw_hbb(vis, detections)

        # 按类别分目录保存：无目标放 ok/，有目标按第一个类别放 {class}/
        if not detections:
            ok_count += 1
            save_dir = os.path.join("assert", "detail", "ok")
        else:
            first_code = sorted({d["code"] for d in detections})[0]
            save_dir = os.path.join("assert", "detail", first_code)
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, img_name), vis)

        defect_str = ", ".join(f"{k}:{v}" for k, v in
                               sorted(defaultdict(int, {d["code"]: 1 for d in detections}).items()))
        print(f"[{idx}/{len(images)}] {img_name} -> {len(detections)} defects [{defect_str}]")

    print("-" * 60)
    total = sum(class_counts.values())
    print(f"全部完成! 共检测到 {total} 个目标, {ok_count} 张无目标")
    for k, v in sorted(class_counts.items()):
        print(f"  {k}: {v}")

    # 柱状图
    plot_bar_chart(class_counts, os.path.join("assert", "class_distribution.png"))


if __name__ == "__main__":
    main()
