import importlib.metadata

_orig_read_text = importlib.metadata.PathDistribution.read_text


def _safe_read_text(self, name):
    try:
        return _orig_read_text(self, name)
    except UnicodeDecodeError:
        return None


importlib.metadata.PathDistribution.read_text = _safe_read_text

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import cv2
import numpy as np
import tornado.ioloop
import tornado.web
from loguru import logger

from ultralytics import YOLO


def create_polygon_mask(img_shape, points):
    """根据多边形顶点创建mask points: list of [x, y] 坐标 返回: 单通道mask，多边形内为255，其余为0.
    """
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    return mask


def apply_region_mask(img, a_config):
    """根据配置中的inner/outer对图像做mask处理 - inner: 只保留inner多边形区域，其余置0 - outer: 将outer多边形区域置0，保留其余部分 - 无inner/outer: 返回原图.
    """
    inner_cfg = a_config.get("inner")
    outer_cfg = a_config.get("outer")

    if inner_cfg:
        points = inner_cfg.get("points", [])
        if points:
            mask = create_polygon_mask(img.shape, points)
            return cv2.bitwise_and(img, img, mask=mask)

    if outer_cfg:
        points = outer_cfg.get("points", [])
        if points:
            mask = create_polygon_mask(img.shape, points)
            mask_inv = cv2.bitwise_not(mask)
            return cv2.bitwise_and(img, img, mask=mask_inv)

    return img


def compute_iou(box_a, box_b):
    """计算两个框的IoU，box格式为[x1, y1, x2, y2]."""
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
    """对检测结果进行NMS合并，处理重叠区域的重复检测 detections: list of dict，每个dict包含 box, confidence, code 等字段 iou_thresh: IoU阈值，超过此值的框会被合并
    返回合并后的检测结果列表.
    """
    if not detections:
        return []

    # 按置信度降序排序
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
                # 合并两个框：取并集
                merged_box[0] = min(merged_box[0], det["box"][0])
                merged_box[1] = min(merged_box[1], det["box"][1])
                merged_box[2] = max(merged_box[2], det["box"][2])
                merged_box[3] = max(merged_box[3], det["box"][3])
                # 保留置信度更高的
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


# 配置日志
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)

# 配置按日保存的日志文件
logger.add(
    os.path.join(log_dir, "{time:YYYY-MM-DD}.log"),
    rotation="00:00",  # 每天00:00创建新文件
    retention="7 days",  # 保留7天
    compression="zip",  # 压缩旧日志
    level="INFO",
)

# # 配置控制台输出
# logger.add(
#     sink=lambda msg: print(msg, end=""),
#     level="INFO"
# )


class ImageDefectHandler(tornado.web.RequestHandler):
    ultralytics_dir = os.path.join(os.getcwd(), ".ultralytics")
    os.makedirs(ultralytics_dir, exist_ok=True)
    os.environ["ULTRALYTICS_SETTINGS"] = os.path.join(ultralytics_dir, "settings.json")
    os.environ["ULTRALYTICS_CACHE"] = os.path.join(ultralytics_dir, "cache")
    model = None

    def initialize(self, template_dir, mask_dir, config_path):
        self.template_dir = template_dir
        self.mask_dir = mask_dir
        self.config_path = config_path
        self.detect_config = self.load_config()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.model_lock = threading.Lock()

        model_path = self.detect_config.get("model_path", "models/yolo26x.pt")
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.getcwd(), model_path)
        ImageDefectHandler.model = YOLO(model_path, task="detect")
        logger.info(f"Loaded model: {model_path}")

    def load_config(self):
        """加载detect_item.json配置文件."""
        if os.path.exists(self.config_path):
            with open(self.config_path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    async def post(self):
        try:
            data = json.loads(self.request.body)

            job_id = data.get("job_id", "")
            sample_id = data.get("sample_id", "")
            pose_id = data.get("pose_id", "")
            file_names = data.get("file_names", [])
            relative_dir = data.get("relative_dir", "")

            logger.info(
                f"Received request: job_id={job_id}, sample_id={sample_id}, pose_id={pose_id}, file_names={file_names}"
            )

            if not file_names:
                raise ValueError("file_names is required")

            image_path = os.path.join(relative_dir, file_names[0])

            # 从配置文件读取滑窗参数
            use_slide_window = self.detect_config.get("use_slide_window", False)
            slide_rows = int(self.detect_config.get("slide_rows", 0))
            slide_cols = int(self.detect_config.get("slide_cols", 0))
            overlap_pixels = int(self.detect_config.get("overlap_pixels", 0))
            border_expand = int(self.detect_config.get("border_expand", 0))
            self.imgsz = int(self.detect_config.get("imgsz", 640))
            self.device = self.detect_config.get("device", "cpu")

            if use_slide_window:
                logger.info(
                    f"Using slide window mode: {slide_rows}x{slide_cols}, "
                    f"overlap={overlap_pixels}px, border_expand={border_expand}px"
                )
                results = await self.process_image_slide_window(
                    image_path, pose_id, sample_id, slide_rows, slide_cols, overlap_pixels, border_expand
                )
            else:
                logger.info("Using direct crop mode (slide_window disabled)")
                results = await self.process_image_slide_window(
                    image_path, pose_id, sample_id, slide_rows, slide_cols, overlap_pixels, border_expand
                )

            # 如果有检测结果，异步绘制并保存，不阻塞主事件循环
            if results:
                await tornado.ioloop.IOLoop.current().run_in_executor(
                    self.executor, self.draw_defects, image_path, results, sample_id
                )

            response = {
                "product_type": "",
                "job_id": job_id,
                "pose_id": pose_id,
                "results": results,
                "file_names": file_names,
            }

            logger.info(f"Processing completed, found {len(results)} defects")

            # 更改响应格式为 {"error_code": 0, "error_msg": 'OK','data':response}
            self.write({"error_code": 0, "error_msg": "OK", "data": response})

        except Exception as e:
            logger.error(f"Error processing request: {e!s}")
            self.set_status(500)
            # 错误时返回 {"error_code": 999, "error_msg": 'False'}
            self.write({"error_code": 999, "error_msg": "False"})

    def run_yolo_inference(self, img, imgsz=640):
        """执行YOLO模型推理，确保线程安全."""
        try:
            with self.model_lock:
                # 执行推理
                results = self.model(
                    img,
                    conf=0.25,  # 置信度阈值
                    iou=0.45,  # IOU阈值
                    imgsz=imgsz,  # 推理图像尺寸
                    device=self.device,
                    verbose=False,
                )
                return results
        except Exception as e:
            logger.error(f"YOLO inference error: {e!s}")
            raise

    def calculate_slide_positions(self, img_height, img_width, slide_rows, slide_cols, overlap_pixels):
        """计算滑窗在原图上的位置坐标.

        Args:
            img_height: 图片高度
            img_width: 图片宽度
            slide_rows: 滑窗行数
            slide_cols: 滑窗列数
            overlap_pixels: 相邻窗口的交叠像素数

        Returns:
            list of dict，每个dict包含:
                - row: 行索引
                - col: 列索引
                - roi: (y1, x1, y2, x2) 在原图上的坐标（无外扩）
        """
        # 计算每个窗口的基础尺寸（不含外扩）
        # 总尺寸 = 窗口数 * 窗口尺寸 - (窗口数 - 1) * 交叠
        # 窗口尺寸 = (总尺寸 + (窗口数 - 1) * 交叠) / 窗口数
        window_height = (img_height + (slide_rows - 1) * overlap_pixels) // slide_rows
        window_width = (img_width + (slide_cols - 1) * overlap_pixels) // slide_cols

        positions = []
        for r in range(slide_rows):
            for c in range(slide_cols):
                # 计算窗口左上角坐标
                y1 = r * (window_height - overlap_pixels)
                x1 = c * (window_width - overlap_pixels)
                y2 = y1 + window_height
                x2 = x1 + window_width

                # 裁剪到图片边界内
                y1 = max(0, y1)
                x1 = max(0, x1)
                y2 = min(img_height, y2)
                x2 = min(img_width, x2)

                positions.append({"row": r, "col": c, "roi": (y1, x1, y2, x2)})

        return positions

    def extract_window(self, img, roi, border_expand):
        """从原图提取窗口，支持边框外扩/内陷.

        Args:
            img: 原图 (numpy array)
            roi: (y1, x1, y2, x2) 窗口在原图上的坐标
            border_expand: 边框外扩像素数
                          - 正数: 向外扩展，扩展区域填充灰度0（黑色）
                          - 负数: 向内收缩，只取窗口内部区域
                          - 0: 不做调整

        Returns:
            window: 提取的窗口图像 (numpy array)
            offset: (offset_y, offset_x) 窗口左上角在原图上的偏移量 用于后续将检测坐标映射回原图
        """
        y1, x1, y2, x2 = roi
        img_h, img_w = img.shape[:2]

        # 应用边框外扩/内陷
        # border_expand > 0: 向外扩展
        # border_expand < 0: 向内收缩
        ext_y1 = y1 - border_expand
        ext_x1 = x1 - border_expand
        ext_y2 = y2 + border_expand
        ext_x2 = x2 + border_expand

        # 记录实际提取的区域在原图上的有效范围
        valid_y1 = max(0, ext_y1)
        valid_x1 = max(0, ext_x1)
        valid_y2 = min(img_h, ext_y2)
        valid_x2 = min(img_w, ext_x2)

        # 计算扩展后的窗口尺寸
        ext_h = ext_y2 - ext_y1
        ext_w = ext_x2 - ext_x1

        # 创建窗口画布（扩展区域填充灰度0 = 黑色）
        if len(img.shape) == 3:
            window = np.zeros((ext_h, ext_w, img.shape[2]), dtype=img.dtype)
        else:
            window = np.zeros((ext_h, ext_w), dtype=img.dtype)

        # 将原图有效区域复制到窗口的对应位置
        # 源区域在原图中的位置: [valid_y1:valid_y2, valid_x1:valid_x2]
        # 目标位置在窗口中的偏移: (valid_y1 - ext_y1, valid_x1 - ext_x1)
        dst_y = valid_y1 - ext_y1
        dst_x = valid_x1 - ext_x1
        src_h = valid_y2 - valid_y1
        src_w = valid_x2 - valid_x1

        if src_h > 0 and src_w > 0:
            window[dst_y : dst_y + src_h, dst_x : dst_x + src_w] = img[valid_y1:valid_y2, valid_x1:valid_x2]

        # 窗口左上角在原图上的偏移量（用于坐标映射）
        offset_y = ext_y1
        offset_x = ext_x1

        return window, (offset_y, offset_x)

    def map_boxes_to_original(self, results, offset, roi, border_expand, img_shape):
        """将检测框坐标从窗口坐标系映射回原图坐标系.

        Args:
            results: YOLO推理结果
            offset: (offset_y, offset_x) 窗口左上角在原图上的偏移
            roi: (y1, x1, y2, x2) 原始窗口（无外扩）在原图上的坐标
            border_expand: 边框外扩像素数
            img_shape: 原图尺寸 (H, W)

        Returns:
            list of dict，每个dict包含 box, confidence, code 等字段
        """
        offset_y, offset_x = offset
        _img_h, _img_w = img_shape[:2]
        detections = []

        # 原始窗口（无外扩）在扩展窗口中的位置
        # 由于外扩是向外扩展，原始窗口在扩展窗口中的起始位置是 border_expand
        border_expand + (roi[2] - roi[0])
        border_expand + (roi[3] - roi[1])

        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = result.names.get(cls_id, str(cls_id))

                # 获取框坐标（xyxy格式）
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # 转换到原图坐标系
                orig_x1 = x1 + offset_x
                orig_y1 = y1 + offset_y
                orig_x2 = x2 + offset_x
                orig_y2 = y2 + offset_y

                # 裁剪到原始窗口范围内（去除外扩部分的检测结果）
                # 只保留在原始窗口（无外扩）内的部分
                clip_x1 = max(orig_x1, roi[1])
                clip_y1 = max(orig_y1, roi[0])
                clip_x2 = min(orig_x2, roi[3])
                clip_y2 = min(orig_y2, roi[2])

                # 如果框完全在原始窗口外，跳过
                if clip_x1 >= clip_x2 or clip_y1 >= clip_y2:
                    continue

                # 对于跨越边界的框，使用裁剪后的坐标
                # 但保留原始检测框用于NMS合并
                detection = {
                    "code": class_name,
                    "box": [int(orig_x1), int(orig_y1), int(orig_x2), int(orig_y2)],
                    "clip_box": [int(clip_x1), int(clip_y1), int(clip_x2), int(clip_y2)],
                    "confidence": confidence,
                    "area": (clip_x2 - clip_x1) * (clip_y2 - clip_y1),
                    "length": max(clip_x2 - clip_x1, clip_y2 - clip_y1),
                }
                detections.append(detection)

        return detections

    def sort_detections_by_position(self, detections, grid_rows=None, grid_cols=None):
        """按位置排序检测结果（先按行再按列，即从上到下、从左到右）.

        Args:
            detections: 检测结果列表，每个dict包含 box 字段
            grid_rows: 网格行数（可选，用于精确排序）
            grid_cols: 网格列数（可选，用于精确排序）

        Returns:
            排序后的检测结果列表
        """
        if not detections:
            return detections

        # 按框的中心点y坐标排序，相同y再按x排序
        def sort_key(det):
            box = det["box"]
            center_y = (box[1] + box[3]) / 2
            center_x = (box[0] + box[2]) / 2
            return (center_y, center_x)

        return sorted(detections, key=sort_key)

    async def process_image_slide_window(
        self, image_path, pose_id, sample_id, slide_rows, slide_cols, overlap_pixels, border_expand
    ):
        """基于滑窗的YOLO目标检测推理方法.

        Args:
            image_path: 图片路径
            pose_id: 姿态ID
            sample_id: 样本ID
            slide_rows: 滑窗行数
            slide_cols: 滑窗列数
            overlap_pixels: 相邻窗口的交叠像素数
            border_expand: 边框外扩像素数（正数外扩，负数内陷，默认0）

        Returns:
            list of dict，检测结果列表
        """
        try:
            # 读取待检测图片
            img = cv2.imread(image_path)

            if img is None:
                raise FileNotFoundError(f"Image not found: {image_path}")

            # 获取当前pose_id对应的配置
            pose_config = self.detect_config.get(pose_id, {})
            if not pose_config:
                raise ValueError(f"No configuration found for pose_id: {pose_id}")

            img_height, img_width = img.shape[:2]
            logger.info(
                f"Image size: {img_width}x{img_height}, slide: {slide_rows}x{slide_cols}, "
                f"overlap: {overlap_pixels}px, border_expand: {border_expand}px"
            )

            # 计算滑窗位置
            positions = self.calculate_slide_positions(img_height, img_width, slide_rows, slide_cols, overlap_pixels)
            logger.info(f"Generated {len(positions)} slide windows")

            all_detections = []

            # 对每个区域配置进行滑窗检测
            for a_config in pose_config.values():
                threshold = a_config.get("threshold", 0.25)

                # 应用polygon mask（inner保留区域，outer排除区域）
                masked_img = apply_region_mask(img, a_config)

                # 对整张图片滑窗
                slide_tasks = []
                for pos in positions:
                    slide_tasks.append((masked_img, pos, threshold))

                # 并行执行推理
                for window_img, pos, thr in slide_tasks:
                    roi = pos["roi"]

                    # 提取窗口（含外扩处理）
                    window, offset = self.extract_window(window_img, roi, border_expand)

                    # 执行推理
                    results = await tornado.ioloop.IOLoop.current().run_in_executor(
                        self.executor, self.run_yolo_inference, window, self.imgsz
                    )

                    # 映射坐标回原图
                    detections = self.map_boxes_to_original(results, offset, roi, border_expand, img.shape)

                    # 过滤置信度低于阈值的检测结果
                    detections = [d for d in detections if d["confidence"] > thr]

                    # 标记检测结果所属的窗口位置（用于后续排序）
                    for det in detections:
                        det["slide_row"] = pos["row"]
                        det["slide_col"] = pos["col"]

                    all_detections.extend(detections)

            # 合并重叠区域的重复检测
            logger.info(f"Total raw detections: {len(all_detections)}")
            merged_detections = nms_merge(all_detections, iou_thresh=0.5)
            logger.info(f"After NMS merge: {len(merged_detections)} detections")

            # 按位置排序（从上到下、从左到右）
            sorted_detections = self.sort_detections_by_position(merged_detections)

            # 格式化输出结果
            final_results = []
            for det in sorted_detections:
                result = {
                    "code": det["code"],
                    "box": det["box"],
                    "area": det["area"],
                    "length": det["length"],
                    "confidence": det["confidence"],
                }
                final_results.append(result)

            logger.info(f"YOLO slide-window detection completed, found {len(final_results)} defects")
            return final_results

        except Exception as e:
            logger.warning(f"Error in process_image_slide_window: {e!s}")
            raise

    def draw_defects(self, image_path, defects, sample_id="", output_dir="res"):
        """绘制检测结果到res文件夹，按年/月/日/Sxxxxxxx编号的文件夹结构保存."""
        # 获取当前日期
        now = datetime.now()
        year = now.strftime("%Y")
        month = now.strftime("%m")
        day = now.strftime("%d")

        # 创建文件夹结构：res/年/月/日/Sxxxxxxx
        save_dir = os.path.join(output_dir, year, month, day, sample_id)
        os.makedirs(save_dir, exist_ok=True)

        # 读取原始图片
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Cannot read image: {image_path}")
            return None

        # 绘制每个缺陷
        for defect in defects:
            box = defect.get("box", [])
            code = defect.get("code", "unknown")
            confidence = defect.get("confidence", 0.0)

            if len(box) == 4:
                x1, y1, x2, y2 = box
                # 绘制矩形框
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # 添加标签和分数
                label = f"{code}: {confidence:.2f}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 生成输出文件名
        base_name = os.path.basename(image_path)
        output_path = os.path.join(save_dir, f"detected_{base_name}")

        # 保存绘制结果
        cv2.imwrite(output_path, img)
        logger.info(f"Detection results saved to: {output_path}")

        return output_path


def make_app(template_dir, mask_dir, config_path):
    return tornado.web.Application(
        [
            (
                r"/industry/image_defect",
                ImageDefectHandler,
                {"template_dir": template_dir, "mask_dir": mask_dir, "config_path": config_path},
            ),
        ]
    )


if __name__ == "__main__":
    template_dir = "templates"
    mask_dir = "masks"
    config_path = "config\\assemble_detect_item.json"

    os.makedirs(template_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    # 从配置文件读取端口号，未配置则默认8000
    with open(config_path, encoding="utf-8") as f:
        server_config = json.load(f)
    port = int(server_config.get("port", 8000))

    app = make_app(template_dir, mask_dir, config_path)
    app.listen(port)
    logger.info(f"Server is running on port {port}")
    logger.info(f"Template directory: {os.path.abspath(template_dir)}")
    logger.info(f"Mask directory: {os.path.abspath(mask_dir)}")
    logger.info(f"Config file: {os.path.abspath(config_path)}")
    logger.info(f"Log directory: {os.path.abspath(log_dir)}")
    tornado.ioloop.IOLoop.current().start()
