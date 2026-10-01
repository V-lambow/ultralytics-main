import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import cv2
import tornado.ioloop
import tornado.web
from loguru import logger

from ultralytics import YOLO

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
    onnx_model_path = r"ckpt\yolo11n.pt"

    model = YOLO(onnx_model_path, task="detect")

    def initialize(self, template_dir, mask_dir, config_path):
        self.template_dir = template_dir
        self.mask_dir = mask_dir
        self.config_path = config_path
        self.detect_config = self.load_config()
        # 创建线程池执行器，用于异步处理可视化任务和模型推理
        self.executor = ThreadPoolExecutor(max_workers=4)

        # 设置Ultralytics的配置目录到当前项目目录，避免权限问题

        # 加载YOLO模型 - 这里使用Ultralytics的默认模型，实际部署时应替换为ONNX模型
        # 如果有ONNX模型文件，可以使用 YOLO("model.onnx") 加载
        try:
            # logger.info("Loading YOLO model...")
            # 尝试加载ONNX模型，如果没有则使用默认模型

            # logger.info(f"Loaded ONNX model: {onnx_model_path}")

            # 创建线程锁，确保推理过程线程安全
            self.model_lock = threading.Lock()
            # logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e!s}")
            raise RuntimeError(f"Model loading failed: {e!s}")

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

            results = await self.process_image(image_path, pose_id, sample_id)

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

    async def process_image(self, image_path, pose_id, sample_id):
        """基于Ultralytics YOLO的目标检测推理方法 保留原接口兼容性，返回格式与原方法一致.
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

            all_defects = []

            # 按配置文件信息region裁剪图片，支持多个A**区域配置时循环裁剪并推理
            for a_label, a_config in pose_config.items():
                region = a_config.get("region", [])
                if len(region) != 4:
                    logger.warning(f"Invalid region configuration for {a_label}: {region}")
                    continue

                x1, y1, x2, y2 = region

                # 裁剪区域
                roi = img[y1:y2, x1:x2]
                if roi.size == 0:
                    logger.warning(f"Empty ROI for {a_label} with region {region}")
                    continue

                # 使用线程池异步执行模型推理，确保线程安全
                results = await tornado.ioloop.IOLoop.current().run_in_executor(
                    self.executor, self.run_yolo_inference, roi
                )

                # 获取当前区域的阈值
                threshold = a_config.get("threshold", 0.25)

                # 检查是否有置信度大于threshold的项
                has_high_confidence = False
                max_confidence = 0.0
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        confidence = float(box.conf[0])
                        int(box.cls[0])
                        max_confidence = max(max_confidence, confidence)
                        if confidence > threshold:
                            has_high_confidence = True

                    if has_high_confidence:
                        break

                # 根据TODO要求：若有置信度大于threshold的项，返回空结果。若没有返回region
                if has_high_confidence:
                    logger.info(f"Found high confidence defect in {a_label}, returning empty result for this region")
                    # 返回空缺陷列表表示该区域有问题
                    continue
                else:
                    # 返回该区域的位置信息
                    defect = {
                        "code": "CuoLouZhuang",  # 将h_label改为code，值为CuoLouZhuang
                        "box": region,  # 返回整个区域
                        "area": (x2 - x1) * (y2 - y1),
                        "length": max(x2 - x1, y2 - y1),
                        "confidence": max_confidence,  # 没有检测到缺陷，置信度为0
                    }
                    all_defects.append(defect)

            logger.info(f"YOLO detection completed, found {len(all_defects)} regions without high confidence defects")
            return all_defects

        except Exception as e:
            logger.warning(f"Error in process_image: {e!s}")
            # 保留原接口的异常处理行为
            raise

    def run_yolo_inference(self, img):
        """执行YOLO模型推理，确保线程安全."""
        try:
            with self.model_lock:
                # 执行推理
                results = self.model(
                    img,
                    conf=0.25,  # 置信度阈值
                    iou=0.45,  # IOU阈值
                    device="cpu",  # 使用CPU，可根据需要改为'cuda'
                    verbose=False,
                )
                return results
        except Exception as e:
            logger.error(f"YOLO inference error: {e!s}")
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


def make_app(template_dir, mask_dir, config_path, port=30016):
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

    port = 8002
    app = make_app(template_dir, mask_dir, config_path, port)
    app.listen(port)
    logger.info(f"Server is running on port {port}")
    logger.info(f"Template directory: {os.path.abspath(template_dir)}")
    logger.info(f"Mask directory: {os.path.abspath(mask_dir)}")
    logger.info(f"Config file: {os.path.abspath(config_path)}")
    logger.info(f"Log directory: {os.path.abspath(log_dir)}")
    tornado.ioloop.IOLoop.current().start()
