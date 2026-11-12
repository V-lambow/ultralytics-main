# YOLO26 + SAHI 切片推理 / 切片训练脚本

## 概述

SAHI（Slicing Aided Hyper Inference）把大图切成带重叠的小块分别推理，再合并结果，用于提升**小目标**和**超大分辨率图片**的检出率。

本目录提供两个脚本：

| 文件 | 功能 |
|---|---|
| `run_yolo26_predict_sahi.py` | 切片推理：单图 / 文件夹 / 视频 / 网络流 / 摄像头 |
| `run_yolo26_train_sahi.py` | 切片训练：YOLO 数据集 → 切片 → 转 YOLO 格式 → 训练 |

> SAHI 本身是纯推理库，没有训练 API。训练脚本的做法是先用 SAHI 把数据集切片，再用 ultralytics 在切片上训练（小目标数据集的标准做法）。

## 环境要求

- 已安装 `sahi`（验证版本 0.12.8）与 `ultralytics`（验证版本 8.4.102）
- 本仓库环境：`D:\.conda\envs\ultraylytics\python.exe`
- 新环境安装：`pip install sahi`

## 切片推理

核心调用：

```python
AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=..., device=..., confidence_threshold=..., image_size=...)
get_sliced_prediction(image, model, slice_height=..., slice_width=..., overlap_*_ratio=..., postprocess_type=..., ...)
```

流程：

1. 整图按 `slice_height x slice_width` 切片，重叠比例 `overlap_*_ratio`
2. 每块推理，得到块内检测框
3. 默认同时做一次整图推理（`perform_standard_pred=True`），弥补切片对大目标的削弱；可用 `--no-standard-pred` 关闭
4. 按 `postprocess_type` 合并去重（默认 `GREEDYNMM`，匹配度量 `IOS`）

输出：

```
runs/detect/predict_sahi/        # --project/--name 可改
├── bus.png                      # 标注图（export_visuals）
├── bus.txt                      # --save-txt 时生成
└── video.mp4                    # 视频输入时输出单个标注视频
```

- 图片：`export_visuals` 保存标注图
- 视频：逐帧切片推理后用 `cv2.VideoWriter` 合成**单个 mp4**（参考示例是逐帧存 PNG，长视频会产出大量文件）
- `--save-txt` 格式：`cls conf x1 y1 x2 y2`（绝对像素坐标，视频额外带帧号前缀）

## 切片训练

流程：

1. `check_det_dataset` 解析 `--data` 指定的 YOLO 数据集配置（沿用 ultralytics 的路径解析与自动下载）
2. 内置 yolo→coco 转换（类别 id 保持 0 基，与 SAHI 导出逻辑一致）
3. `slice_coco` 切片（`min_area_ratio` 过滤切片边缘的残缺目标）
4. `Coco.export_as_yolo` 导出 YOLO 格式（图片与 txt 并排，ultralytics 可直接读取）
5. 生成 `data.yml`，调用 `YOLO.train()` 训练

输出结构：

```
runs/sahi_data/                  # --output 可改
├── data.yml                     # 切片数据集配置（train/val/nc/names）
├── train_source.json            # 原始训练集转出的 coco 标注
├── val_source.json
├── train/                       # 切片图 + 切片 coco json
│   ├── train_coco.json
│   └── *.png
├── val/
└── yolo/
    ├── train/                   # 切片图片 + 同名 txt 标签
    └── val/
```

重复训练时用 `--skip-slice` 复用已切好的数据，不必重切。

## 限制与注意事项

- **仅支持检测（bbox）标签**：标签必须是 5 列 `cls cx cy w h`；遇到分割/obb 标签会直接报错，不会静默跳过
- **Windows 默认复制图片**：`os.symlink` 需要管理员或开发者模式，默认改为复制；确认有权限可用 `--symlink` 节省磁盘
- **输出目录非空会报错**：脚本不删除任何文件，需换 `--output` 或加 `--skip-slice`
- **`--data` 必须包含 `val`**：ultralytics 的 `check_det_dataset` 强制要求 train/val 两个键
- **`--conf` 双重生效**：同时作为模型推理阈值和切片合并后的过滤阈值

详细命令与参数说明见 [run_sahi_usage.md](run_sahi_usage.md)。

## 参考

- 官方示例：`examples/YOLOv8-SAHI-Inference-Video/yolov8_sahi.py`
- SAHI 文档：https://github.com/obss/sahi
