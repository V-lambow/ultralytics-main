# detect_server_slidewind — 滑动窗口 YOLO 缺陷检测服务

基于 Tornado + Ultralytics YOLO 的图像缺陷检测 HTTP 服务，支持**滑动窗口推理**，适用于大图高精度缺陷检测场景。通过 Polygon Mask（inner/outer）精确控制检测区域。

---

## 目录

- [功能特性](#功能特性)
- [架构概览](#架构概览)
- [环境依赖](#环境依赖)
- [目录结构](#目录结构)
- [配置文件说明](#配置文件说明)
  - [配置结构](#配置结构)
  - [字段说明](#字段说明)
  - [Polygon Mask 说明](#polygon-mask-说明)
  - [完整配置示例](#完整配置示例)
- [启动服务](#启动服务)
- [PyInstaller 打包](#pyinstaller-打包)
- [API 接口文档](#api-接口文档)
  - [请求格式](#请求格式)
  - [响应格式](#响应格式)
  - [错误响应](#错误响应)
- [滑动窗口原理](#滑动窗口原理)
  - [参数说明](#参数说明)
  - [窗口尺寸计算公式](#窗口尺寸计算公式)
  - [边框外扩机制](#边框外扩机制)
  - [Polygon Mask 处理流程](#polygon-mask-处理流程)
- [NMS 合并机制](#nms-合并机制)
- [结果可视化输出](#结果可视化输出)
- [日志管理](#日志管理)
- [部署建议](#部署建议)

---

## 功能特性

| 特性 | 说明 |
|------|------|
| **滑动窗口推理** | 将大图分割为多个窗口并行推理，再合并结果，解决大图推理精度不足问题 |
| **Polygon Mask** | 支持 inner/outer 多边形区域精确控制检测范围，替代简单 bbox 裁剪 |
| **多区域配置** | 每个 `pose_id` 可配置多个检测区域（A1、A2 等），各自独立阈值和 mask |
| **NMS 合并** | 跨窗口的重叠检测结果自动进行 IoU 合并，避免重复检测 |
| **异步推理** | 基于 `ThreadPoolExecutor` 的异步推理，不阻塞 Tornado 事件循环 |
| **线程安全** | 推理过程加锁，确保多线程环境下模型调用安全 |
| **结构化日志** | 基于 Loguru，按日切割、保留7天、自动压缩 |
| **可视化输出** | 检测结果自动绘制并按 `res/YYYY/MM/DD/Sxxxxxxx/` 目录结构保存 |
| **PyInstaller 兼容** | 内置 monkey-patch 解决 PyInstaller 打包 torch 时的元数据读取问题 |

---

## 架构概览

```
客户端 POST 请求
       │
       ▼
┌─────────────────────────────────────┐
│  ImageDefectHandler (Tornado)       │
│                                     │
│  1. 解析 JSON 请求体                │
│  2. 加载 pose_id 对应配置           │
│  3. 对每个区域 (A1/A2/...):         │
│     ├─ 读取 inner/outer 多边形配置  │
│     ├─ 创建 Polygon Mask            │
│     │   ├─ inner: 保留区域，其余置0 │
│     │   └─ outer: 排除区域，保留其余│
│     └─ 对全图滑窗推理              │
│                                     │
│  4. NMS 合并重叠检测                │
│  5. 按位置排序                      │
│  6. 异步绘制可视化结果              │
│  7. 返回 JSON 响应                  │
└─────────────────────────────────────┘
```

---

## 环境依赖

```
Python >= 3.8
tornado >= 6.0
opencv-python >= 4.0
numpy >= 1.20
ultralytics >= 8.0
loguru >= 0.5
```

安装：

```bash
pip install tornado opencv-python numpy ultralytics loguru
```

---

## 目录结构

```
project/
├── detect_server_slidewind.py          # 主服务文件
├── config/
│   └── assemble_detect_item.json       # 检测配置文件（含端口、模型路径、区域配置）
├── models/
│   └── yolo26x.pt                      # YOLO 模型权重
├── log/                                # 日志目录（自动创建）
│   ├── 2026-08-28.log
│   └── ...
├── res/                                # 可视化结果（自动创建）
│   └── 2026/08/28/S0000001/
│       └── detected_image.jpg
├── templates/                          # 模板目录
├── masks/                              # 掩码目录
└── .ultralytics/                       # Ultralytics 配置缓存
    ├── settings.json
    └── cache/
```

---

## 配置文件说明

### 配置结构

配置文件为 JSON 格式，包含全局参数和按 **pose_id → 区域标签 → 区域参数** 三级结构组织的检测区域配置：

```json
{
  "port": 8000,
  "model_path": "models/yolo26x.pt",
  "use_slide_window": true,
  "slide_rows": 2,
  "slide_cols": 3,
  "overlap_pixels": 32,
  "border_expand": 0,
  "imgsz": 1088,
  "device": "0",

  "<pose_id>": {
    "<A_label>": {
      "shape_type": "polygon",
      "inner": {
        "points": [[x1, y1], [x2, y2], ...]
      },
      "outer": {
        "points": [[x1, y1], [x2, y2], ...]
      },
      "threshold": 0.25
    }
  }
}
```

### 字段说明

#### 全局参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `port` | int | `8000` | 服务监听端口 |
| `model_path` | string | `"models/yolo26x.pt"` | YOLO 模型权重路径（相对或绝对） |
| `use_slide_window` | bool | `true` | 是否启用滑窗模式 |
| `slide_rows` | int | `2` | 滑窗行数 |
| `slide_cols` | int | `3` | 滑窗列数 |
| `overlap_pixels` | int | `32` | 相邻窗口交叠像素数 |
| `border_expand` | int | `0` | 窗口边界外扩像素数（正数外扩，负数内陷） |
| `imgsz` | int | `640` | 模型推理图像尺寸 |
| `device` | string | `"cpu"` | 推理设备（`"cpu"` 或 `"0"` / `"cuda"`） |

#### 区域参数

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `pose_id` | string (顶层 key) | 是 | 姿态/工位 ID，请求中传入以匹配对应配置 |
| `A_label` | string (二级 key) | 是 | 区域标签，如 `"A1"`, `"B1"`，支持多区域 |
| `shape_type` | string | 否 | 多边形类型，固定为 `"polygon"` |
| `inner` | object | 仅 outer 不存在时 | 内多边形区域，该区域保留，其余置0 |
| `outer` | object | 仅 inner 不存在时 | 外多边形区域，该区域置0，其余保留 |
| `threshold` | float | 否 | 该区域的置信度阈值，默认 `0.25` |

### Polygon Mask 说明

每个区域（A1/B1/C1）**只会配置 inner 或 outer 其中之一**，不会同时存在：

| 配置类型 | mask 行为 | 典型用途 |
|----------|-----------|----------|
| `inner` | 多边形区域内保留原图，区域外全部 `× 0`（置黑） | 精确指定需要检测的目标区域 |
| `outer` | 多边形区域内 `× 0`（置黑），区域外保留原图 | 排除干扰区域，只检测其余部分 |
| 无 inner/outer | 不做 mask，对整张图推理 | 全图检测 |

**inner/outer 结构：**

```json
"inner": {
  "points": [[x1, y1], [x2, y2], [x3, y3], ...]
}
```

- `points`：多边形顶点坐标列表，格式为 `[x, y]`，坐标基于**原图像素坐标系**
- 坐标来源：通常由 LabelMe 等标注工具导出，或手动根据原图尺寸定义
- 支持任意凸/凹多边形

### 完整配置示例

```json
{
  "port": 8000,
  "model_path": "models/yolo26x.pt",
  "use_slide_window": true,
  "slide_rows": 2,
  "slide_cols": 3,
  "overlap_pixels": 32,
  "border_expand": 0,
  "imgsz": 1088,
  "device": "0",

  "pose_1": {
    "A1": {
      "shape_type": "polygon",
      "inner": {
        "points": [
          [413.72, 2662.14], [414.69, 2706.37], [424.30, 2745.15],
          [439.69, 2772.08], [481.99, 2807.01], [511.48, 2820.79],
          [551.22, 2828.17], [586.16, 2825.28], [627.19, 2809.26],
          [662.76, 2774.64], [682.96, 2745.79], [697.70, 2704.45],
          [700.90, 2666.63], [692.25, 2627.21], [667.57, 2590.67],
          [637.44, 2563.74], [614.69, 2551.56], [586.48, 2541.31],
          [549.62, 2539.71], [511.16, 2545.79], [466.93, 2567.91],
          [441.61, 2595.15], [424.94, 2621.12], [416.29, 2645.79]
        ]
      },
      "threshold": 0.3
    },
    "A2": {
      "shape_type": "polygon",
      "outer": {
        "points": [
          [2648.99, 2568.64], [2620.87, 2568.64], [2597.10, 2572.99],
          [2573.04, 2581.68], [2543.48, 2601.97], [2524.35, 2627.77],
          [2511.88, 2650.09], [2506.67, 2671.25], [2505.93, 2690.21],
          [2505.93, 2707.88], [2507.60, 2716.47], [2514.20, 2736.75],
          [2531.30, 2766.90], [2545.51, 2782.84], [2571.59, 2802.26],
          [2590.43, 2817.33], [2590.72, 2826.03], [2606.96, 2824.58],
          [2637.39, 2813.28], [2663.48, 2796.46], [2688.99, 2769.51],
          [2704.93, 2739.94], [2714.49, 2703.13], [2713.62, 2673.86],
          [2707.83, 2643.13], [2699.42, 2621.10], [2682.03, 2596.75],
          [2663.48, 2577.91]
        ]
      },
      "threshold": 0.25
    }
  },
  "pose_full_image": {
    "D1": {
      "threshold": 0.15
    }
  }
}
```

**说明：**

- `pose_1.A1` 配置了 `inner` 多边形，只检测该多边形内部区域
- `pose_1.A2` 配置了 `outer` 多边形，排除该区域后检测其余部分
- `pose_full_image.D1` 未配置 inner/outer，对整张图进行全图检测
- 端口和模型路径均在配置文件中指定

---

## 启动服务

### 默认启动

```bash
python detect_server_slidewind.py
```

默认参数（从 `config/assemble_detect_item.json` 读取）：
- 端口：`8000`
- 模型：`models/yolo26x.pt`
- 配置文件：`config/assemble_detect_item.json`

### 自定义启动

修改 `detect_server_slidewind.py` 底部的 `__main__` 部分：

```python
if __name__ == "__main__":
    template_dir = "templates"
    mask_dir = "masks"
    config_path = "config/assemble_detect_item.json"  # 修改为你的配置文件路径

    os.makedirs(template_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    # 端口从配置文件读取，未配置默认 8000
    with open(config_path, "r", encoding="utf-8") as f:
        server_config = json.load(f)
    port = int(server_config.get("port", 8000))

    app = make_app(template_dir, mask_dir, config_path)
    app.listen(port)
    logger.info(f"Server is running on port {port}")
    tornado.ioloop.IOLoop.current().start()
```

---

## PyInstaller 打包

### 打包命令

```bash
pyinstaller -F -n detect_server detect_server_slidewind.py
```

### PyInstaller 兼容性说明

脚本顶部已内置 `importlib.metadata` monkey-patch，解决 PyInstaller 打包 torch 时读取 `.dist-info` 元数据文件的 `UnicodeDecodeError` 问题：

```python
import importlib.metadata
_orig_read_text = importlib.metadata.PathDistribution.read_text
def _safe_read_text(self, name):
    try:
        return _orig_read_text(self, name)
    except UnicodeDecodeError:
        return None
importlib.metadata.PathDistribution.read_text = _safe_read_text
```

此 patch 对 `--onefile` 和 `--onedir` 模式均兼容，不影响正常运行。

### 打包注意事项

1. **模型文件**：需手动复制到打包输出目录，或在配置文件中使用绝对路径
2. **配置文件**：同上，确保运行时能正确找到 `config/assemble_detect_item.json`
3. **建议**：使用 `--onedir` 模式打包，便于管理模型和配置文件

---

## API 接口文档

### 端点

```
POST /industry/image_defect
Content-Type: application/json
```

### 请求格式

```json
{
  "job_id": "job_001",
  "sample_id": "S0000001",
  "pose_id": "pose_1",
  "file_names": ["image_001.jpg"],
  "relative_dir": "/data/images"
}
```

#### 请求字段

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `job_id` | string | 是 | 任务 ID |
| `sample_id` | string | 是 | 样本编号，用于可视化结果的目录命名 |
| `pose_id` | string | 是 | 姿态 ID，匹配配置文件中的对应配置 |
| `file_names` | list[string] | 是 | 待检测图片文件名列表（取第一个） |
| `relative_dir` | string | 是 | 图片文件的相对路径目录 |

**注意**：滑窗参数（`slide_rows`、`slide_cols`、`overlap_pixels`、`border_expand`）现在从配置文件读取，不再通过请求传入。

### 响应格式

#### 成功响应

```json
{
  "error_code": 0,
  "error_msg": "OK",
  "data": {
    "product_type": "",
    "job_id": "job_001",
    "pose_id": "pose_1",
    "results": [
      {
        "code": "scratch",
        "box": [100, 50, 800, 600],
        "area": 420000,
        "length": 700,
        "confidence": 0.85
      }
    ],
    "file_names": ["image_001.jpg"]
  }
}
```

#### results 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `code` | string | 模型检测到的缺陷类别名 |
| `box` | `[x1, y1, x2, y2]` | 检测框坐标（原图像素坐标） |
| `area` | int | 检测框面积（像素²） |
| `length` | int | 检测框最长边长度（像素） |
| `confidence` | float | 检测置信度（0.0 ~ 1.0） |

### 错误响应

```json
{
  "error_code": 999,
  "error_msg": "False"
}
```

常见错误原因：
- `file_names` 为空
- `pose_id` 在配置文件中不存在
- 图片路径无效
- 模型加载失败

---

## 滑动窗口原理

### 参数说明

滑动窗口参数在配置文件中设置：

```
slide_rows  × slide_cols = 窗口总数
overlap_pixels           = 相邻窗口重叠像素
border_expand            = 窗口边界外扩像素
```

### 窗口尺寸计算公式

```
窗口高度 = (图片高度 + (slide_rows - 1) × overlap_pixels) ÷ slide_rows
窗口宽度 = (图片宽度 + (slide_cols - 1) × overlap_pixels) ÷ slide_cols
```

**示例：** 图片 1920×1080，slide_rows=3, slide_cols=4, overlap=50

```
窗口高度 = (1080 + 2×50) ÷ 3 = 393 px
窗口宽度 = (1920 + 3×50) ÷ 4 = 517 px
```

### 边框外扩机制

`border_expand` 控制窗口边界调整：

| 值 | 行为 | 用途 |
|----|------|------|
| 正数 | 向外扩展，扩展区域填充黑色 | 增加上下文信息，减少边缘误检 |
| 负数 | 向内收缩，只取窗口内部区域 | 去除窗口边缘区域 |
| 0 | 不做调整 | 默认行为 |

**坐标映射流程：**
1. 提取窗口图像（含外扩部分）
2. 在扩展后的窗口上执行推理
3. 将检测框坐标映射回原图坐标
4. 裁剪掉外扩部分，只保留原始窗口范围内的检测结果

### Polygon Mask 处理流程

```
原始图像 (img)
    │
    ▼
apply_region_mask(img, a_config)
    │
    ├─ inner 配置 → cv2.fillPoly 创建多边形 mask → bitwise_and 保留区域
    │
    ├─ outer 配置 → cv2.fillPoly 创建多边形 mask → bitwise_not 取反 → bitwise_and 排除区域
    │
    └─ 无配置 → 返回原图
    │
    ▼
masked_img（已 mask 的全图）
    │
    ▼
calculate_slide_positions() → 滑窗位置（全图坐标系）
    │
    ▼
extract_window(masked_img, roi, border_expand) → 提取窗口
    │
    ▼
YOLO 推理 → 坐标映射回原图 → NMS 合并
```

**关键函数：**
- `create_polygon_mask(img_shape, points)`：根据多边形顶点创建 mask（`detect_server_slidewind.py:15-24`）
- `apply_region_mask(img, a_config)`：根据 inner/outer 配置处理图像（`detect_server_slidewind.py:27-50`）

---

## NMS 合并机制

滑动窗口模式下，相邻窗口可能检测到同一目标，需要合并重复结果。

**合并策略：**
1. 按置信度降序排列所有检测结果
2. 逐一取出最高置信度的结果
3. 与剩余结果计算 IoU（仅同 code 的结果比较）
4. IoU ≥ 0.5 的结果：**取并集合并框**，保留最高置信度
5. IoU < 0.5 的结果：保留为独立检测

**关键函数：** `compute_iou()` 和 `nms_merge()`（`detect_server_slidewind.py:53-119`）

---

## 结果可视化输出

检测完成后，可视化结果异步保存至：

```
res/{YYYY}/{MM}/{DD}/{sample_id}/detected_{filename}
```

**示例：**
```
res/2026/08/28/S0000001/detected_image_001.jpg
```

**绘制内容：**
- 红色矩形检测框
- 类别标签 + 置信度文本

---

## 日志管理

使用 Loguru 管理日志，配置位于 `detect_server_slidewind.py:125-136`：

| 配置 | 值 | 说明 |
|------|------|------|
| 日志目录 | `log/` | 自动创建 |
| 文件命名 | `{YYYY-MM-DD}.log` | 按日切割 |
| 切割时间 | 每天 00:00 | 创建新日志文件 |
| 保留天数 | 7 天 | 超期自动删除 |
| 压缩方式 | zip | 旧日志自动压缩 |
| 日志级别 | INFO | 记录 INFO 及以上 |

---

## 部署建议

1. **模型选择**：在配置文件 `model_path` 中指定训练好的模型权重，支持 `.pt`、`.onnx` 等格式
2. **GPU 加速**：配置文件中 `device` 设为 `"0"` 或 `"cuda"`（需安装 CUDA 版 PyTorch）
3. **并发调优**：`ThreadPoolExecutor(max_workers=4)` 可根据 CPU 核数和 GPU 能力调整
4. **配置文件热加载**：如需运行时更新配置，可在 `initialize()` 中每次请求重新 `load_config()`
5. **进程管理**：生产环境建议使用 Supervisor 或 systemd 管理进程，配合 Nginx 反向代理
6. **PyInstaller 打包**：使用 `--onedir` 模式，模型和配置文件放在输出目录下
