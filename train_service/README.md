# train_service — YOLO 训练控制服务

基于 **ZMQ + Ultralytics YOLO** 的常驻训练服务：后台进程接收控制指令、在独立线程中执行训练，并通过 PUB/SUB 实时广播训练指标。适用于远程触发训练、无终端环境下的训练监控、多客户端同时观测训练曲线等场景。

服务可选开启 **ROI 裁剪训练**：只在原图的指定矩形区域内训练与验证，支持检测、分割、姿态、OBB 四种任务，数据集文件不做任何修改。

---

## 目录

- [功能特性](#功能特性)
- [环境与安装](#环境与安装)
- [快速开始](#快速开始)
- [命令行用法](#命令行用法)
- [配置参考](#配置参考)
- [命令协议](#命令协议)
- [推送消息格式](#推送消息格式)
- [ROI 裁剪说明](#roi-裁剪说明)
- [暂停、续训与停止](#暂停续训与停止)
- [输出产物](#输出产物)
- [作为库使用](#作为库使用)
- [PyInstaller 打包](#pyinstaller-打包)
- [已知问题](#已知问题)
- [排障](#排障)

---

## 功能特性

| 特性 | 说明 |
|------|------|
| 远程训练控制 | `train` / `pause` / `holdon` / `stop` / `status`，通过 ZMQ REQ-REP 同步下发 |
| 实时指标广播 | 每个 epoch 结束通过 PUB 推送 loss、P、R、mAP50、mAP50-95、分项 loss |
| 多客户端订阅 | SUB 可同时开多个进程观测同一训练，互不影响 |
| 断点续训 | `pause` 保存 `last.pt`，`holdon` 以 `resume` 恢复训练 |
| ROI 裁剪 | `useRoi` 开启后，训练集与验证集都只保留 ROI 区域；支持 `detect` / `segment` / `pose` / `obb` |
| 验证回映射 | 验证时的指标、可视化、`save_txt` / `save_json` 坐标都可回到原图坐标系 |
| 信号退出 | `SIGINT` / `SIGTERM` 优雅关闭，socket `LINGER=0` 避免挂起 |
| 输出隔离 | 结果固定写入配置文件同级的 `output/<alias>/` |
| 可打包 | 附带 `train_service.spec`，PyInstaller onedir 模式产出独立目录 |

---

## 环境与安装

```bash
pip install pyzmq ultralytics
```

- Python ≥ 3.8，`torch` / `torchvision` 按本机 CUDA 版本安装
- `train_service.py` 直接 import 了 ultralytics 的内部模块与函数（`YOLODataset`、`DetectionTrainer` / `SegmentationTrainer` / `PoseTrainer` / `OBBTrainer`、`ultralytics.utils.patches.imread` 等），**升级 ultralytics 前必须回归验证**，这些接口不是稳定 API
- 代码里直接使用 `cv2`（ROI 多边形裁剪、mask 仿射变换、验证画布重建），由 ultralytics 的依赖带入，不要单独卸载 `opencv-python`

---

## 快速开始

### 1. 写配置文件

新建 `config.json`（放在哪都行，输出目录跟着它走）：

```json
{
  "train": {
    "port": 9257,
    "modelPath": "yolo26n.pt",
    "dataYaml": "E:/datasets/curb/data.yaml",
    "epochs": 100,
    "batch": 16,
    "imgsz": 640,
    "device": "auto",
    "task": "detect",
    "alias": "curb_v1",
    "useRoi": false
  }
}
```

### 2. 启动服务

```bash
python main.py --config config.json
```

启动日志：

```
[INFO] main: PUB bound to tcp://*:9257
[INFO] main: REP bound to tcp://*:9258
[INFO] train_service: Train service started. Waiting for commands...
```

服务阻塞运行，`Ctrl+C` 退出。**同一份配置只会被读取一次**，改参数需重启服务。

### 3. 另开终端：先订阅指标

```bash
python main.py --config config.json --subscribe
```

### 4. 再下发训练命令

```bash
python main.py --config config.json --cmd train
# {"status": "ok", "message": "Training started"}
```

订阅端开始打印：

```
Epoch    1/100 | loss: 3.1024 (box: 2.0113 cls: 0.8809 dfl: 0.2102) | P: 0.6210  R: 0.5531  mAP50: 0.6012  mAP50-95: 0.3121
...
[DONE] Results saved to: output/curb_v1
```

### 5. 查看状态 / 暂停 / 继续 / 停止

```bash
python main.py --config config.json --cmd status
python main.py --config config.json --cmd pause
python main.py --config config.json --cmd holdon
python main.py --config config.json --cmd stop
```

> 顺序很重要：PUB/SUB 没有回放，**先启动订阅端再开训练**，否则会漏掉开头几个 epoch。

---

## 命令行用法

`main.py` 有三种模式，由参数组合决定：

| 参数组合 | 模式 | 说明 |
|----------|------|------|
| `--config <json>` | 服务端 | 启动服务并阻塞，端口取自 `train.port` |
| `--config <json> --cmd <cmd>` | 客户端 | 向运行中的服务发送一条命令，打印 JSON 响应后退出 |
| `--config <json> --subscribe` | 订阅端 | 持续接收推送，`Ctrl+C` 退出（不影响服务端） |
| `--subscribe` | 订阅端 | 不带 `--config` 时用默认端口 9257 |
| 无参数 / 缺 `--config` | — | `argparse` 直接报错退出 |

`--cmd` 的可用取值：`train`、`pause`、`holdon`、`stop`、`status`。

**客户端模式只用到 `train.port`**，其余字段不影响行为；服务端也只用到 `train` 节点。

退出码与输出：命令响应以 JSON 打印到 stdout；服务不在线时 ZMQ 会阻塞等待（没有超时），需要自己 `Ctrl+C`。

---

## 配置参考

### 完整示例（含 ROI）

```json
{
  "train": {
    "port": 9257,
    "modelPath": "yolo26n.pt",
    "dataYaml": "E:/datasets/curb/data.yaml",
    "epochs": 200,
    "batch": 16,
    "imgsz": 640,
    "device": "auto",
    "amp": true,
    "cos_lr": true,
    "optimizer": "auto",
    "task": "detect",
    "alias": "curb_roi_v1",
    "useRoi": true,
    "roi": { "x": 640, "y": 320, "width": 1280, "height": 720 }
  }
}
```

### train 字段

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `port` | int | `9257` | PUB 端口；REP 端口固定为 `port + 1` |
| `modelPath` | string | `"yolo26n.pt"` | 权重或模型 YAML，**建议绝对路径** |
| `dataYaml` | string | `"coco8.yaml"` | 数据集配置，**建议绝对路径** |
| `epochs` | int | `100` | 训练轮数 |
| `batch` | int | `16` | 批大小，`-1` 触发 AutoBatch |
| `imgsz` | int | `640` | 输入尺寸（ROI 模式下作用于裁剪后的图） |
| `device` | string | `"auto"` | `auto` → 有 CUDA 用 `"0"`，否则 `"cpu"`；也可写 `"0"`、`"0,1"`、`"cpu"` |
| `amp` | bool | `true` | 混合精度 |
| `cos_lr` | bool | `false` | 余弦学习率调度 |
| `optimizer` | string | `"auto"` | `auto` / `SGD` / `AdamW` / `Adam` |
| `task` | string | `"detect"` | `detect` / `segment` / `pose` / `obb`，**必须与 `modelPath` 的任务一致** |
| `alias` | string | `""` | 输出子目录名（映射到 `name`）；为空时由 ultralytics 生成 `train`、`train2`… |
| `useRoi` | bool | `false` | 是否启用 ROI 裁剪 |
| `roi` | object | — | ROI 矩形，见下节 |

`plots=True`、`exist_ok=True`、`project=<config 同级>/output` 在代码里写死，不可配置。

未暴露、只能改代码的常用超参：`workers`、`patience`、`lr0`、`lrf`、`weight_decay`、`warmup_epochs`、`seed`、`cache`、`save_period`、`close_mosaic` 等。参数含义见仓库根目录 `train_param.md`。调整方式：修改 `TrainService._train_loop()` 里构造的 `kw` 字典。

> `workers` 未暴露的影响：ultralytics 默认 `workers=8`，Windows 下每个 worker 都是独立进程并重新 import torch，内存占用明显。长时间常驻的服务建议直接用代码把 `workers` 降到 2~4 后打包。

### roi 字段

| 字段 | 说明 |
|------|------|
| `x`, `y` | ROI 左上角像素坐标，**原图坐标系**（不是裁剪图或缩放图） |
| `width`, `height` | ROI 宽高，必须 `> 0` |

启用条件（同时满足）：`useRoi == true`、`roi` 里同时存在 `x` / `y` / `width` / `height`、`width > 0 且 height > 0`。

任一条件不满足时**静默回退为全图训练**（不打印警告），排查时先看启动日志里有没有 `ROI enabled: (...)`。

ROI 超出图像边界时自动收窄到图像内（`x/y/width/height` 不需要保证在图像内），但裁剪后必须至少剩 1 像素，否则抛错。

---

## 命令协议

REQ/REP，地址 `tcp://localhost:<port + 1>`，JSON 报文。

请求：

```json
{ "cmd": "train" }
```

响应：

```json
{ "status": "ok", "message": "Training started" }
```

| 命令 | 作用 | 前置条件 | 返回 |
|------|------|----------|------|
| `train` | 新起一次训练（`resume=False`） | `training == false` | `{"status": "ok", "message": "Training started"}` |
| `pause` | 置 `trainer.stop`，当前 epoch 结束后保存 `last.pt` 并退出 | 正在训练 | `{"status": "ok", "message": "Training paused"}` |
| `holdon` | 用 `resume` 重启训练 | `paused == true` | `{"status": "ok", "message": "Training resumed"}` |
| `stop` | 同 `pause`，但不置 `paused`，之后不可续训 | 正在训练 | `{"status": "ok", "message": "Training stopped"}` |
| `status` | 查询状态 | 无 | 见下 |

`status` 响应字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | `"ok"` |
| `training` | bool | 训练线程是否在运行 |
| `paused` | bool | 是否处于暂停态（可 `holdon`） |
| `save_dir` | string \| null | 当前/上次输出目录 |
| `last_ckpt` | string \| null | `output/<alias>/weights/last.pt` |

失败时统一返回：

```json
{ "status": "error", "message": "Already training" }
```

所有 handler 都包了 `try/except`，未知命令同样返回 error，服务不会因此退出；但训练线程内的异常只会通过 PUB 推送 `error` 主题，`status` 仍会回到 `training: false`。

---

## 推送消息格式

PUB socket 使用两帧 multipart：`[topic][json]`。

### `metrics`（每个 epoch 结束）

| 字段 | 类型 | 说明 |
|------|------|------|
| `epoch` | int | 当前轮次，从 1 开始 |
| `total_epochs` | int | 总轮数 |
| `loss` | float | `trainer.loss` |
| `box_loss` / `cls_loss` / `dfl_loss` | float | `trainer.tloss[0/1/2]` |
| `precision` / `recall` | float | `metrics/precision(B)`、`metrics/recall(B)` |
| `mAP50` / `mAP50_95` | float | `metrics/mAP50(B)`、`metrics/mAP50-95(B)` |

> 三个分项 loss 直接按下标取 `trainer.tloss`，**只对 `detect` 准确**。`segment` 的 `tloss` 是 `(box, seg, cls, dfl)`，`pose` 是 `(box, pose, kobj)`，字段名会对不上。指标本身（P/R/mAP）取的是 box 指标，四种任务都有。

### `error`

| 字段 | 类型 | 说明 |
|------|------|------|
| `message` | string | 训练线程捕获的异常文本 |

### `train_end`

| 字段 | 类型 | 说明 |
|------|------|------|
| `save_dir` | string | 输出目录 |

`pause` 也会触发 `train_end`（训练进程确实退出了），不代表训练真正完成。

---

## ROI 裁剪说明

### 设计

ROI 在**数据读取层**生效，而不是在 transform 链里：

```
YOLODataset.get_labels()      →  _crop_label()   把缓存的标注裁到 ROI 内
YOLODataset.load_image()      →  先裁图，再按 imgsz 缩放
```

- 标注文件（`labels/*.txt`）和数据集 YAML **完全不动**，ROI 只在内存里生效
- `load_image` 被重写，因此 `Mosaic` 取另外 3~8 张图时拿到的也是裁剪后的图，**拼图不会出现"一半裁了一半没裁"**
- `set_rectangle` 用的是裁剪后的 `shape`，rect 分批按 ROI 的宽高比排序
- 训练集与验证集**都**做 ROI 裁剪（`build_dataset` 对 `train` / `val` 传同一个 ROI）

### 裁剪内容

| 类型 | 处理 |
|------|------|
| bbox | 裁到 ROI 内，重算中心与宽高，归一化到 ROI 尺寸；完全在 ROI 外的目标丢弃 |
| 多边形 | Sutherland–Hodgman 逐边裁剪；裁完少于 3 个点的目标丢弃 |
| 关键点 | 坐标裁到 ROI 内并重新归一化；落在 ROI 外的关键点可见性置 0，目标本身保留 |
| OBB | 多边形裁剪后由 `xyxyxyxy2xywhr` 重新生成旋转框 |

### 验证与可视化

验证时模型看到的是 ROI 视图，因此：

- **指标（P/R/mAP、mask IoU）在 ROI 视图内计算**。ROI 边界上的目标按裁剪后的框算 IoU，与整图标注不是严格等价
- **可视化与 `save_txt` / `save_json` 会映射回原图坐标**：验证器通过 mixin 接管 `_prepare_batch` / `scale_preds` / `plot_predictions`，用 `crop_shape`、`full_shape`、`roi_origin` 和 letterbox 参数构造仿射关系
- 绘制验证图时按原图重建画布（`_full_canvas`），ROI 区域在画布中按原图比例摆放，框/关键点/mask 叠在对应位置上

### 支持的任务

`build_roi_trainer(task, roi)` 支持 `detect` / `segment` / `pose` / `obb`，其它任务直接抛 `ValueError`（会被转成 PUB 的 `error` 消息）。生成的是动态子类：

```
ROITrainerMixin + DetectionTrainer / SegmentationTrainer / PoseTrainer / OBBTrainer
```

`get_validator()` 返回的验证器实例会就地换成 `ROIValidatorMixin + 原验证器类`，这样不用为四个任务各写一个验证器。

---

## 暂停、续训与停止

ultralytics 没有原生"暂停"，这里用 **`trainer.stop = True` 中断 + `resume` 重启** 模拟。

```
train  ──►  _train_loop()  ──►  YOLO.train(resume=False)
                                     │
pause  ──►  trainer.stop = True ─────┤  当前 epoch 结束后 break
                                     │  final_eval() → 保存 last.pt
                                     │  on_train_end → 广播 train_end
                                     ▼
                              training=False, paused=True
                                     │
holdon ──►  join(old thread, 30s) ───┤  置 resume → 新建训练线程
                                     ▼
                              YOLO.train(resume=True)
```

| 命令 | 实际行为 | 可否续训 |
|------|----------|----------|
| `pause` | 终止当前训练，保存 `last.pt`，广播 `train_end` | 可以，用 `holdon` |
| `holdon` | 从 checkpoint 恢复，`project` / `name` 沿用 checkpoint 里记录的值 | — |
| `stop` | 同样终止训练，但不置 `paused` | 不可以（`holdon` 会返回 `Not paused`） |

`stop` 之后再 `train` 是**从头训练**，并且因为 `exist_ok=True` 会写回同一个输出目录，可能覆盖上一次结果。想保留现场用 `pause`。

---

## 输出产物

输出根目录 = **配置文件所在目录** `/output`（`config_path` 为空时是相对路径 `output/`）。

```
<config 所在目录>/output/<alias>/
├── weights/
│   ├── best.pt
│   └── last.pt          # pause / holdon 的续训依据
├── results.csv
├── args.yaml
├── results.png
├── confusion_matrix.png
├── PR_curve.png
├── F1_curve.png
├── train_batch*.jpg
└── val_batch*_labels.jpg / val_batch*_pred.jpg
```

启用 ROI 时，`val_batch*_labels.jpg` / `val_batch*_pred.jpg` 画的是**原图**，ROI 外区域保留但通常没有标注和预测。

---

## 作为库使用

### 直接内嵌服务（不走命令行）

```python
import json
from train_service import TrainService

with open("config.json", encoding="utf-8") as f:
    cfg = json.load(f)

service = TrainService(cfg, "config.json")
service.start_server()          # 阻塞
```

### 从自己的程序下发命令

```python
import json
import zmq

ctx = zmq.Context()
req = ctx.socket(zmq.REQ)
req.setsockopt(zmq.LINGER, 0)
req.connect("tcp://localhost:9258")      # port + 1

req.send_json({"cmd": "status"})
print(req.recv_json())

req.close()
ctx.term()
```

### 订阅指标

```python
import json
import zmq

ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.setsockopt(zmq.LINGER, 0)
sub.connect("tcp://localhost:9257")      # port
sub.setsockopt(zmq.SUBSCRIBE, b"")       # 订阅全部 topic

while True:
    topic, payload = sub.recv_multipart()
    print(topic.decode(), json.loads(payload))
```

---

## PyInstaller 打包

```bash
cd train_service
pyinstaller train_service.spec
```

| 配置 | 值 | 含义 |
|------|-----|------|
| 入口 | `main.py` | `train_service.py` 由 import 自动收入 |
| `console` | `True` | 保留控制台，服务端要看日志 |
| `exclude_binaries` + `COLLECT` | `True` | onedir 模式，输出 `dist/train_service/` |
| `upx` | `True` | 有 UPX 时压缩，没装自动跳过 |
| `datas` / `binaries` | 空 | 不额外打包数据文件 |

打包后注意：

1. **权重、数据集 YAML、字体都不会打进去**。配置里一律用绝对路径，或把文件放到 exe 旁边
2. **绘图字体**：matplotlib 风格校验会找 `Arial.ttf`，ultralytics 的查找顺序是 `~/.config/Ultralytics/` → 系统字体 → 联网下载到 `~/.config/Ultralytics/`。离线机器上第一次画图会失败，请提前把 `Arial.ttf` 放进 `%USERPROFILE%\.config\Ultralytics\`
   仓库里的 `ultralytics/Arial.ttf` **不会被自动使用**（不在查找路径内），需要显式拷到上面那个目录或打进 `datas`
3. **ultralytics 的 YAML 模型配置**在运行时读取，需保证可访问
4. **Ultralytics 配置目录**：首次运行会创建 `~/.config/Ultralytics/`
5. **PyInstaller + torch**：遇到 `.dist-info` 的 `UnicodeDecodeError`，参考 `README_detect_server_slidewind.md` 里的 `importlib.metadata` monkey-patch

---

## 已知问题

按影响从大到小：

1. **ROI 验证的坐标回映射有偏差** — `_sample_geometry()` 里 `gain` 的算法基于旧版 ultralytics 的 `ratio_pad` 语义（`((letterbox_ratio, letterbox_ratio), (pad_x, pad_y))` + pre-letterbox 的 `resized_shape`）；当前仓库的 `BaseDataset` 把 `ratio_pad` 存成 `(h_ratio, w_ratio)`、`LetterBox` 又把 `resized_shape` 覆盖成 letterbox 画布尺寸，两者嵌套后 `ratio_pad[si][0][0]` 拿到的是"裁剪图 → load_image 缩放"的比例（约为 `imgsz / 长边`）。于是当原图明显大于 `imgsz` 时（例如 1920×1080）`gain` 会偏小一个数量级，**验证图上的预测框/mask、以及 `save_txt` / `save_json` 的坐标会偏大并错位**；指标本身在 ROI 视图内计算，不受影响。修正方式：用 `ratio_pad[si][0] * crop_shape` 反推 letterbox 输入尺寸，再乘 letterbox 比例得到画布增益，同时把 letterbox 的 `left/top` 从 `norm_affine` 里扣掉
2. **`holdon` 续训取错 checkpoint** — 代码只传 `resume=True`，ultralytics 的 `check_resume()` 会去 **当前工作目录** 下 `get_latest_run()` 找 `last.pt`，而不是本次训练的 `output/<alias>/weights/last.pt`。多份 run 并存或 CWD 不同时会续错，甚至报 `Resume checkpoint not found`。`self.last_ckpt` 已经记在 `TrainService` 上，直接传 `resume=self.last_ckpt` 即可
3. **`holdon` 可能起第二个训练线程** — `pause` 后 `join(timeout=30)`，若当前 epoch 超过 30 秒，join 超时后仍会新建线程，两个训练同时抢 GPU。join 之后应检查 `is_alive()` 并直接返回错误
4. **`pause` 与 `on_train_start` 的竞态** — 训练线程刚启动、`self.trainer` 还没赋值时下发 `pause`，`trainer.stop` 设不上，但 `paused` 已经被置 True。随后 `holdon` 会认为可以续训
5. **`cache="disk"` 对 ROI 失效** — `ROIDataset.load_image()` 重写了基类实现，丢掉了基类里读取 `*.npy` 的分支，磁盘缓存完全不走，大图数据集会明显变慢（`cache="ram"` 正常）
6. **灰度图验证绘图会崩** — `_full_canvas()` 无条件做 `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`，灰度（`channels: 1`）数据集会抛异常；`imread()` 也没有沿用 `self.cv2_flag`
7. **`_full_canvas()` 每个验证 batch 重新读原图** — 前 3 个 batch × 每 batch 全部样本都要全分辨率解码一次，大图数据集验证会明显变慢；`cache="ram"` 时也照样落盘读
8. **`_map_gt` 的 mask 分支靠形状猜语义** — 用 `masks.shape[0] == len(batch_idx) and masks.max() <= 1` 判断是"每实例一张"还是"每图一张索引图"。`overlap_mask=True` 且每个 batch 恰好每图 1 个目标时会被误判成按实例处理，验证图上的 GT mask 会串图（只影响画图）
9. **ROI 不出警告** — `useRoi=true` 但 `roi` 字段不全或宽高 ≤ 0 时静默回退全图训练，只靠 `ROI enabled` 日志判断
10. **DDP 不支持 ROI** — `build_roi_trainer()` 生成的是 `type()` 动态类，没有注册成模块属性，spawn 方式的多进程（Windows DDP）无法 pickle；ROI 目前只能单卡单进程
11. **异常信息不回传状态** — 训练线程里的异常只走 PUB，`status` 只能看到 `training=false`，不知道失败原因；订阅端没连上就彻底静默
12. **`_publish()` 没有保护** — 服务关闭时 socket 可能已被 `_cleanup()` 关闭，此时训练线程仍在推送会抛 `ZMQError`，异常会冒泡进 ultralytics 的 callback 链
13. **无鉴权、无超时** — 绑定 `tcp://*`，同网段任何机器都能下发 `train`；客户端 REQ 没有超时，服务不在线时会一直等
14. **状态不持久化** — 服务重启后 `training` / `paused` / `save_dir` 归零，只能从 `output/` 目录推断之前的进度
15. **配置不可热更新** — 启动读一次 JSON，改端口必须重启；改训练超参同样要重启（`train_cfg` 是启动时解析的 dict）
16. **`multi_modal` 数据集不支持** — `ROITrainerMixin.build_dataset()` 直接构造 `ROIDataset`（基类 `YOLODataset`），绕过了 `build_yolo_dataset()` 里对 `YOLOMultiModalDataset` / 语义分割数据集的判断，YOLO-World / YOLOE 这类需要文本特征的数据集无法开 ROI
17. **依赖 ultralytics 内部 API** — `YOLODataset`、四个 Trainer、`unwrap_model`、`imread`、`colorstr` 均为半私有接口，升级时可能失效

---

## 排障

| 现象 | 排查方向 |
|------|----------|
| `--cmd` 一直卡住 | 服务没起、端口不一致（`port` vs `port + 1`）、防火墙拦截 |
| 收不到指标 | 订阅端晚于训练启动（PUB 不回放）；`--config` 里的 `port` 与服务端不一致 |
| 训练没开始就 `training: false` | 起订阅端看 `error` 主题；常见是 `dataYaml` / `modelPath` 路径错、`task` 与权重不匹配 |
| 输出目录被覆盖 | `exist_ok=True` + 相同 `alias`；换 `alias` 或先清理旧目录 |
| ROI 没生效 | 启动日志里有没有 `ROI enabled: (x, y, w, h)`；`useRoi` 拼写；`roi` 四个字段是否齐全且宽高 > 0 |
| 验证图框位置不对 | 见[已知问题](#已知问题)第 1 条，当前实现的坐标回映射在缩图场景有偏差 |
| `holdon` 报 `Resume checkpoint not found` | 见[已知问题](#已知问题)第 2 条，续训找的是 CWD 下的 `last.pt` |
| 内存暴涨 | `workers` 默认 8（Windows 每 worker 重新 import torch）；改代码降 `workers` 后重启 |

---

## 文件结构

```
train_service/
├── main.py                  # 入口：argparse + 服务端 / 客户端 / 订阅端
├── train_service.py         # TrainService + ROI 数据集/训练器/验证器
├── train_service.spec       # PyInstaller 描述文件
├── build/ dist/             # 打包中间产物与输出
├── output/                  # 训练输出（运行时创建，位于配置文件同级）
└── README.md
```
