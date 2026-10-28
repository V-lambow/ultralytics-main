# train_service — YOLO 训练控制服务

基于 **ZMQ + Ultralytics YOLO** 的常驻训练服务：后台进程接收控制指令、在独立线程中执行训练，并通过发布订阅（PUB/SUB）实时广播训练指标。适用于**远程/上位机触发训练**、**无终端环境下的训练监控**、**多客户端同时观测训练曲线**等场景。

---

## 目录

- [概述](#概述)
- [功能特性](#功能特性)
- [架构概览](#架构概览)
  - [双 Socket 设计](#双-socket-设计)
  - [线程模型](#线程模型)
- [环境依赖](#环境依赖)
- [目录结构](#目录结构)
- [配置文件说明](#配置文件说明)
  - [完整配置示例](#完整配置示例)
  - [字段说明](#字段说明)
  - [ROI 配置](#roi-配置)
- [启动服务](#启动服务)
- [客户端命令](#客户端命令)
- [指标订阅](#指标订阅)
- [命令协议](#命令协议)
- [推送消息格式](#推送消息格式)
- [ROI 裁剪实现](#roi-裁剪实现)
- [暂停与续训实现](#暂停与续训实现)
- [输出目录与产物](#输出目录与产物)
- [PyInstaller 打包](#pyinstaller-打包)
- [已知限制](#已知限制)

---

## 概述

`train_service` 把一次 YOLO 训练封装为**可远程控制的长驻服务**。服务启动后绑定两个 TCP 端口：

- **REP 端口（port + 1）**：接收控制命令，同步返回 JSON 结果
- **PUB 端口（port）**：广播训练指标，任意数量客户端可同时订阅

训练在独立 daemon 线程中执行，命令循环不被阻塞，因此在训练过程中随时可以查询状态或请求中断。

---

## 功能特性

| 特性 | 说明 |
|------|------|
| **远程训练控制** | `train` / `pause` / `holdon` / `stop` / `status` 五个命令，通过 ZMQ REQ-REP 同步下发 |
| **实时指标广播** | 基于 ZMQ PUB/SUB，每个 epoch 结束推送 loss、P、R、mAP50、mAP50-95 |
| **多客户端订阅** | SUB 模式可同时开多个终端观测同一训练过程，互不干扰 |
| **断点续训** | `pause` 中断后 `holdon` 以 `resume=True` 从 `last.pt` 恢复，复用原输出目录 |
| **ROI 裁剪训练** | 可选的 `useRoi`，只在指定矩形区域内采样训练样本，注入到 transforms 链首 |
| **信号优雅退出** | `SIGINT` / `SIGTERM` 触发关闭，socket `LINGER=0` 保证不挂起 |
| **输出目录隔离** | 结果固定落在配置文件同级的 `output/` 目录，`alias` 控制子目录名 |
| **PyInstaller 打包** | 附带 `train_service.spec`，可产出独立可执行文件 |

---

## 架构概览

```
 客户端 (REQ)                     订阅端 (SUB)
 python main.py --cmd train       python main.py --subscribe
        │                                 ▲
        │  tcp://localhost:9258           │  tcp://localhost:9257
        ▼                                 │
┌───────────────────────────────────────────────────────────┐
│  main.py  →  TrainService.start_server()                  │
│                                                           │
│  ┌──────────────┐    poll(100ms)    ┌──────────────────┐   │
│  │ REP socket   │◄────────────────► │ _handle_command  │   │
│  │ port + 1     │                   │  train/pause/    │   │
│  └──────────────┘                   │  holdon/stop/    │   │
│                                     │  status          │   │
│  ┌──────────────┐                   └────────┬─────────┘   │
│  │ PUB socket   │ ◄──── _publish() ──────────┤             │
│  │ port         │                            ▼             │
│  └──────────────┘                   ┌──────────────────┐   │
│                                     │ daemon Thread    │   │
│                                     │ _train_loop()    │   │
│                                     │   YOLO.train()   │   │
│                                     │   callbacks:     │   │
│                                     │   on_train_start │   │
│                                     │   on_fit_epoch_end│  │
│                                     │   on_train_end   │   │
│                                     └──────────────────┘   │
└───────────────────────────────────────────────────────────┘
```

### 双 Socket 设计

| Socket | 类型 | 绑定地址 | 方向 | 用途 |
|--------|------|----------|------|------|
| `self.pub` | PUB | `tcp://*:port`（默认 9257） | 服务端 → 所有订阅者 | 广播 `metrics` / `error` / `train_end` |
| `self.rep` | REP | `tcp://*:port + 1`（默认 9258） | 客户端 ↔ 服务端 | 同步命令与响应 |

命令循环用 `zmq.Poller` 以 `timeout=100ms` 轮询 REP socket，因此：

- 训练线程崩溃或阻塞不会卡死命令循环
- `stop_server()` 置 `_running=False` 后最多 100ms 内退出循环

### 线程模型

- **主线程**：`start_server()` 命令循环，只做收发 JSON 和状态变更
- **训练线程**：`start_train()` / `holdon_train()` 每次新建 `threading.Thread(daemon=True)`，目标函数 `_train_loop()`
- **状态共享**：`self.training` / `self.paused` / `self.trainer` / `self._resume` 由主线程写、训练线程读，无锁（依赖 GIL 与单写者约定）
- **训练结束**：`_train_loop()` 的 `finally` 必置 `self.training = False` 并 `_train_done.set()`

---

## 环境依赖

```
Python >= 3.8
pyzmq >= 22
ultralytics >= 8.0
torch >= 1.8
numpy
```

安装：

```bash
pip install pyzmq ultralytics
```

> `train_service.py` 直接 import `ultralytics.data.augment`、`ultralytics.data.build`、`ultralytics.models.yolo.detect.train` 等内部模块，ultralytics 版本升级时这些私有 API 可能变动，升级前需回归验证。

---

## 目录结构

```
train_service/
├── main.py                  # 入口：argparse + 三种模式（server / client / subscribe）
├── train_service.py         # TrainService 主体 + ROI 相关类
├── train_service.spec       # PyInstaller 打包描述文件
├── build/                   # PyInstaller 中间产物
├── dist/                    # PyInstaller 打包输出
├── __pycache__/
├── output/                  # 训练输出（运行时自动创建，位于配置文件同级）
└── README.md
```

---

## 配置文件说明

配置为 JSON 文件，所有训练参数集中在 `train` 节点下。服务端启动时读取一次，**运行期间修改不生效，需重启服务**。

`--cmd` / `--subscribe` 模式也读取该文件，但**只用到 `train.port`**（用于定位服务端口）。

### 完整配置示例

```json
{
  "train": {
    "port": 9257,
    "modelPath": "yolo26n.pt",
    "dataYaml": "coco8.yaml",
    "epochs": 100,
    "batch": 16,
    "imgsz": 640,
    "device": "auto",
    "amp": true,
    "cos_lr": false,
    "optimizer": "auto",
    "task": "detect",
    "alias": "exp_v1",
    "useRoi": false,
    "roi": { "x": 0, "y": 0, "width": 1920, "height": 1080 }
  }
}
```

### 字段说明

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `port` | int | `9257` | PUB 端口；REP 端口固定为 `port + 1` |
| `modelPath` | string | `"yolo26n.pt"` | 模型权重路径，`.pt` 或模型 YAML |
| `dataYaml` | string | `"coco8.yaml"` | 数据集配置文件 |
| `epochs` | int | `100` | 训练轮数 |
| `batch` | int | `16` | 批大小，`-1` 触发 AutoBatch |
| `imgsz` | int | `640` | 输入图像尺寸 |
| `device` | string | `"auto"` | `"auto"` 时自动选：有 CUDA 用 `"0"`，否则 `"cpu"`；也可填 `"0"`、`"0,1"`、`"cpu"` |
| `amp` | bool | `true` | 混合精度训练 |
| `cos_lr` | bool | `false` | 余弦退火学习率调度 |
| `optimizer` | string | `"auto"` | 优化器，`auto` / `SGD` / `AdamW` / `Adam` |
| `task` | string | `"detect"` | 任务类型；**ROI 模式下仅 `detect` 可用** |
| `alias` | string | `""` | 输出子目录名（映射到 `name`）；为空时由 ultralytics 自动生成 `train`、`train2`… |
| `useRoi` | bool | `false` | 是否启用 ROI 裁剪训练 |
| `roi` | object | — | ROI 矩形，见下节 |

**未暴露的常用超参**：`workers`、`patience`、`lr0`、`close_mosaic`、`seed`、`cache`、`save_period` 等目前只能走 ultralytics 默认值。如需调整，要改 `train_service.py` 中 `_train_loop()` 的 `kw` 字典（完整参数含义可参考仓库根目录 `train_param.md`）。

### ROI 配置

```json
"useRoi": true,
"roi": { "x": 512, "y": 256, "width": 1280, "height": 768 }
```

| 字段 | 说明 |
|------|------|
| `x`, `y` | ROI 左上角像素坐标（原图坐标系） |
| `width`, `height` | ROI 宽高，必须 **> 0** 才会启用 |

启用条件（三者同时满足）：`useRoi == true`、`roi` 含齐 `x/y/width/height`、`width > 0 and height > 0`。否则自动回退为普通训练。

---

## 启动服务

```bash
# 服务端（阻塞运行，Ctrl+C 退出）
python main.py --config config.json
```

启动日志：

```
... [INFO] main: PUB bound to tcp://*:9257
... [INFO] main: REP bound to tcp://*:9258
... [INFO] train_service: Train service started. Waiting for commands...
```

---

## 客户端命令

```bash
python main.py --config config.json --cmd <command>
```

| 命令 | 作用 | 前置状态要求 |
|------|------|--------------|
| `train` | 启动一次新训练 | 当前未在训练 |
| `pause` | 中断当前训练并保存 checkpoint | 正在训练 |
| `holdon` | 从上次 checkpoint 续训 | 处于已暂停状态 |
| `stop` | 彻底停止训练（不可续） | 正在训练 |
| `status` | 查询服务状态 | 无 |

响应为 JSON，直接打印到终端：

```bash
$ python main.py --config config.json --cmd status
{
  "status": "ok",
  "training": true,
  "paused": false,
  "save_dir": "output/exp_v1",
  "last_ckpt": "output/exp_v1/weights/last.pt"
}
```

```bash
$ python main.py --config config.json --cmd train
{
  "status": "ok",
  "message": "Training started"
}
```

失败时返回：

```json
{ "status": "error", "message": "Already training" }
```

---

## 指标订阅

```bash
python main.py --config config.json --subscribe
```

按 Ctrl+C 退出订阅，不影响服务端运行。输出示例：

```
Epoch   12/100 | loss: 2.4130 (box: 1.8021 cls: 0.4107 dfl: 0.2002) | P: 0.7812  R: 0.7345  mAP50: 0.8103  mAP50-95: 0.5210
Epoch   13/100 | loss: 2.3774 (box: 1.7740 cls: 0.4021 dfl: 0.2013) | P: 0.7890  R: 0.7402  mAP50: 0.8177  mAP50-95: 0.5268
[DONE] Results saved to: output/exp_v1
```

**注意**：PUB/SUB 有慢启动慢订阅问题 —— 订阅端在 `SUBSCRIBED` 之前发出的消息会被丢弃。应先启动订阅端，再下发 `train` 命令，否则会错过开头几个 epoch。

---

## 命令协议

**请求**（REQ → REP，`tcp://localhost:<port+1>`）：

```json
{ "cmd": "train" }
```

**响应**（REP → REQ）：

```json
{ "status": "ok", "message": "Training started" }
```

**`status` 响应字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | `"ok"` / `"error"` |
| `training` | bool | 是否有训练线程在运行 |
| `paused` | bool | 是否处于暂停状态（可 `holdon`） |
| `save_dir` | string \| null | 当前/上次训练输出目录 |
| `last_ckpt` | string \| null | `last.pt` 路径 |

**异常兜底**：`_handle_command()` 对所有 handler 包一层 `try/except`，任何异常都会转成 `{"status": "error", "message": ...}` 返回，不会让服务端崩溃。未知命令同样返回 error。

---

## 推送消息格式

PUB socket 采用 `[topic][json]` 两帧 multipart，topic 为 UTF-8 字符串。

### topic: `metrics`（每个 epoch 结束推送一次）

| 字段 | 类型 | 说明 |
|------|------|------|
| `epoch` | int | 当前轮次（从 1 开始） |
| `total_epochs` | int | 总轮数 |
| `loss` | float | 总 loss（`trainer.loss`） |
| `box_loss` / `cls_loss` / `dfl_loss` | float | 分项 loss（`trainer.tloss[0/1/2]`） |
| `precision` | float | `metrics/precision(B)` |
| `recall` | float | `metrics/recall(B)` |
| `mAP50` | float | `metrics/mAP50(B)` |
| `mAP50_95` | float | `metrics/mAP50-95(B)` |

### topic: `error`

| 字段 | 类型 | 说明 |
|------|------|------|
| `message` | string | 训练线程捕获的异常文本 |

### topic: `train_end`

| 字段 | 类型 | 说明 |
|------|------|------|
| `save_dir` | string | 训练结果目录 |

> 指标数值取自 ultralytics 的 `trainer.metrics`，未跑验证的 epoch 会保持上一轮值或 0。

---

## ROI 裁剪实现

核心思路：**不改数据集文件，而是在 transform 链首插入一个裁剪步骤**，同时裁剪图像和修正标注框。

### 类继承链

```
BaseTransform
└── ROICrop                    # 纯函数式 transform：裁图 + 裁框 + 重归一化

YOLODataset
└── ROIDataset                 # build_transforms() 中 insert(0, ROICrop)

DetectionTrainer
└── ROITrainer                 # build_dataset() 返回 ROIDataset，并下推 ROI 参数
```

### 数据流

```
__getitem__(index)
    │  load_image() 读入完整原图
    ▼
Compose([ ROICrop, Mosaic, RandomPerspective, ..., LetterBox, Format ])
           ↑ insert(0)
    │
    ├─ apply_image():   img[y1:y2, x1:x2] 裁剪
    └─ apply_instances():
         1. denormalize → convert_bbox("xyxy")       还原绝对坐标
         2. 与 ROI 矩形求交，保留相交的框（IoU > 0）
         3. clip 到 ROI 内并平移原点到 (x1, y1)
         4. 完全落在 ROI 外的框直接丢弃
         5. convert_bbox("xywh") → normalize(bw, bh)  重新归一化
```

### 参数传递

ROI 值通过**类属性**（`ROIDataset._roi` / `ROITrainer._roi`）传递，因为 ultralytics 构造 dataset 时不支持额外 kwargs：

```python
# train_service.py:241-248
ROITrainer.set_roi(rx, ry, rw, rh)
kw["trainer"] = ROITrainer        # model.train(trainer=ROITrainer, ...)
# ROITrainer.build_dataset() 内
ROIDataset.set_roi(*self._roi)    # 再下推给 dataset
```

### 行为要点

- **只在训练集生效**：`build_transforms` 中的条件是 `if self._roi and self.augment`，`augment` 仅 `mode == "train"` 为真，验证集不裁剪。
- **验证集不裁剪**：导致训练视角与验证视角不一致，mAP 反映的是全图效果。
- **Mosaic 的另外 3 张图不受影响**：`Mosaic.get_indexes()` 走 `dataset.load_image()` 直接读原图，不经过 `ROICrop`；只有当前样本这张图被裁。
- **越界安全**：`apply_image` 对 ROI 做了 `clip` 到图像边界，不会 IndexError。

---

## 暂停与续训实现

ultralytics 没有原生"暂停"，这里用 **`trainer.stop = True` 中断 + `resume=True` 重启** 模拟。

```
train  ──►  _train_loop()  ──►  YOLO.train(resume=False)
                                     │
pause  ──►  trainer.stop = True ─────┤  当前 epoch 结束后 break
                                     │  final_eval() → 保存 last.pt
                                     │  on_train_end → 广播 train_end
                                     ▼
                              training=False, paused=True
                                     │
holdon ──►  join(old thread, 30s) ───┤  _resume = True
                                     │  新建训练线程
                                     ▼
                              YOLO.train(resume=True)
                              project/name 复用上次的值
```

**语义澄清（重要）**：

| 命令 | 实际行为 | 可否续训 |
|------|----------|----------|
| `pause` | **终止**当前训练进程，保存 `last.pt`，触发 `on_train_end` | 是，用 `holdon` |
| `holdon` | 用 `resume=True` 重启，从 `last.pt` 接着训 | — |
| `stop` | 同样终止训练，但 `paused` 置为 `False` | **否** |

因此：

- `pause` 不是挂起进程，订阅端在暂停时会收到一条 `train_end` 消息 —— 这是预期行为，不代表训练真正完成。
- `stop` 之后 `holdon` 会返回 `Not paused` 错误。若此时再发 `train`，会以 `resume=False` 从头开始，且因为 `exist_ok=True` **覆盖同一输出目录**。

**续训时的目录复用**（`train_service.py:250-252`）：

```python
if resume and self._saved_project and self._saved_name:
    kw["project"] = self._saved_project
    kw["name"] = self._saved_name
```

`_saved_project` / `_saved_name` 在 `on_train_start` 回调中记录，保证续训写回同一目录。

---

## 输出目录与产物

输出根目录 = **配置文件所在目录下的 `output/`**（`config_path` 为空时为相对路径 `output/`）。

```
<config 所在目录>/output/<alias>/
├── weights/
│   ├── best.pt          # 最优权重
│   └── last.pt          # 最新权重（续训依据）
├── results.csv          # 每轮指标
├── args.yaml            # 实际生效的训练参数
├── confusion_matrix.png
├── PR_curve.png
├── F1_curve.png
├── results.png
└── train_batch*.jpg     # 训练样本可视化（plots=True）
```

`plots=True` 与 `exist_ok=True` 在 `kw` 中硬编码，不可配置。

---

## PyInstaller 打包

### 打包命令

```bash
pyinstaller train_service.spec
```

`train_service.spec` 说明：

| 配置 | 值 | 含义 |
|------|-----|------|
| 入口 | `main.py` | 仅打包入口，`train_service.py` 由 import 自动收入 |
| `console` | `True` | 保留控制台（服务端需要看日志） |
| `exclude_binaries` | `True` + `COLLECT` | **onedir** 模式，输出 `dist/train_service/` |
| `upx` | `True` | 启用 UPX 压缩（未装 UPX 时自动跳过） |
| `datas` / `binaries` | 空 | 不额外打包数据文件 |

### 打包注意事项

1. **模型权重和数据集 YAML 不会打包**，需在配置里用绝对路径，或手动复制到输出目录旁
2. **ultralytics 的 YAML 模型配置**（`cfg/models/*.yaml`）同样在运行时读取，需保证可访问
3. **首次运行会创建 `~/.config/Ultralytics/`** 配置缓存目录
4. **PyInstaller + torch**：如遇 `.dist-info` 元数据读取的 `UnicodeDecodeError`，参见 `README_detect_server_slidewind.md` 中的 `importlib.metadata` monkey-patch 方案

---

## 已知限制

1. **训练/验证视角不一致** — ROI 只作用于 `augment=True` 的训练集，验证集走全图。若 ROI 外存在大量背景目标，验证指标会偏低；若 ROI 外基本无目标，验证指标会虚高。
2. **Mosaic 混入未裁剪样本** — Mosaic 的另外 3/8 张图直接 `load_image()` 原图，只有当前样本被 ROI 裁剪，导致 mosaic 拼图里"有的块裁了、有的块没裁"。彻底解决需同时改 `Mosaic` 的取图路径。
3. **ROI 仅支持 detect** — `ROITrainer` 继承 `DetectionTrainer`，`task` 设为 `segment`/`pose`/`obb` 时 ROI 不会按预期工作。
4. **`stop` 后不可续训，且重训会覆盖** — `stop` 不置 `paused`，`holdon` 会拒绝；再次 `train` 因 `exist_ok=True` 写回同一目录，可能覆盖上一次结果。想保留现场用 `pause`。
5. **`start_train` 与 `on_train_start` 之间的竞态** — 若在训练线程刚启动、`self.trainer` 尚未赋值时下发 `pause`，`trainer.stop` 不会被设置，`paused` 却已置 True。此时 `holdon` 的 `join(timeout=30)` 超时后仍会启动第二个训练线程，与原线程争抢 GPU。窗口很短但存在，无锁保护。
6. **配置不可热更新** — 启动时读一次 JSON，改 `port` 必须重启服务；改训练超参同样需重启（`train_cfg` 引用的是启动时解析的 dict）。
7. **超参覆盖不全** — `workers`、`patience`、`lr0`、`close_mosaic`、`seed`、`cache`、`save_period` 未暴露，只能改代码。
8. **PUB 慢订阅丢消息** — 订阅端晚于训练启动连接时，会丢失此前的 `metrics`；且 PUB 无确认机制，网络抖动可能丢帧。
9. **状态无持久化** — 服务重启后 `training`/`paused` 归零，需要 `status` 无法得知重启前的训练进度（只能从 `output/` 目录推断）。
10. **依赖 ultralytics 内部 API** — `BaseTransform`、`YOLODataset`、`DetectionTrainer`、`unwrap_model` 等均为私有/半私有接口，升级时可能失效；另有 7 个未使用的 import（`numpy`、`Compose`、`LetterBox`、`Format`、`build_yolo_dataset`、`build_dataloader`、`LOGGER`）可清理。
