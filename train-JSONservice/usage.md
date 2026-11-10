# 使用手册

面向「怎么跑、怎么 build、怎么调用、怎么通讯」。概念、架构与设计取舍见 [README.md](README.md)。

---

## 目录

- [一、怎么跑](#一怎么跑)
- [二、怎么 build](#二怎么-build)
- [三、怎么调用](#三怎么调用)
- [四、怎么通讯](#四怎么通讯)
- [排障](#排障)

---

## 一、怎么跑

### 1.1 环境准备

```powershell
# 本机 conda 环境
conda activate ultraylytics

# 关键：仓库内 ultralytics 是 8.4.102，site-packages 里可能是旧版
# yolo26n.pt 等新权重只有 8.4.102 能解析，必须把仓库根加进 PYTHONPATH
$env:PYTHONPATH = "E:\m_code\ultralytics-main"

cd E:\m_code\ultralytics-main\train-JSONservice
```

> 不设 `PYTHONPATH` 的典型症状：`SPPF.__init__() takes from 3 to 4 positional arguments but 6 were given`。
> 这是权重结构与 ultralytics 版本不匹配，不是本服务的 bug。

### 1.2 三条命令

```powershell
# ① 跑一次训练（同步，父进程活到训练结束）
python main.py --config config.json

# ② 看上次训练状态（不启动训练）
python main.py --config config.json --status

# ③ 续训
python main.py --config config.json --resume                 # 从 state 推导 weights/last.pt
python main.py --config config.json --resume E:/runs/exp1/weights/last.pt
```

**停止**：训练中按 `Ctrl+C` → 杀掉整个进程树 → 退出码 `130`。

### 1.3 配置

单层 JSON，`train` 对象放训练参数：

```json
{
  "train": {
    "model": "E:/m_code/ultralytics-main/yolo26n.pt",
    "data":  "D:/m_code/ultralytics-main/ultralytics/NEU-DET/data.yaml",
    "epochs": 100,
    "batch": 16,
    "imgsz": 640,
    "device": "auto",
    "task": "detect",
    "project": "E:/m_code/ultralytics-main/train-JSONservice/output",
    "name": "exp1",
    "exist_ok": true,
    "workers": 4,
    "useRoi": false,
    "roi": { "x": 640, "y": 320, "width": 1280, "height": 720 }
  }
}
```

完整字段与白名单见 [README - 配置字段](README.md#配置字段)。从样例复制一份改路径即可：

```powershell
Copy-Item config.json.example config.json
```

**旧 ZMQ 服务的配置字段直接可用**，解析时自动归一化：

| 旧字段 | 归一化为 |
|---|---|
| `modelPath` | `model` |
| `dataYaml` | `data` |
| `alias` | `name` |

新字段名优先（两者同时出现取新的）。`port` 会被忽略（新服务暂无 ZMQ）。
归一化在 `load_train_config()` 里完成，不在 `build_train_args()` 里——`--status` 与 `--resume`
读的 `project` / `name` 才能拿到 `alias` 归一化后的值。
详见 [README - 旧字段别名](README.md#旧字段别名兼容-train_zmqservice)。

> 旧配置常缺 `project`——旧服务固定写到 `<config 所在目录>/output/<alias>/`，
> 新服务没有这个逻辑，不给 `project` 就落到 ultralytics 默认的 `runs/`，
> 状态文件也会跟着跑到 `runs/service_state.json`，`--resume` 会找错地方。

### 1.4 单通道图与浮点 ROI

灰度 `.tif` 直接可用，不用转数据。上游 `imread` 的 tif 分支硬编码 `IMREAD_UNCHANGED`、无视 `flags`，
会把单通道还原成 `(H, W, 1)`，导致第一个 conv 层报 `expected input[16, 3, 3, 3] ... got 1 channels`。
`worker.install_gray_shim()` 在读取层把 `(H, W, 1)` 补成 `(H, W, 3)`，装在 **worker 模块级**
（Windows DataLoader 走 `spawn`，子进程重跑模块代码，装在函数里 worker 拿不到补丁）。
完整说明与限制见 [README - 单通道图像兼容](README.md#单通道图像兼容)。

`roi` 四项进 `roi.py` 前统一 `int(round(float(x)))`——anno_studio 的 `QRectF` 传的是浮点，
按像素切片时浮点下标会 `TypeError: slice indices must be integers`。

### 1.5 输出长什么样

```
Ultralytics 8.4.102  Python-3.10.18 torch-2.1.0+cu118 CUDA:0 (NVIDIA GeForce RTX 4060 Laptop GPU, 8188MiB)
engine\trainer: ... epochs=100, batch=16, imgsz=640 ...
[START] epochs=100 save_dir=E:\...\output\exp1
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/100     0.518G      2.373      4.665    0.05289         15        640: 100% ━━━━━━━━ 45/45 4.2it/s
[training_metrics] {"epoch": 1, "total_epochs": 100, "loss": 2.41, "box_loss": 2.12, "cls_loss": 0.28, "dfl_loss": 0.04, "precision": 0.61, "recall": 0.55, "mAP50": 0.58, "mAP50_95": 0.41, "task": "detect"}
...
100 epochs completed in 0.42 hours.
[DONE] Training completed successfully
```

- 进度条用 `\r` 原地刷新，普通日志前会先清一次进度条残留
- `[START]` / `[training_metrics]` / `[DONE]` / `[STOPPED]` / `[ERROR]` 是本服务的事件标记，
  其余是 ultralytics 原始输出（`training_log`）

### 1.6 退出码

| 码 | 含义 | 对应终态事件 |
|---|---|---|
| `0` | 训练正常完成 | `training_completed` |
| `1` | 训练/配置/协议出错 | `training_error` |
| `130` | 被 Ctrl+C 停止 | `training_stopped` |

### 1.7 状态文件

每个有意义的事件都会更新 `<project>/<name>/service_state.json`
（`training_log` / `training_metrics` 不写盘，理由见 [4.3](#43-通道-c状态文件共享)）：

```json
{
  "event": "training_completed",
  "updated_at": "2026-11-10T11:47:53",
  "save_dir": "E:/m_code/ultralytics-main/train-JSONservice/output/exp1",
  "results": "Training completed successfully"
}
```

`--status` 打印的就是这个文件。服务重启后靠它恢复 `save_dir`（供 `--resume` 用）。

---

## 二、怎么 build

> 本节的打包产物运行验证见文末 [2.5 打包产物自检](#25-打包产物自检)。

### 2.1 打包命令

```powershell
cd E:\m_code\ultralytics-main\train-JSONservice
$env:PYTHONPATH = "E:\m_code\ultralytics-main"   # 打包进去的必须是 8.4.102
python -m PyInstaller train_json_service.spec --noconfirm --clean
```

PyInstaller 已装在 conda 环境（6.19.0）。

### 2.2 spec 说明

`train_json_service.spec`（onedir 模式，与 `train_ZMQservice/train_service.spec` 结构一致）：

| 配置 | 值 | 含义 |
|---|---|---|
| 入口 | `main.py` | CLI 入口 |
| `hiddenimports` | `worker`, `roi`, `protocol`, `manager` | **必须显式声明**，见下 |
| `console` | `True` | 保留控制台，训练日志要打到 stdout |
| `exclude_binaries` + `COLLECT` | `True` | onedir 模式，输出 `dist/train_json_service/` |
| `upx` | `True` | 有 UPX 就压缩，没装自动跳过 |
| `datas` / `binaries` | 空 | 权重、数据集 YAML 不打进包 |

> **为什么 `hiddenimports` 要写死**：`worker.py` 和 `roi.py` 都不在 `main.py` 的静态 import 链上
> （`worker` 由 `manager` 用 `subprocess` 启动、`roi` 在 `worker.build_roi_trainer()` 函数体内动态 import）。
> PyInstaller 只做静态分析，不写进去就会在运行时 `ModuleNotFoundError`。

### 2.3 产物

```
train-JSONservice/
├── build/train_json_service/   # 中间产物（可删）
└── dist/train_json_service/    # 交付目录
    ├── train_json_service.exe
    └── _internal/              # 依赖、PYZ 里的模块
```

### 2.4 打包后注意

1. **权重、数据集 YAML、字体都不会打进去**。配置里一律用绝对路径，或把文件放到 exe 旁边
2. **绘图字体**：ultralytics 找 `Arial.ttf` 的顺序是 `~/.config/Ultralytics/` → 系统字体 → 联网下载。
   离线机器第一次画图会失败，提前把 `Arial.ttf` 放进 `%USERPROFILE%\.config\Ultralytics\`
3. **`~/.config/Ultralytics/` 首次运行会自动创建**，里面还有 `settings.json`（`datasets_dir` 等）。
   注意 `datasets_dir` 默认是 `C:\Users\datasets`，不可写时数据集自动下载会 `WinError 5`
4. **PyInstaller + torch**：遇到 `.dist-info` 的 `UnicodeDecodeError`，
   参考 `README_detect_server_slidewind.md` 里的 `importlib.metadata` monkey-patch
5. **本机 `site-packages` 与仓库版本不一致**：打包时务必带 `PYTHONPATH`，
   否则打进去的是旧版 ultralytics，运行时报 `SPPF.__init__()` 参数不匹配

### 2.5 打包产物自检

在本机实测（onedir，约 30 MB exe + `_internal/`）：

| 命令 | 结果 |
|---|---|
| `train_json_service.exe --help` | 正常，`--worker` 不出现在 usage 里 |
| `... --config x.json --status` | 正常打印 `service_state.json`，退出码 0 |
| `... --config x.json` | **正常训练**：`[START]` → `[training_metrics]` → `[DONE]`，退出码 0 |
| `... --config x.json --resume` | 路径推导正确（`model=` 指向 `weights/last.pt`） |
| 无参数 | argparse 正常报 `--config required` |

**打包后子进程怎么起来的**：`manager.build_worker_command()` 检测 `sys.frozen`——

```python
if getattr(sys, "frozen", False):
    return [sys.executable, "--worker", str(payload_path)]   # exe 重入自己当 worker
worker = Path(__file__).with_name("worker.py")
return [sys.executable, "-u", str(worker), str(payload_path)]  # 源码模式
```

`main.py` 在 argparse 之前分流 `--worker`，转调 `worker.main(payload)`。

> **为什么必须这样**：打包后 `sys.executable` 是 exe 本身、`worker.py` 也不在产物里
> （它不被 `main.py` 静态 import）。若照搬源码模式的 `[sys.executable, "-u", worker.py, payload]`，
> exe 会被当成 worker 重新启动，argparse 报 `--config required` 直接退出——
> 症状是 `[ERROR] Training process exited with code 2`。
>
> 另外 exe 重入时 stdout 是**全缓冲**的（`-u` 不存在了），
> 所以 `worker.main()` 里显式 `reconfigure(line_buffering=True)`，否则事件不会实时回传。

**打包版仍不随包携带**：权重、数据集 YAML、字体（见 2.4）。

**父进程被强杀时子进程不会被带走**（任务管理器结束 exe / `kill -9`）。
只有走 `stop_training()` 或 Ctrl+C 才会 `taskkill /T` 杀整棵树。这条与源码模式一致。

---

## 三、怎么调用

三层调用方式，从外到内：

| 层 | 方式 | 适用 |
|---|---|---|
| 外 | CLI（`main.py`） | 人工跑一次、脚本串联 |
| 中 | Python API（`TrainingManager`） | 常驻服务、被其它程序嵌入 |
| 内 | 子进程协议（`worker.py` + payload） | 自己写父进程/接口层 |

### 3.1 CLI 调用

```powershell
python main.py --config <path>              # 跑训练
python main.py --config <path> --status     # 只看状态
python main.py --config <path> --resume [CKPT]
```

| 参数 | 必填 | 说明 |
|---|---|---|
| `--config` | 是 | 配置 JSON 路径 |
| `--status` | 否 | 打印 `service_state.json` 后退出，不启动训练 |
| `--resume [CKPT]` | 否 | 不带路径 = 从 state 的 `save_dir` 推导 `weights/last.pt` |

### 3.2 Python API 调用

`TrainingManager` 是常驻服务的正确入口。CLI 只是它的一层外壳。

```python
import json
import sys

sys.path.insert(0, r"E:\m_code\ultralytics-main\train-JSONservice")

from manager import TrainingManager, read_state

EVENTS = {
    "training_completed": "done",
    "training_error": "failed",
    "training_stopped": "stopped",
}


def on_event(event, data):
    """callback 签名固定为 (event: str, data: dict)。

    抛异常不会打断训练——notify_callbacks 内部已 try/except。
    """
    if event == "training_started":
        print(f"started, save_dir={data['save_dir']}")
    elif event == "training_log":
        print(data["message"], end=" " if "\r" in data["message"] else "\n")
    elif event in EVENTS:
        print(f"terminal: {event} {data}")
    # 其余事件（protocol_error 等）自行处理


train_args = {
    "model": r"E:\m_code\ultralytics-main\yolo26n.pt",
    "data": r"D:\m_code\ultralytics-main\ultralytics\NEU-DET\data.yaml",
    "epochs": 100,
    "batch": 16,
    "imgsz": 640,
    "device": "auto",
    "task": "detect",
    "project": r"E:\runs",
    "name": "exp1",
    "exist_ok": True,
    "workers": 4,
}

manager = TrainingManager(callbacks=[on_event])

# 1) 起训练：同步返回，训练在子进程里跑
ok, message = manager.start_training(train_args)
if not ok:
    raise SystemExit(message)

# 2) 可以在这里干别的；训练不阻塞当前线程
manager.is_training      # -> True
manager.save_dir         # -> 训练开始后为 save_dir 字符串，之前为 None

# 3) 等它结束（或自己择机 stop）
manager.wait()           # 阻塞到终态；也可 manager.wait(timeout=3600)

# 4) 想停
manager.stop_training()  # 杀整个进程树，终态为 training_stopped
manager.wait(timeout=90)

# 5) 事后读状态
print(json.dumps(read_state("E:/runs", "exp1"), ensure_ascii=False, indent=2))
```

#### `TrainingManager` API

| 成员 | 签名 | 说明 |
|---|---|---|
| 构造 | `TrainingManager(callbacks=None)` | `callbacks` 是 `[(event, data) -> None]` |
| `start_training` | `(train_args, options=None) -> (bool, str)` | 已在训练时返回 `(False, "Training is already in progress")` |
| `stop_training` | `() -> bool` | 未在训练返回 `False`；异步杀进程树，需配合 `wait()` |
| `wait` | `(timeout=None) -> bool` | 等到本轮结束；返回是否在超时前结束 |
| `is_training` | `bool` | 是否有训练在跑 |
| `save_dir` | `str \| None` | 取自 `training_started` 事件 |
| `notify_callbacks` | `(event, data) -> None` | 手动广播一个事件给所有 callback |

#### 约定

- **单实例**：同一时刻只能有一个训练，第二次 `start_training()` 直接拒绝
- **一轮恰好一个终态事件**：`training_completed` / `training_error` / `training_stopped` 三选一
- **状态文件不会被日志刷爆**：`training_log` / `training_metrics` 只走 callback，不写 `service_state.json`
- **callback 里别做重活**：`training_log` 每行文本都会回调一次，回调会拖慢子进程读取
- **payload 文件用完即删**：由 `manager` 在本轮结束时 `os.remove`，异常退出会残留 `train-json-*.json` 在系统 TEMP

### 3.3 子进程协议调用（自己写父进程）

如果你想接 ZMQ / gRPC / HTTP，不必改本服务，直接按 [第四节](#四怎么通讯) 的协议自己包一层。

最小可用的父进程骨架：

```python
import subprocess, sys
from protocol import create_payload, parse_line, TERMINAL_EVENTS

payload_path = create_payload("train", {"model": "...", "data": "...", "epochs": 1})

proc = subprocess.Popen(
    [sys.executable, "-u", r"E:\...\train-JSONservice\worker.py", payload_path],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, encoding="utf-8", errors="replace", bufsize=1,
)

for line in proc.stdout:
    parsed = parse_line(line)
    if parsed is None:
        continue
    event, data = parsed
    if event in TERMINAL_EVENTS:
        print("terminal:", event, data)
        break
    # 转发到你自己的通道
```

协议工具函数：

| 函数 | 说明 |
|---|---|
| `create_payload(action, args, paths=None, options=None) -> str` | 写临时 JSON，返回路径 |
| `load_payload(path) -> dict` | 读 + 七层校验，非法直接 `ValueError` |
| `emit(event, **data) -> None` | 子进程侧发事件到 stdout |
| `parse_line(line) -> (event, data) \| None` | 父进程侧解析一行；空行返回 `None` |
| `EventLogStream(event="training_log")` | 把子进程 stdout 文本切成事件（`\n` / `\r` 都切） |
| `read_state(project, name) -> dict \| None` | 读状态文件，损坏返回 `None` |
| `state_path(project, name) -> Path` | 状态文件路径 |

---

## 四、怎么通讯

本服务有三条通道，**互相独立**：

```
┌─ 通道 A：payload JSON（进）──────────────────────┐
│  父进程 ──写临时文件──▶ 子进程                    │
│  路径：sys TEMP / train-json-*.json，用完即删     │
└──────────────────────────────────────────────────┘

┌─ 通道 B：stdout 事件流（出）─────────────────────┐
│  子进程 ──管道──▶ 父进程 ──callback──▶ 你          │
│  事件行与普通日志行混在同一管道                     │
└──────────────────────────────────────────────────┘

┌─ 通道 C：磁盘状态（共享）────────────────────────┐
│  <project>/<name>/service_state.json             │
│  父进程写，任何进程可读，是事实来源                 │
└──────────────────────────────────────────────────┘
```

### 4.1 通道 A：payload 报文（进）

`protocol.create_payload()` 写出的文件内容：

```json
{
  "version": 1,
  "action": "train",
  "args": {
    "model": "E:/m_code/ultralytics-main/yolo26n.pt",
    "data": "D:/.../NEU-DET/data.yaml",
    "epochs": 100,
    "batch": 16,
    "device": "auto",
    "task": "detect",
    "project": "E:/runs",
    "name": "exp1",
    "exist_ok": true,
    "workers": 4,
    "useRoi": false,
    "roi": { "x": 640, "y": 320, "width": 1280, "height": 720 }
  },
  "paths": {},
  "options": {}
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `version` | int | 必须等于 `PROTOCOL_VERSION`（当前 `1`） |
| `action` | str | 当前只有 `"train"` |
| `args` | object | 训练参数，走白名单校验 |
| `paths` | object | 预留给路径类参数，值必须是字符串 |
| `options` | object | 预留给服务级选项，当前未使用 |

**七层校验**（任一不过直接 `ValueError`，不会启动训练）：

| # | 检查 | 报错文案 |
|---|---|---|
| 1 | 版本号 | `Unsupported payload version: ...` |
| 2 | 动作 | `Unsupported payload action: ...` |
| 3 | 顶层字段白名单 | `Unsupported payload fields: [...]` |
| 4 | `args` / `paths` / `options` 必须是 object | `Payload field 'x' must be an object` |
| 5 | `args` 走训练参数白名单 | `Unsupported training arguments: [...]` |
| 6 | `model` + `data` 非空 | `Missing required training arguments: [...]` |
| 7 | `paths` 值必须是字符串 | `Payload paths must be strings` |

`useRoi` / `roi` 属于服务专属字段，在进入 `model.train()` 前会被 pop 掉，不会透传给 ultralytics。
`useRoi: true` 但 `roi` 缺省或四项全 0 时视为**未启用 ROI**，走全图训练；只有非全 0 且宽高 <= 0、缺字段才算非法。

### 4.2 通道 B：stdout 事件流（出）

**哨兵前缀**：

```
__TRAIN_SERVICE_EVENT__={"version": 1, "event": "training_started", ...}
```

- 前缀常量：`protocol.EVENT_PREFIX`
- 一行一条事件，行尾 `\n`
- **不带前缀的行一律当普通日志**，转成 `training_log` 事件

**混流规则**：

| 规则 | 实现 |
|---|---|
| 事件与日志同管道 | `stderr=STDOUT` 合并，子进程的报错也能被父进程读到 |
| 编码 | `PYTHONIOENCODING=utf-8` + `errors="replace"` + `stream.reconfigure()` |
| 切行 | 同时按 `\n` 和 `\r` 切（进度条用 `\r` 原地刷新） |
| 坏 JSON | 不丢弃，回退成 `training_log`，原样带 `message` |
| 版本不符 | 发 `protocol_error` 事件 |
| `emit` 绕过重定向 | 写 `sys.__stdout__`，子进程内部把 `sys.stdout` 换成日志流也不会吞掉事件 |

**真实报文样例**（从实测输出摘录）：

```text
__TRAIN_SERVICE_EVENT__={"version": 1, "event": "training_started", "total_epochs": 1, "save_dir": "C:\\Users\\zydon\\AppData\\Local\\Temp\\opencode\\jsonservice_out\\smoke_neu"}
__TRAIN_SERVICE_EVENT__={"version": 1, "event": "training_log", "message": "Starting training for 1 epochs..."}
__TRAIN_SERVICE_EVENT__={"version": 1, "event": "training_log", "message": "all          3          4     0.013        0.5     0.0443     0.0208"}
__TRAIN_SERVICE_EVENT__={"version": 1, "event": "training_completed", "results": "Training completed successfully"}
```

**事件表**：

| 事件 | 触发时机 | 关键字段 | 终态 |
|---|---|---|---|
| `training_started` | ultralytics `on_train_start` 回调 | `total_epochs`、`save_dir` | 否 |
| `training_metrics` | 每个 epoch 结束（`on_fit_epoch_end`） | 见下 | 否 |
| `training_log` | 子进程任意 stdout/stderr 文本行 | `message` | 否 |
| `training_completed` | 训练正常结束 | `results` | **是** |
| `training_error` | 初始化/训练抛异常、非零退出 | `error`、`exception_type`、`traceback` | **是** |
| `training_stopped` | 父进程 stop / Ctrl+C | — | **是** |
| `protocol_error` | 事件版本不符或缺 `event` | `error` | 否 |

#### `training_metrics` 字段

字段名与旧 ZMQ 的 `metrics` topic **一一对齐**，接薄壳时客户端不用改：

| 字段 | 类型 | 说明 |
|---|---|---|
| `epoch` | int | 从 1 开始 |
| `total_epochs` | int | 总轮数 |
| `loss` | float \| null | `trainer.loss` 总损失 |
| `box_loss` | float \| null | `trainer.tloss[0]`，四种任务都是 box |
| `cls_loss` | float \| null | 仅 `detect` / `obb` 填，其余 `null` |
| `dfl_loss` | float \| null | 仅 `detect` / `obb` 填，其余 `null` |
| `precision` / `recall` | float | `metrics/precision(B)`、`metrics/recall(B)` |
| `mAP50` / `mAP50_95` | float | `metrics/mAP50(B)`、`metrics/mAP50-95(B)` |
| `task` | str | 本次训练的 task，**新增字段**，旧客户端可忽略 |

实测报文：

```text
__TRAIN_SERVICE_EVENT__={"version": 1, "event": "training_metrics", "epoch": 1, "total_epochs": 1, "loss": 14.87548542022705, "box_loss": 2.3572921752929688, "cls_loss": 4.644078731536865, "dfl_loss": 0.0465957336127758, "precision": 0.01299, "recall": 0.5, "mAP50": 0.04426, "mAP50_95": 0.02079, "task": "detect"}
```

**`cls_loss` / `dfl_loss` 为什么只对 detect/obb 填**：`trainer.tloss` 的分项顺序随任务变化
（`segment` 是 `(box, seg, cls, dfl)`，`pose` 是 `(box, pose, kobj)`），按下标硬取会张冠李戴。
`box_loss` 四种任务都是 `tloss[0]`，所以始终填。宁可给 `null` 也不给错值。

**最后一条 `epoch` 可能等于 `total_epochs + 1`**：这不是重复上报。
ultralytics 的 `final_eval()`（`trainer.py:916`）会 `self.epoch += 1` 后再发一次 `on_fit_epoch_end`，
目的是把 **best.pt 的验证指标**记到 `epochs+1` 步、不覆盖最后一个 epoch。
每条的 `epoch` 都是唯一的，最后一条即 best 指标——与 `<save_dir>/results.csv` 的语义一致。
（只在 `best.pt` 存在时才会多发这一条；`save=false` 时不会。）

**终态保证**：`manager._dispatch()` 对终态事件做幂等守卫，**每轮运行恰好发出一个终态事件**。
即使在「子进程已发 `training_completed`、父进程尚未退出循环」的窗口里按下 stop，也不会再补发 `training_stopped`。

> **与旧 ZMQ 服务的对照**：`training_started` ↔ `on_train_start`、`training_metrics` ↔ `metrics`、
> `training_error` ↔ `error`、`training_completed` ↔ `train_end`，语义一一对应。
> 差别是新服务把它们统一到**一条 stdout 事件流**里（旧服务是 PUB 多 topic），且多带了 `traceback` 与 `task`。
>
> epoch 级指标也可以不走事件、直接读 `<save_dir>/results.csv`（磁盘为事实来源），两者数值一致。

### 4.3 通道 C：状态文件（共享）

路径：`<project>/<name>/service_state.json`

| 字段 | 类型 | 说明 |
|---|---|---|
| `event` | str | 最近一次事件名（通常是终态） |
| `updated_at` | str | ISO 时间，秒级 |
| `save_dir` | str \| null | 取自 `training_started` |
| `results` | str | 仅 `training_completed` 时有 |
| `error` | str | 仅 `training_error` 时有 |
| `total_epochs` | int | 仅 `training_started` 时有 |

- **不写 traceback**：避免状态文件被几十 KB 堆栈污染，traceback 只走通道 B
- 文件损坏 / 不存在时 `read_state()` 返回 `None`，不抛异常
- 有意义的事件全量重写（不是追加），永远是最新一份；
  `training_log` / `training_metrics` 被 `IGNORED_STATE_EVENTS` 排除，不写盘

### 4.4 后续接 ZMQ 的薄壳

按之前定的方向，接口层只做薄壳，不碰本服务内部。对应关系：

| 旧 ZMQ 协议 | 新服务来源 | 薄壳怎么转 |
|---|---|---|
| REQ `{"cmd":"train"}` | `manager.start_training(train_args)` | `(ok, msg)` → `{"status": "ok"/"error", "message": msg}` |
| REQ `{"cmd":"stop"}` | `manager.stop_training()` | 同上 |
| REQ `{"cmd":"status"}` | `manager.is_training` / `save_dir` + `read_state()` | 组装 `training` / `save_dir` / `last_ckpt` |
| REQ `{"cmd":"pause"}` / `"holdon"` | **无对应** | 本服务不做 pause；建议返回 error，或映射成 `stop` + `--resume` |
| PUB `metrics` | `training_metrics` | 字段一一对齐，直接转 topic `metrics` |
| PUB `error` | `training_error.error` | 直接转 topic `error` |
| PUB `train_end` | `training_started.save_dir` / `training_completed` | 转 topic `train_end` |

薄壳骨架：

```python
import zmq
from manager import TrainingManager, read_state

manager = TrainingManager(callbacks=[lambda e, d: publish(e, d)])   # 事件 → PUB

ctx = zmq.Context()
rep = ctx.socket(zmq.REP)
rep.bind("tcp://*:9258")                                            # 旧服务 REP = port + 1

while True:
    req = rep.recv_json()
    cmd = req.get("cmd")
    if cmd == "train":
        ok, msg = manager.start_training(req["args"])
    elif cmd == "stop":
        ok, msg = manager.stop_training(), "Training stopped"
    elif cmd == "status":
        ok, msg = True, {"training": manager.is_training, "save_dir": manager.save_dir,
                         **(read_state(req.get("project"), req.get("name")) or {})}
    else:
        ok, msg = False, f"Unknown command: {cmd}"
    rep.send_json({"status": "ok" if ok else "error", "message": msg})
```

> PUB 是两帧 multipart：`[topic][json]`，与旧服务一致。
> **PUB 没有回放**：先起订阅端再开训练，否则会漏掉开头的事件。

---

## 排障

| 现象 | 原因 / 处理 |
|---|---|
| `SPPF.__init__() takes from 3 to 4 positional arguments but 6 were given` | `PYTHONPATH` 没指仓库根，site-packages 是旧版 ultralytics |
| `PermissionError: [WinError 5] ... C:\Users\datasets` | ultralytics `settings.json` 的 `datasets_dir` 不可写；改设置或先放好数据集 |
| `--resume` 报 `training to N epochs is finished, nothing to resume` | 上次已跑满，checkpoint 路径推导是对的，改成不带 `--resume` 重跑 |
| `[ERROR] Training is already in progress` | 单实例限制；等上一轮结束或先 `stop_training()` |
| 训练没开始就 `training_error` | 看 `traceback` 字段；常见是 `data` 路径错、`task` 与权重不匹配 |
| ROI 开了但看着像全图 | `roi` 四项全 0 就视为未启用（anno_studio 默认勾选状态），直接全图训练；只有非全 0 的非法 ROI 才报错 |
| 系统 TEMP 残留 `train-json-*.json` | 上一轮异常退出没走到 `os.remove`，可直接删 |
| 收不到 `training_started` | 它由 `on_train_start` 发出；初始化阶段就失败的话只会收到 `training_error` |
| 打包版一训练就 `Training process exited with code 2` | 用的是修复前的包——旧版会让 exe 自己重入并撞上 argparse。重新 `pyinstaller` 即可（见 2.5） |
| 最后一条 `training_metrics` 的 `epoch` 比 `total_epochs` 大 1 | 正常，那是 `final_eval` 记录的 best 指标（见 4.2） |
| `training_metrics` 的 `cls_loss` 是 `null` | 非 `detect`/`obb` 任务故意给 `null`，避免按下标取错分项（见 4.2） |
| 事件没实时打印、攒了一堆才出来 | 子进程 stdout 缓冲。源码模式靠 `-u`，打包模式靠 `reconfigure(line_buffering=True)`，两者都已内置 |
| 输出目录变成 `<alias>2` / `<alias>3` | `exist_ok` 默认 false，同名目录已存在就自增。`service_state.json` 仍写在 `<project>/<name>/`，续训读 state 里的 `save_dir`，不受影响；要固定目录就在 config 里加 `"exist_ok": true` |
| `expected input[16, 3, 3, 3] ... got 1 channels` | 灰度 tif。`install_gray_shim()` 已覆盖；仍报这条说明跑的是修复前的包，重新 `pyinstaller` |
| `TypeError: slice indices must be integers` | ROI 传了浮点。现版本已 `int(round())`；仍报则是旧包 |
| `Grayscale shim failed: ultralytics.utils.patches.imread not found` | ultralytics 升级后模块布局变了，同步改 `install_gray_shim()` |
| `Missing required training arguments: ['data']` 但配置里有 `dataYaml` | `dataYaml` 是空串 `""`。别名会归一化成 `data`，但必填是**真值检查**，空串同样拒绝 |
| 旧 ZMQ 配置跑通了，但输出跑到 `runs/`、状态文件也在那 | 缺 `project`。旧服务固定写 `<config 目录>/output/<alias>/`，新服务没这逻辑，需显式给 `project` |
