# train-JSONservice

仿照 X-AnyLabeling 的「子进程 + payload JSON + stdout 事件流」方式实现的 YOLO 训练服务。

核心思路：**训练跑在独立子进程里，父进程只读它的 stdout**。参数用临时 JSON 文件传入（避免命令行转义地狱），进度与状态用带哨兵前缀的 NDJSON 事件回传，磁盘上的 `service_state.json` 是事实来源。

本目录只交付 **worker + 管理器 + CLI**，可独立运行，**尚未接入 ZMQ**（接口层后续再加薄壳）。动作只有 `train` / `stop`，**不做 pause/holdon**。

---

## 目录

- [架构](#架构)
- [文件结构](#文件结构)
- [快速开始](#快速开始)
- [配置字段](#配置字段)
- [事件协议](#事件协议)
- [状态落盘](#状态落盘)
- [ROI 裁剪训练](#roi-裁剪训练)
- [与 train_ZMQservice 的差异](#与-train_zmqservice-的差异)
- [已知限制](#已知限制)
- [验证记录](#验证记录)

---

## 架构

```
main.py (CLI)
   │  读 config.json，组装 train_args
   ▼
manager.TrainingManager          ← 父进程
   │  ① create_payload() 写临时 payload.json
   │  ② Popen 启动 worker.py，stdout+stderr 合并进管道
   │  ③ 逐行读管道，parse_line() 分流事件 / 日志
   │  ④ 终态兜底 + 杀进程树 + 状态落盘
   ▼
worker.py                        ← 子进程
   │  load_payload() 七层校验
   │  install_gray_shim() 单通道图补 3 通道（模块级，DataLoader 子进程继承）
   │  resolve_device() 张量预检（父进程不依赖 torch）
   │  build_roi_trainer() 按需构造 ROI trainer
   │  YOLO(model).train(**args)
   │  on_train_start  回调 → training_started（带 save_dir）
   │  on_fit_epoch_end回调 → training_metrics（loss / P / R / mAP）
   ▼
ultralytics 训练
```

**四条可靠性约定**

| 约定 | 说明 |
|---|---|
| 恰好一个终态事件 | `manager._dispatch()` 对 `TERMINAL_EVENTS` 做幂等守卫，`training_completed` / `training_error` / `training_stopped` 只会发出一个 |
| 杀整个进程树 | `taskkill /T`（失败则 `/F /T`），POSIX 走 `killpg`，且强杀后必 `wait()` 回收，不留孤儿 DataLoader |
| 参数白名单 | `TRAIN_ARGS_ALLOWED` 之外的 key 直接拒绝，不透传给 ultralytics |
| 磁盘为事实来源 | 进度读 `results.csv`，状态读 `service_state.json`，不依赖推送 |

---

## 文件结构

```
train-JSONservice/
├── protocol.py              payload 构造/七层校验、事件 emit 与解析、日志流切行
├── worker.py                子进程：校验 → 灰度兼容 → 解析 device → 构造 ROI trainer → YOLO().train()
├── roi.py                   ROIDataset / ROIValidatorMixin / ROITrainerMixin / build_roi_trainer
├── manager.py               父进程：拉子进程、读管道、终态兜底、杀进程树、状态落盘
├── main.py                  CLI：跑训练、看状态、续训；打包后兼作 worker 入口
├── train_json_service.spec  PyInstaller 描述文件（onedir）
├── config.json.example      配置样例
├── README.md                架构、协议、与旧服务的差异、已知限制
└── usage.md                 怎么跑 / 怎么 build / 怎么调用 / 怎么通讯
```

---

## 快速开始

### 环境

Python ≥ 3.8，已安装 `ultralytics`、`torch`、`opencv-python`、`numpy`。

> **本机注意**：`yolo26n.pt` 这类新权重的结构只有仓库内那份 ultralytics 能解析。
> 如果 `site-packages` 里装的是旧版本，需要把仓库根加进 `PYTHONPATH`：
>
> ```powershell
> $env:PYTHONPATH = "E:\m_code\ultralytics-main"
> ```

### 跑一次训练

```powershell
cd E:\m_code\ultralytics-main\train-JSONservice
python main.py --config config.json
```

父进程活到训练结束，**按 Ctrl+C = 停止**（杀进程树，退出码 130）。

### 看上次状态

```powershell
python main.py --config config.json --status
```

### 续训

```powershell
# 不带路径：从上次的 service_state.json 取 save_dir，推导 weights/last.pt
python main.py --config config.json --resume

# 显式指定 checkpoint
python main.py --config config.json --resume E:/runs/exp/weights/last.pt
```

`--resume` 不带路径时**不会**去 CWD 找 `last.pt`（规避了原 ZMQ 服务 `resume=True` 的坑）。

### 退出码

| 码 | 含义 |
|---|---|
| 0 | `training_completed` |
| 1 | `training_error`，或配置/协议错误 |
| 130 | `training_stopped`（Ctrl+C） |

---

## 配置字段

配置是单层 JSON，`train` 对象里放训练参数：

```json
{
  "train": {
    "model": "E:/m_code/ultralytics-main/yolo26n.pt",
    "data":  "D:/dataset/xxx/data.yaml",
    "epochs": 100,
    "batch": 16,
    "imgsz": 640,
    "device": "auto",
    "task": "detect",
    "project": "E:/runs",
    "name": "exp1",
    "exist_ok": true,
    "workers": 4,

    "useRoi": false,
    "roi": { "x": 640, "y": 320, "width": 1280, "height": 720 }
  }
}
```

### 训练参数（白名单）

`model`、`data` **必填**，其余可选：

```
model data project name exist_ok task device epochs batch imgsz workers
time patience resume cache fraction classes single_cls rect pretrained
seed deterministic verbose plots val save save_period optimizer cos_lr amp
multi_scale close_mosaic lr0 lrf momentum weight_decay warmup_epochs
warmup_momentum warmup_bias_lr hsv_h hsv_s hsv_v degrees translate scale
shear perspective dropout box cls dfl pose kobj
```

白名单之外的 key 会被 `load_payload()` 拒绝并报 `Unsupported training arguments`。

### 服务专属字段

| 字段 | 说明 |
|---|---|
| `useRoi` | `true` 时才 `import roi`，默认路径完全黑盒 |
| `roi` | `{x, y, width, height}`，图像坐标系，宽高必须 > 0 |

`useRoi: true` 但 `roi` 缺省或四项全 0 时视为**未启用 ROI**（anno_studio 的复选框默认勾选、ROI 未绘制时就是这种状态），直接走全图训练。
| `device` | `"auto"` → 有 CUDA 取 `0`，否则 `cpu`；也支持 `"0"`、`"0,1"`、`"cpu"`、`"mps"` |

`useRoi` / `roi` 会在进入 `model.train()` 之前被 pop 掉，不会透传给 ultralytics。

### 旧字段别名（兼容 train_ZMQservice）

旧 ZMQ 服务的配置字段名在解析时会自动归一化，`anno_studio` 等既有配置**无需改动**：

| 旧字段 | 归一化为 | 说明 |
|---|---|---|
| `modelPath` | `model` | 权重或模型 YAML 路径 |
| `dataYaml` | `data` | 数据集 YAML 路径 |
| `alias` | `name` | 输出子目录名 |

规则：

- **新字段名优先**：`model` 与 `modelPath` 同时存在时取 `model`，其余同理
- 别名只做重命名，仍走同一套白名单；`modelPath` 不会绕过校验
- `port` 仍是**静默丢弃**——新服务暂无 ZMQ，接薄壳时再启用（见 [usage.md 4.4](usage.md#44-后续接-zmq-的薄壳)）
- 旧服务固定写到 `<config 所在目录>/output/<alias>/`；新服务**没有这个逻辑**，
  需要显式给 `project`，否则落到 ultralytics 默认的 `runs/`

`model` / `data` 是**真值检查**：缺字段或空串（如旧配置里 `"dataYaml": ""`）都会直接报
`Missing required training arguments: [...]`，不会带着空路径进 ultralytics 再吐一句
`Dataset '.' error`。

---

## 事件协议

子进程往 stdout 写事件，每行一条，**带哨兵前缀**，其余输出按普通日志处理：

```
__TRAIN_SERVICE_EVENT__={"version": 1, "event": "training_started", ...}
```

`PROTOCOL_VERSION = 1`，只有这一套协议（不复刻 X-AnyLabeling 的裸 payload / 带版本 payload 双协议历史包袱）。

### 事件表

| 事件 | 触发时机 | 关键字段 |
|---|---|---|
| `training_started` | ultralytics `on_train_start` 回调 | `total_epochs`、`save_dir` |
| `training_metrics` | 每个 epoch 结束（`on_fit_epoch_end`） | `epoch`、`loss`、`box_loss`、`precision`、`mAP50`、`mAP50_95`、`task` 等 |
| `training_log` | 子进程任意 stdout/stderr 文本行 | `message` |
| `training_completed` | 训练正常结束 | `results` |
| `training_error` | 初始化或训练抛异常、非零退出 | `error`、`exception_type`、`traceback` |
| `training_stopped` | 父进程收到 stop / Ctrl+C | — |
| `protocol_error` | 事件版本不符或缺 `event` 字段 | `error` |

`training_started` 改由 `on_train_start` 回调发出，因此**初始化失败不会误报 started**（数据集错误、权重不匹配等都会直接走 `training_error`）。

`training_metrics` 的字段与旧 ZMQ 的 `metrics` topic 一一对齐，详见 [usage.md - 4.2](usage.md#42-通道-bstdout-事件流出)。

### 管道切行规则

- `EventLogStream` 同时按 `\n` 和 `\r` 切行；
- 父进程打印进度条时用 `\r` 原地刷新；
- 打印普通日志前先写一个 `\r` 清掉进度条残留。

`stderr` 用 `stderr=STDOUT` 合并进同一管道，事件与日志同管道回传。

---

## 状态落盘

每个有意义的事件都会更新 `<project>/<name>/service_state.json`
（`training_log` 与 `training_metrics` **不落盘**——否则进度条每刷新一行就写一次盘，
且状态文件的 `event` 字段会停在无意义的 `training_log` 上）：

```json
{
  "event": "training_completed",
  "updated_at": "2026-11-10T11:47:53",
  "save_dir": "E:/runs/exp1",
  "results": "Training completed successfully"
}
```

- `save_dir` 取自 `training_started`；
- **不写 traceback**（避免状态文件被几十 KB 的堆栈污染），traceback 只走事件流；
- 文件损坏或不存在时 `read_state()` 返回 `None`，不抛异常。

---

## ROI 裁剪训练

代码与 `train_ZMQservice/train_service.py` 中的 ROI 部分一致，原样搬入 `roi.py`：

| 组件 | 作用 |
|---|---|
| `ROIDataset` | 读取层裁剪图像与标注（bbox / segment / keypoints 同步裁剪与重归一化） |
| `ROIValidatorMixin` | 验证批次与预测回映射到原图坐标系，mask 做仿射变换 |
| `ROITrainerMixin` | train/val 两个 split 都建 `ROIDataset`，并把 validator 换成带 ROI 的子类 |
| `build_roi_trainer(task, roi)` | 按 task 动态构造 `ROI{BaseTrainer}` 类 |

通过 `model.train(trainer=...)` 注入（`ultralytics/engine/model.py:726,807`），**默认路径不 import 任何 ultralytics 内部类**。

**支持的 task**：`detect` / `segment` / `pose` / `obb`。`classify` 会直接报错。

`roi` 四项在进入 `roi.py` 前统一 `int(round(float(x)))` 取整——anno_studio 的 `QRectF` 传来的是浮点，
`roi.py` 内部按像素切片，浮点下标会直接 `TypeError`。

**非法 ROI 直接报错，不再静默回退全图**（修复原已知问题 9）：

```
[ERROR] ROI is missing fields: ['x', 'y', 'width', 'height']
[ERROR] ROI width/height must be > 0, got 0x100
[ERROR] ROI (500, 500, 100, 100) does not intersect image .../crazing_1.jpg of shape (200, 200)
[ERROR] ROI training is not supported for task 'classify', expected one of ['detect', 'obb', 'pose', 'segment']
```

---

## 单通道图像兼容

上游 `ultralytics.utils.patches.imread` 的 `.tif/.tiff` 分支硬编码 `IMREAD_UNCHANGED`、无视传入的 `flags`，
灰度 tif 会被 `np.stack([2D], axis=2)` 还原成 `(H, W, 1)`，绕过了函数末尾那句
`im[..., None] if im.ndim == 2`（那时已经是 3 维了）。结果训练、验证、推理全在第一个 conv 层崩：

```
RuntimeError: Given groups=1, weight of size [16, 3, 3, 3],
expected input[16, 1, 640, 640] to have 3 channels, but got 1 channels instead
```

`worker.install_gray_shim()` 在读取层把 `(H, W, 1)` 补成 `(H, W, 3)`：

| 项 | 行为 |
|---|---|
| 生效范围 | `ultralytics.data.base`（训练/验证数据集）、`ultralytics.data.loaders`（推理）、`roi.py`（惰性 import，自动拿到补丁） |
| 只改通道 | 其余 shape / dtype 原样返回；`flags=IMREAD_GRAYSCALE` 的调用不动，避免改掉调用方明确要灰度的路径 |
| 安装时机 | **模块级**（`install_gray_shim()` 在 `if __name__` 之前）。Windows DataLoader 走 `spawn`，子进程重跑主模块代码；装在 `run_train()` 里 worker 拿不到补丁，表现为 8 个 dataloader worker 全崩 |
| 幂等 | 补丁函数带 `is_gray_shim` 标记，重复调用直接返回，不套娃 |
| 找不到就报错 | `patches.imread` 不存在 → `RuntimeError: Grayscale shim failed: ...`，不静默降级 |

**不管位深**：只补通道，不做 uint16 → uint8 的拉伸。12/16 bit 工业 tif 会原样得到 uint16 张量，属另一个需求。

**`cache='ram'` 内存 ×3**：默认 `cache=False` 不受影响；若开启且此前留过单通道 `*.npy`，需清一次缓存。

上游修好后这个 shim 可以直接删。

---

## 与 train_ZMQservice 的差异

| 维度 | train_ZMQservice | train-JSONservice |
|---|---|---|
| 执行模型 | 服务进程内开线程跑训练 | 训练在独立子进程 |
| 传参 | ZMQ 消息体 | 临时 payload JSON 文件 |
| 进度回传 | ZMQ PUB 推送 | stdout 事件流（父进程可再包一层接口） |
| 动作 | train / pause / holdon / stop / status | train / stop / status / resume |
| 续训 | `resume=True` → 去 CWD 找 checkpoint | `--resume` 从 state 的 `save_dir` 推导绝对路径 |
| 停止 | 置 `trainer.stop`，存在双线程抢 GPU 竞态 | 杀整个进程树，无竞态 |
| ROI 非法 | 静默回退全图 | 四项全 0 视为未启用；其余非法直接报错 |
| 灰度 / 单通道图 | 同样崩在 conv 层 | `install_gray_shim()` 读取层补 3 通道 |
| ROI 浮点坐标 | `TypeError: slice indices must be integers` | 进 `roi.py` 前 `int(round())` |
| 异常回传 | 仅日志，客户端拿不到 | `training_error` 带 `error` + `traceback` |
| 状态持久化 | 无 | `service_state.json` |
| 协议 | 单套无版本 | 单套带 `version: 1` |

原 README 记录的 17 条已知问题中，本服务已针对性解决 **#2**（续训取错 checkpoint）、**#4**（pause 时序竞态）、**#9**（ROI 静默回退）、**#11**（异常不回传）、**#14**（状态不持久化）；**#3 / #12** 因取消 pause 与加终态兜底而不再适用。

---

## 已知限制

1. **单实例**：`TrainingManager` 同时只允许一个训练，第二次 `start_training()` 返回 `False`。
2. **无 pause**：长训练只能 stop 后 `--resume` 续训。
3. **CLI 是一次性同步模式**：父进程活到训练结束，不是常驻服务。要做常驻服务，直接用 `TrainingManager` 的 API（`start_training` / `stop_training` / `wait` / callbacks），CLI 只是它的一层外壳。
4. **依赖 ultralytics 半私有 API**：ROI 路径用 `YOLODataset`、四个 Trainer、`unwrap_model`；
   单通道兼容用 `imread`（灰度 shim 对**所有**数据都生效）。升级 ultralytics 时两处都可能失效。
5. **未接 ZMQ**：客户端协议需要后续再包一层。
6. **stop 后不保证 checkpoint 完整**：进程树是被杀的，最后一个完整 checkpoint 取决于 ultralytics 的保存时机。
7. **父进程被强杀时子进程不被带走**（任务管理器结束 / `kill -9`）。只有走 `stop_training()` 或 Ctrl+C 才会杀整棵树。
8. **打包版依赖 `sys.frozen` 分流**：`build_worker_command()` 在 frozen 时让 exe 重入自己当 worker
   （`--worker` 参数），`worker.main()` 显式 `reconfigure(line_buffering=True)` 保证事件实时。
   用修复前的包会看到 `Training process exited with code 2`。

---

## 验证记录

在本机（RTX 4060 Laptop / ultralytics 8.4.102 / conda `ultraylytics`）实测：

| 用例 | 结果 |
|---|---|
| `python -m py_compile` 全部 5 个 py 文件 | 通过 |
| 协议往返单测（`create_payload` → `load_payload` → `emit` → `parse_line`） | 通过 |
| 非法 payload 拒绝 ×7（版本 / 动作 / 多余字段 / 缺 model / 未白名单参数 / paths 类型 / options 类型） | 全部拒绝 |
| `EventLogStream` 按 `\n` 与 `\r` 切行、空行、坏 JSON 回退 | 通过 |
| 端到端：NEU-DET 跑 1 epoch | `[START]` 带 save_dir → `[DONE]`，退出码 0 |
| `service_state.json` 落盘 + `--status` | 通过 |
| `--resume` 不带路径推导 `last.pt` | 路径推导正确 |
| stop：杀进程树 + 恰好一个终态事件 + Popen 回收 + 无孤儿进程 | 通过 |
| stop 后重启可接受 | 通过 |
| ROI 合法配置跑通 | 通过 |
| ROI 越界 / 宽高为 0 / 缺字段 / task 不支持 | 四种均直接报错，退出码 1 |
| `useRoi: true` + 四项全 0 | 视为未启用，全图训练，不报错 |
| ROI 浮点坐标（anno_studio `QRectF`） | 取整后跑通，1 epoch 退出码 0 |
| 灰度 tif（150 train / 38 val）跑 1 epoch，`workers=8` | `[START]` → `Using 8 dataloader workers` → `[DONE]`，退出码 0 |
| 灰度 tif + 真实 ROI 跑 1 epoch | 同上，退出码 0 |
| shim 不改彩色 jpg / 不改 `IMREAD_GRAYSCALE` 调用 | 通过 |
| 初始化失败不误报 `training_started` | 通过 |
| `training_metrics` 事件（源码模式） | 字段与旧 ZMQ `metrics` 对齐 |
| `training_metrics` 事件（打包模式） | 同上，含 `task` 字段 |
| PyInstaller onedir 打包 | exit 0 |
| 打包版 `--help` / `--status` / 训练 / `--resume` | 四条全部通过，训练退出码 0 |
| 打包版 `--worker` 不泄漏到 `--help` | 通过 |
| 状态文件不被 `training_log` / `training_metrics` 覆盖 | 通过 |
