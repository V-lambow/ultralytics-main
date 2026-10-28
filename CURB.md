# CURB：固定预算下的课程重采样（Curriculum Resampling Under Budget）

CURB 在**训练计划完全不变**的前提下（200 epochs、每 epoch 抽样次数恒等于 `len(dataset)`、学习率调度不受影响），周期性地对训练集做一次"模型自评"，按每张图的检测难度重排抽样权重：难图多抽、易图少抽，并用类配额、停滞止损防止权重塌缩到个别类/个别图。

- 默认关闭，`curb: False`；开启后仅作用于 **train 集**，val 集不参与任何采样决策。
- 当前仅支持 `task=detect`、**单进程训练**（`RANK=-1`）。

---

## 1. 设计不变量（协议）

| 不变量 | 实现方式 |
| --- | --- |
| 总 epoch 数不变 | 不触碰循环，只换 sampler 权重 |
| 每 epoch 抽样次数 = `len(dataset)` | `CurbSampler(WeightedRandomSampler)` 构造时固定 `num_samples=len(dataset)`，刷新只换权重张量、永不改预算 → LR schedule 完全不受影响 |
| val 不参与决策 | 难度刷新只跑 train 路径的图（以 val 模式构建，保证无增广） |
| 可复现 / 可回退 | `curb: False` 即完全回到原生 shuffle 行为（回归冒烟已验证） |

## 2. 机制总览（五件套）

每个 `curb_interval` 个 epoch，在 epoch 末（`on_fit_epoch_end`，validate/存档之后）执行一次刷新：

```
[A] 难度刷新      对全训练集做一次无增广推理（EMA 权重），得到每图难度 d_new
[B] 指数平滑      d̄ = (1 - curb_ema)·d̄ + curb_ema·d_new        （首次直接 d̄ = d_new）
[C] 图级权重      w = clip((d̄ / median(d̄))^0.5, 0.3, 6.0)      （median ≤ 1e-6 时全 1）
[D] 停滞止损      连续 curb_stall 次刷新类均值无实质改善 → 该类"停滞"；
                  标签全部停滞的图去掉加权抬升：w = min(w, 1)
[E] 类配额        每 epoch 该类图的期望抽中量封顶在 curb_kappa × 类图数
       顺序：[C] → [D] → [E]（配额最后执行，保证上限恒成立）
[F] 原子切换      sampler 权重就地 copy_ + train_loader.reset()
                  （丢弃 InfiniteDataLoader 按旧权重预取的下一批索引）
```

**时机**：1-based epoch 满足 `epoch % curb_interval == 0` 且训练未结束（`trainer.stop` 为假）才刷新——默认第 25、50、175 epoch 末各刷一次，最后一次（200）不刷（没有下一个 epoch 消费新权重）。

### 难度公式

类感知贪心匹配（按 IoU 降序、GT/预测各只能用一次、类别必须相同、IoU ≥ 0.5）：

```
d = clip( (FN + 0.5·FP + Σ_TP (1 - IoU)) / max(1, n_gt + n_pred), 0, 1 )
```

- 漏检图 → 1.0；只有虚检的图 → 上限 0.5；全对且 IoU=1 → 0；完全空图（无 GT 无预测）→ 0。
- 错类重叠会同时计入 FN 和 FP（类别不匹配 → 不算 TP）。
- 与 `DetectionValidator` 同源：同一套 EMA 权重、val 模式 letterbox（`rect=True`）、同一 NMS（conf 取 `0.25` 或显式 `conf=`、`iou=0.7`、`multi_label`、`max_det`、`end2end` 同步）、同一 GT 转换（`xywh2xyxy × [w,h,w,h]`），按 `im_file` 对回训练集。

### 类配额

- `e_c = N · Σ_{i∈c} p_i`：当前权重下每 epoch 类 c 所属图的期望抽中量（单类图即其抽中概率 × N）。
- `g_c = min(1, κ·n_c / e_c)`，多类图取其所有类的 `min(g_c)`，≤5 轮定点迭代至无变化。
- 注意 `e_c ≤ N` 恒成立，而 `κ·n_c` 只对**稀有类**才可能小于 `N`——配额实际只会约束长尾类，主类（如 SK 81%）永远不会被封顶。

### 停滞止损

- 类均值改善条件：`prev_mu - cur_mu ≥ max(0.005, 0.01·|prev_mu|)` → `stall_count = 0`，否则 `+1`。
- `stall_count ≥ curb_stall` → 该类停滞；停滞只作用于**加权抬升**（`min(w,1)`，允许继续被压低到 1 以下）。
- 训练集中**没有任何样本的类**不累计停滞计数（恒 0，永不判停滞）。

## 3. 配置参数

`ultralytics/cfg/default.yaml`（Train settings 段新增）：

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `curb` | `False` | 总开关（bool） |
| `curb_interval` | `25` | 刷新间隔（epoch，≥1，`CFG_INT_MIN` 拦截 0） |
| `curb_kappa` | `2.0` | 类暴露量封顶倍数（>0，trainer 里 assert） |
| `curb_ema` | `0.3` | 新观测的 EMA 权重（∈[0,1]，`CFG_FRACTION_KEYS` 校验） |
| `curb_stall` | `3` | 连续多少次无改善判停滞（≥1） |

固定常量（不进配置，`ultralytics/data/curb.py` 顶部）：

| 常量 | 值 | 含义 |
| --- | --- | --- |
| `CLIP` | `(0.3, 6.0)` | 图级权重裁剪区间 |
| `POW` | `0.5` | 难度比指数（温和课程） |
| `MATCH_IOU` | `0.5` | 匹配 IoU 阈值 |
| `FP_PENALTY` | `0.5` | 虚检的罚分权重（=1 个漏检的一半） |
| `STALL_ABS` / `STALL_REL` | `0.005` / `0.01` | 停滞判定的绝对/相对改善门槛 |
| `QUOTA_PASSES` | `5` | 配额定点迭代上限 |
| `DIFFICULTY_EPS` | `1e-6` | 中位难度低于此值视为全图同难度 → 权重全 1 |

## 4. 逐文件修改点

| 文件 | 改动 |
| --- | --- |
| `ultralytics/data/curb.py`（新建，~300 行） | `CurbSampler`（固定预算加权采样器，`update()` 原地换权重）；`_match`/`_image_difficulty`（贪心匹配+难度打分）；`_evaluate`（`@smart_inference_mode` 全训练集难度 pass，镜像 validator）；`_refresh`（EMA→权重→止损→配额→切换→日志）；`_log_csv`/`_log_review`（产物写出）；`curb_on_fit_epoch_end`（回调入口，跳过条件 `trainer.stop or epoch % interval`） |
| `ultralytics/cfg/default.yaml` | 新增 `# Curriculum resampling (CURB)` 段 5 个键（`compile:` 之后） |
| `ultralytics/cfg/__init__.py` | 类型注册：`curb`→`CFG_BOOL_KEYS`；`curb_kappa`→`CFG_FLOAT_KEYS`；`curb_ema`→`CFG_FRACTION_KEYS`；`curb_interval`/`curb_stall`→`CFG_INT_KEYS` + `CFG_INT_MIN=1`（防除零） |
| `ultralytics/data/build.py` | `build_dataloader` 新增 `sampler=None` 参数：传入时优先于 shuffle/分布式 sampler（`samples = len(sampler)`，预算即采样器长度）；rank 分支改为 `if sampler is None and rank != -1` |
| `ultralytics/models/yolo/detect/train.py` | ① `__init__`：`self._curb=None`；`curb` 开启且 `task=detect` 时初始化状态 `{d, mu, stall, sampler}`、注册 `on_fit_epoch_end` 回调、非 resume 时截断 `curb.csv`/`curb_review.txt`（镜像 `results.csv` 语义）；其他任务打 warning 后保持关闭；`curb_kappa > 0` assert。② `get_dataloader`：train 分支创建/复用 `CurbSampler`（OOM 重建 dataloader 时沿用同一实例与权重），`assert rank == -1`（DDP 不支持，fail-fast），并把 `sampler` 传给 `build_dataloader` |

改动总量：4 个文件 +43/-7 行，新增 1 个文件；`ruff format` / `ruff check` 对新增文件全绿（仓库存量 8 个 lint 告警为 ruff 0.16 默认规则集扩大所致，均在本次未触碰的行上）。

## 5. 运行产物

每次刷新在 run 目录写：

- **日志行**：`CURB: refreshed at epoch 50 | median difficulty 0.412 | weight range [0.30, 6.00] | stalled classes: LX, FG`
- **`curb.csv`**（每类每刷新一行，新文件自动带表头）：
  `epoch,class,mean_diff,exposure,cap,stall_count,stalled`
  `exposure` = 当前期望每 epoch 抽中量，`cap = curb_kappa × 类图数`。
- **`curb_review.txt`**（仅有停滞类时追加）：停滞类 + 该类难度最高的 5 张图路径（难度值），供人工核查标注质量或换难例。

## 6. 使用

```bash
# CLI
yolo detect train model=yolo26n.pt data=defects.yaml epochs=200 curb=True curb_interval=25

# 训练脚本（本仓库 run_yolo26_train.py 已接入 5 个参数）
python run_yolo26_train.py --model yolo26n.pt --data defects.yaml --epochs 200 \
  --curb --curb-interval 25 --curb-kappa 2.0 --curb-ema 0.3 --curb-stall 3
```

```python
# Python
from ultralytics import YOLO
YOLO("yolo26n.pt").train(
    data="defects.yaml", epochs=200,
    curb=True, curb_interval=25, curb_kappa=2.0, curb_ema=0.3, curb_stall=3,
)
```

开销：每次刷新 = 一遍全训练集推理（无增广、EMA 权重）。30k 图 / yolo26n / 640px 约 1–2 分钟，`curb_interval=25` 时约占总训练时间 <1%。

## 7. 验证（coco8 冒烟，YOLO26n，RTX 4060）

| 阶段 | 内容 | 结果 |
| --- | --- | --- |
| 回归 | `curb=False` 训 1 epoch | 行为与原生一致，`trainer._curb is None`，`results.csv` 正常 ✅ |
| 主流程 | `curb=True curb_interval=1 curb_stall=1` 训 4 epoch | 3 次刷新（epoch 1/2/3，末轮按设计跳过）；预算不变量 `num_samples == len(dataset)` ✅；`curb.csv` 240 行 = 3×80 类 ✅；停滞计数只落在实际存在的 7 个类上、其余 73 类恒 0 ✅；`_log_review` 产出头 + 最难图 ✅ |
| 续训 | 训练中途抓取未 strip 的 `mid.pt`（epoch 1，optimizer 完整）→ 改 `epochs=6` resume | 状态从 ckpt args 正确重挂 `curb=True` ✅；刷新记录追加 [3,4,5]（末轮跳过），`curb.csv` 241→481 行 ✅；`curb_review.txt` 未被截断（哨兵保留）✅ |

注：随机初始化初期模型检不出任何框，难度全为 1 → 中位数法自然退化为均匀权重（`weight range [1.00, 1.00]`），这是设计内行为，冒烟断言对此分支做了区分处理。

## 8. 已知限制

1. **单进程限定**：DDP 下 `get_dataloader` 直接 assert 失败（fail-fast）。多卡需引入全局归约（各 rank 汇总难度/直方图）后才能支持。
2. **仅 detect 任务**：pose/obb/segment/World 等继承 trainer 会打 warning 并保持关闭。扩展到 segment 只需把 `_evaluate` 的 NMS `nc=0 if detect else nc` 一处放开（镜像 `DetectionValidator.postprocess`），暂未实现。
3. **CURB 状态不入 checkpoint**：resume 后 `d̄/mu/stall` 清零，首个刷新周期前按均匀权重运行、停滞计数从头累计（冒烟日志可见 resume 后首刷 `stalled classes: none`）。`curb.csv`/`curb_review.txt` 本身会跨 resume 追加。
4. **已完成的 run 无法 resume**：fork 对 finalize 后的 `last.pt` 做 strip（`epoch=-1`），`resume_training` 按上游逻辑拒绝续训——这是仓库既有行为，与 CURB 无关。
5. **首刷前无课程**：第 1 个 `curb_interval` 内（默认前 25 epoch）为均匀抽样；难度 pass 成本也集中在这些时点。
6. **小数据/极早期退化**：全图同难度（含模型尚检不出框时）→ 中位数法给出均匀权重，属数学上的正确退化，不是故障。
7. **难度 pass 的 conf 口径**：训练内刷新用 `0.25`（或显式 `conf=`），而 validator 的 `args.conf` 默认是 `0.001`——刷新更看重"有意义的检出"，与 val 的 mAP 口径刻意不同。
8. **类名含逗号**会使 `curb.csv` 列错位（未做 CSV 转义，常见缺陷命名不受影响）。

## 9. 后续扩展路线（未实现）

- DDP：`all_gather` 聚合每图难度与类直方图后再走同一套 `_refresh`。
- 状态入 ckpt：把 `d/mu/stall` 存进 `save_model` 的额外字段，`resume_training` 恢复。
- segment/pose/obb：放开 `_evaluate` 的 `nc` 与 GT 结构（boxes→masks/kpts），配额与止损逻辑完全复用。
