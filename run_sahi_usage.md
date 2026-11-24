# SAHI 脚本使用速查

解释器统一使用本仓库环境：

```powershell
D:\.conda\envs\ultraylytics\python.exe
```

## 一、切片推理 `run_yolo26_predict_sahi.py`

### 常用命令

```powershell
# 单图
D:\.conda\envs\ultraylytics\python.exe run_yolo26_predict_sahi.py --model yolo26n.pt --source ultralytics/assets/bus.jpg

# 图片文件夹
D:\.conda\envs\ultraylytics\python.exe run_yolo26_predict_sahi.py --model yolo26n.pt --source path/to/images

# 视频
D:\.conda\envs\ultraylytics\python.exe run_yolo26_predict_sahi.py --model yolo26n.pt --source path/to/video.mp4 --save-txt

# 摄像头
D:\.conda\envs\ultraylytics\python.exe run_yolo26_predict_sahi.py --model yolo26n.pt --source 0 --show

# 网络流 / 图片 URL
D:\.conda\envs\ultraylytics\python.exe run_yolo26_predict_sahi.py --model yolo26n.pt --source rtsp://user:pass@ip:554/stream

# 只保留指定类别 + 不保存文件
D:\.conda\envs\ultraylytics\python.exe run_yolo26_predict_sahi.py --model yolo26n.pt --source path/to/images --classes 0 2 3 --no-save

# 自训练权重
D:\.conda\envs\ultraylytics\python.exe run_yolo26_predict_sahi.py --model runs/detect/train/weights/best.pt --source path/to/images
```

### 参数速查

| 参数                                           | 默认                           | 说明                                       |
| ---------------------------------------------- | ------------------------------ | ------------------------------------------ |
| `--model`                                      | `yolo26n.pt`                   | 权重路径，本地不存在时按官方发布名自动下载 |
| `--source`                                     | `ultralytics/assets/bus.jpg`   | 单图 / 文件夹 / 视频 / URL / `0` 摄像头    |
| `--imgsz`                                      | 640                            | 推理尺寸，透传给 SAHI 的 `image_size`      |
| `--conf`                                       | 0.25                           | 置信度阈值（模型推理 + 合并后过滤）        |
| `--device`                                     | CUDA 可用为 `0`，否则 `cpu`    | 设备                                       |
| `--slice-height/--slice-width`                 | 512                            | 切片尺寸                                   |
| `--overlap-height-ratio/--overlap-width-ratio` | 0.2                            | 切片重叠比例                               |
| `--postprocess-type`                           | `GREEDYNMM`                    | `GREEDYNMM` / `NMM` / `NMS` / `LSNMS`      |
| `--postprocess-match-metric`                   | `IOS`                          | `IOS` / `IOU`                              |
| `--postprocess-match-threshold`                | 0.5                            | 合并匹配阈值                               |
| `--no-standard-pred`                           | 关                             | 关闭整图推理，只用切片结果                 |
| `--batch-size`                                 | 1                              | 切片推理批大小，调大可提升吞吐             |
| `--classes`                                    | 无                             | 类别过滤，如 `0` 或 `0 2 3`                |
| `--show`                                       | 关                             | 视频/摄像头实时窗口，`q` 退出              |
| `--save` / `--no-save`                         | 开                             | 是否保存结果                               |
| `--save-txt`                                   | 关                             | 保存检测结果 txt                           |
| `--project/--name/--exist-ok`                  | `runs/detect` / `predict_sahi` | 输出目录                                   |
| `--line-width`                                 | 自动                           | 框线宽                                     |
| `--hide-conf`                                  | 关                             | 隐藏置信度文字                             |

### 调参建议

- 小目标多、分辨率高：`--slice-width 512 --slice-height 512`（默认）或降到 384；显存/速度吃紧时可提到 640
- 重叠：目标容易跨切片边界时提高 `--overlap-* -ratio`（如 0.25~0.3），代价是切片数量增加
- 大目标漏检：保留默认的整图推理；只关心小目标时加 `--no-standard-pred` 提速
- 重复框多：`--postprocess-type NMS` 或提高 `--postprocess-match-threshold`
- 速度优先：`--batch-size 8`、`--imgsz 512`、`--no-standard-pred`

## 二、切片训练 `run_yolo26_train_sahi.py`

### 常用命令

```powershell
# 首次：切片 + 训练
D:\.conda\envs\ultraylytics\python.exe run_yolo26_train_sahi.py --model yolo26n.pt --data path/to/data.yaml --epochs 100 --batch 16 --imgsz 640 --output runs/sahi_data --name sahi_train

# 复用已切好的数据，只调训练参数
D:\.conda\envs\ultraylytics\python.exe run_yolo26_train_sahi.py --model yolo26n.pt --skip-slice --output runs/sahi_data --epochs 200 --batch 8 --name sahi_train2

# 小目标：更小的切片 + 更大重叠
D:\.conda\envs\ultraylytics\python.exe run_yolo26_train_sahi.py --model yolo26n.pt --data path/to/data.yaml --slice-height 384 --slice-width 384 --overlap-height-ratio 0.3 --overlap-width-ratio 0.3 --output runs/sahi_data_384 --epochs 100

# 用切片数据集验证流程（coco8 自动下载）
D:\.conda\envs\ultraylytics\python.exe run_yolo26_train_sahi.py --model yolo26n.pt --data coco8.yaml --epochs 1 --batch 4 --imgsz 320 --output runs/sahi_coco8_test
```

### 参数速查

**切片参数**

| 参数                                           | 默认             | 说明                                            |
| ---------------------------------------------- | ---------------- | ----------------------------------------------- |
| `--data`                                       | `coco8.yaml`     | YOLO 数据集配置（必须有 train/val）             |
| `--output`                                     | `runs/sahi_data` | 切片数据集输出目录，须为空或加 `--skip-slice`   |
| `--slice-height/--slice-width`                 | 512              | 切片尺寸                                        |
| `--overlap-height-ratio/--overlap-width-ratio` | 0.2              | 切片重叠比例                                    |
| `--min-area-ratio`                             | 0.1              | 切片内保留注释的最小面积占比，低于该值丢弃      |
| `--skip-slice`                                 | 关               | 复用 `--output` 中已切好的数据                  |
| `--symlink`                                    | 关               | 用软链接代替复制（Windows 需开发者模式/管理员） |

**训练参数**（与 `run_yolo26_train.py` 一致）

`--epochs` 100、`--batch` 16、`--imgsz` 640、`--workers` 8（CPU 自动降到 4）、`--optimizer auto`、`--lr0 0.01`、`--patience 100`、`--amp`（CUDA 可用时默认开）、`--cache False`、`--plots`、`--save-period -1`、`--cfg`、`--seed 0`、`--cos-lr`、`--close-mosaic 10`、`--resume`，以及 CURB 系列 `--curb/--curb-interval/--curb-kappa/--curb-ema/--curb-stall`。

### 输出解读

训练完成后：

- 切片数据集：`runs/sahi_data/data.yml`，可直接传给 `model.train(data=...)` 或 `yolo train`
- 训练结果：`runs/detect/<name>/`（权重、曲线、混淆矩阵）
- 控制台会打印每个 split 的图片数、注释数、跳过数，以及切片导出目录

## 三、故障排查

| 现象                                              | 原因与处理                                                              |
| ------------------------------------------------- | ----------------------------------------------------------------------- |
| `FileExistsError: ... is not empty`               | `--output` 已有内容；用 `--skip-slice` 复用，或换新目录（脚本不删文件） |
| `OSError: [WinError 1314]`                        | 软链接权限不足；去掉 `--symlink`（默认复制）                            |
| `ValueError: Only xywh bbox labels are supported` | 数据集是分割/obb 标签；本脚本仅支持 5 列 bbox                           |
| `Class id ... out of range`                       | 标签类别 id 超出 `data.yaml` 的 `names` 范围                            |
| `No images found for train split`                 | `data.yaml` 的 train/val 路径不正确（相对路径按 ultralytics 规则解析）  |
| 自动下载数据集失败                                | 检查 `yolo settings` 中 `datasets_dir` 是否为可写目录                   |
| 视频输出打不开                                    | 检查 `opencv-python` 是否含 `mp4v` 编码支持                             |
