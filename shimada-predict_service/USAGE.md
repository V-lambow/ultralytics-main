# USAGE — ShimadaPredictService 使用手册

面向产线联调与对接测试的完整操作手册。协议依据《软件算法通讯协议》：`POST http://127.0.0.1:9000/image_defect`，JSON + UTF-8。

---

## 1. 环境要求

| 项          | 要求                                                                                             |
| ----------- | ------------------------------------------------------------------------------------------------ |
| Python      | ≥ 3.8（开发验证环境 3.10）                                                                       |
| pillow      | **必装**（读取图片尺寸 / 生成测试图）                                                            |
| ultralytics | 可选，仅 `config.json` 的 `model` 非 `null`（真实推理）时需要（会自动带上 torch、opencv 等依赖） |
| 操作系统    | Windows / Linux 均可（路径处理已做跨平台）                                                       |

## 2. 安装

```bash
# 仅通讯联调（mock 模式）
pip install pillow

# 真实模型推理模式
pip install ultralytics
```

仓库根目录已有完整 ultralytics 环境时，服务端可直接运行，无需额外安装。

## 3. 快速开始（3 步测通）

**终端 1 — 启动服务：**

```bash
cd D:\m_code\dzy\ultralytics-main\shimada-predict_service
python server.py
```

服务端**不接收任何命令行参数**，启动时只读当前工作目录下的 `config.json`，所以必须在 `shimada-predict_service` 目录内执行（或把 `config.json` 放到你的工作目录）。仓库自带的 `config.json` 是 mock 模式，不装模型即可跑通。

启动输出：

```
ShimadaPredictService started
  config     : D:\m_code\dzy\ultralytics-main\shimada-predict_service\config.json
  endpoint   : http://127.0.0.1:9000/image_defect
  mode       : mock
  share root : D:\m_code\dzy\ultralytics-main\shimada-predict_service\shared
Press Ctrl+C to stop.
```

配置有问题（文件缺失、键名拼错、类型/范围不对）会打印具体原因并以退出码 1 结束，**不会打开端口**，客户端这时只会看到 `connection failed`。

**终端 2 — 客户端测试：**

```bash
cd D:\m_code\dzy\ultralytics-main\shimada-predict_service

# ① 自动跑 13 个协议错误码用例（最快验证服务是否正常）
python client.py test-errors

# ② base64 方式发一张自动生成的测试图
python client.py send-base64

# ③ 生成测试图到本地目录
python client.py gen-images --count 2 --out test_images

# ④ 用刚生成的图走 base64 方式
python client.py send-base64 --images test_images\S00001_C01_P01_L0_XXXXXXXXXXXXXX.jpg
```

`send-base64` 成功输出示例：

```
POST http://127.0.0.1:9000/image_defect
job_id      : J20260211103015234
images      : ['S00001_C01_P01_L0_20260211103015.jpg']
HTTP status : 200
elapsed     : 12.4 ms
error_code  : 0
error_msg   : OK
job_id      : J20260211103015234
sample_id   : S00001
pose_id     : P01
resolution  : 1920 x 1440
defects     : 2
  - file=S00001_C01_P01_L0_20260211103015.jpg code=ZM score=0.8123 area=123456 box=[...] ...
```

## 4. 配置与参数

### 4.1 服务端配置 `config.json`

`server.py` **不接收命令行参数**，启动时只读工作目录（`os.getcwd()`）下的 `config.json`，必须是 UTF-8 编码的 JSON 对象。7 个键**全部可选**，缺省值如下：

| 键           | 类型                | 默认值            | 说明                                                         |
| ------------ | ------------------- | ----------------- | ------------------------------------------------------------ |
| `host`       | string              | `"127.0.0.1"`     | 监听地址；对外服务改为 `"0.0.0.0"`                           |
| `port`       | int                 | `9000`            | 监听端口，1~65535                                            |
| `endpoint`   | string              | `"/image_defect"` | 接口路径，必须以 `/` 开头且不含空白                          |
| `model`      | string / null       | `null`            | ultralytics `*.pt` 权重路径或模型名；**`null` 即 mock 模式** |
| `device`     | string / int / null | `null`            | 推理设备，如 `"cpu"`、`0`；`null` 交给 ultralytics 自选      |
| `conf`       | number              | `0.25`            | YOLO 置信度阈值，0~1                                         |
| `share_root` | string              | `"./shared"`      | 共享盘根目录（客户端 `relative_dir` 的基准），启动时自动创建 |

校验规则：**出现未知键直接报错退出**（拦住 `modle` 这类拼写错误导致的"静默退化成 mock"）；类型/范围不对同样报错退出，错误信息带完整文件路径。

示例——真实模型 + 对外监听：

```json
{
  "host": "0.0.0.0",
  "port": 9000,
  "endpoint": "/image_defect",
  "model": "yolo11n.pt",
  "device": 0,
  "conf": 0.3,
  "share_root": "./shared"
}
```

示例——共享盘指向映射盘（只写要改的键，其余走默认值）：

```json
{
  "share_root": "Z:\\OriginalImages"
}
```

保存后**重启** `python server.py` 才生效：配置只在启动时读一次，运行期不重读，请求里也没有任何字段能覆盖这些设置。

### 4.2 客户端 client.py

全局参数（**必须写在子命令之前**）：

| 参数          | 默认值                               | 说明           |
| ------------- | ------------------------------------ | -------------- |
| `--url`       | `http://127.0.0.1:9000/image_defect` | 服务地址       |
| `--timeout`   | `30.0`                               | 请求超时（秒） |
| `--job-id`    | 自动生成                             | 任务号         |
| `--sample-id` | `S00001`                             | 工件编号       |
| `--pose-id`   | `P01`                                | 位姿编号       |

子命令：

| 子命令        | 参数                                                                                      | 作用                                               |
| ------------- | ----------------------------------------------------------------------------------------- | -------------------------------------------------- |
| `gen-images`  | `--out`（默认 `test_images`）、`--count`（默认 2）、`--width`（1920）、`--height`（1440） | 生成随机测试图，按协议风格命名                     |
| `send-base64` | `--images <路径...>`；**省略则自动生成一张内存测试图**                                    | base64 传图请求                                    |
| `send-path`   | `--dir <相对目录>`（必填）、`--files <文件名...>`（必填）                                 | 共享盘传图请求                                     |
| `test-errors` | 无                                                                                        | 自动验证 13 个错误码用例，退出码 0=全过 / 1=有失败 |

示例：

```bash
# 指定服务地址与工件号
python client.py --url http://192.168.1.50:9000/image_defect --sample-id W2026001 send-base64

# 多张图一起推理
python client.py send-base64 --images a.jpg b.jpg

# 长耗时推理放宽超时
python client.py --timeout 120 send-base64
```

## 5. 共享盘传图完整流程

共享盘模式中，**客户端只传相对路径**，服务端自行从 `share_root` 读取图片。本地联调流程：

```bash
# 终端 1：启动服务（共享根 = ./shared）
python server.py

# 终端 2：把测试图直接生成到共享根下的 demo 子目录
python client.py gen-images --count 2 --out shared\demo

# 用相对目录 + 文件名发送
python client.py send-path --dir demo --files S00001_C01_P01_L0_20260211103015.jpg S00001_C01_P01_L1_20260211103015.jpg
```

`--dir` 写相对路径（如 `demo` 或 `20260211103015/Original/S00001`），服务端拼 `share_root/relative_dir/file_name` 读取。生产环境中 `share_root` 指向网络映射盘（如 `Z:\`），客户端与算法服务共用同一块盘。

安全机制：`relative_dir` 含 `..` 逃出共享根会被拒绝（-1003），文件不存在返回 -1004。

## 6. 真实模型模式

把 `config.json` 的 `model` 改成权重路径或模型名，再重启服务：

```json
{
  "model": "yolo11n.pt"
}
```

```bash
python server.py
```

- 首次运行会自动下载 `yolo11n.pt` 权重；下载失败/路径写错会在启动阶段就崩（进程不监听端口）。
- 检出框 `xyxy` 直接映射协议 `box`；类别名映射 `code`；置信度映射 `score`。
- `area/length/volume` 目前是**框的占位值**（框面积/框长边），接入真实测量逻辑见 README §6。
- 无检出时 `results: []`，`error_code` 仍为 0 —— 这是正常业务结果。

## 7. 错误码

| error_code | error_msg                | 含义 / 常见原因                                                                                                     |
| ---------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| 0          | `OK`                     | 推理成功（`results` 可为空 = 无缺陷）                                                                               |
| -1001      | 系统错误：未知错误       | 服务端内部异常，看服务端控制台 `[ERROR]` 日志                                                                       |
| -1002      | 参数错误：参数不合法     | JSON 非法、缺 job_id/sample_id/file_names、数组长度不匹配、两种传图方式并存或都缺失                                 |
| -1003      | 参数错误：图片名称错误   | 文件名空白、含非法字符（白名单：字母数字下划线点空格横线）、扩展名不在 jpg/jpeg/png/bmp/tif/tiff、relative_dir 越界 |
| -1004      | 参数错误：图片数据不存在 | base64 解码失败/为空、图片字节无法解码、共享盘文件不存在或不可读                                                    |

## 8. C# 对接参考

协议文档中的 C# 示例为对接视角。与 JSON 协议对应的两种发法：

**base64 传图（RestSharp）：**

```csharp
var payload = new
{
    job_id = $"{pId}{DateTime.Now:yyyyMMddHhmmsffff}",
    sample_id = workpieceCode,
    pose_id = "P01",
    file_names = new List<string> { imageName },
    images_base64 = new List<string> { Convert.ToBase64String(imageBytes) },
    ext_params = new Dictionary<string, string>()
};
var request = new RestRequest
{
    Method = Method.Post,
    Timeout = TimeSpan.FromMilliseconds(inspectTimeout),
};
request.AddStringBody(JsonConvert.SerializeObject(payload), DataFormat.Json);
var response = await restClient.ExecuteAsync(request);
// response.StatusCode == OK 且 body 中 error_code == 0 即成功
```

**共享盘传图（HttpWebRequest）：**

```csharp
var payload = new
{
    job_id = $"{pId}{DateTime.Now:yyyyMMddHhmmsffff}",
    sample_id = workpieceCode,
    pose_id = pId,
    file_names = new List<string> { Path.GetFileName(imagePath) },
    relative_dir = "2021111093056/Original/S00001/",
    ext_params = new Dictionary<string, string>(),
};
// POST JSON 到 http://<算法机>:9000/image_defect，UTF-8 编码
// 响应体为 JSON，判断顶层 error_code 字段
```

注意：文档中出现的 `Data-Header + application/octet-stream` 二进制方案是历史残留示例，**本服务实现的是 JSON 报文方案**（协议第 1、2 节明确"请求和响应内容均在 JSON 消息体中"）。如对接方确有二进制头需求，需另行扩展服务端。

## 9. 常见问题（FAQ）

| 现象                                      | 原因与处理                                                                                                                                                                           |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `connection failed: ...`                  | 服务未启动 / 地址端口不对 / 防火墙拦截。先确认服务端控制台有没有打印 `ShimadaPredictService started`（配置不合法会在启动阶段直接退出），再用 `python client.py test-errors` 快速排查 |
| 中文文件名能不能用                        | 可以：白名单正则的 `\w` 在 Python 3 下包含中文，`中文名.jpg` 实测能通过；只需保证请求 body 是 UTF-8。仍建议用 ASCII 命名，避免对接方编码差异                                         |
| 端口被占用                                | `OSError: [WinError 10048]` → 改 `config.json` 的 `port` 或释放 9000 端口                                                                                                            |
| 响应 404 + `-1002` unknown endpoint       | URL 路径与 `config.json` 的 `endpoint` 不一致                                                                                                                                        |
| 响应 -1004 但文件确实存在                 | 服务端 `share_root` 与客户端 `--dir` 拼出的绝对路径不对；共享盘模式要求**服务端**能访问该路径，不是客户端                                                                            |
| 启动即退出：`config file not found`       | 工作目录不对（`config.json` 只从 `os.getcwd()` 读，不按脚本所在目录找）；`cd` 到 `shimada_predict_service` 再启动                                                                    |
| 启动即退出：`has unknown keys ['modle']`  | 键名拼错，按提示里给出的允许键列表改                                                                                                                                                 |
| 改了 `model` 但没生效                     | 配置只在启动时读一次，必须重启 `python server.py`                                                                                                                                    |
| 大图请求被拒 -1002 invalid content length | 请求体超过 256MB 上限（`kMaxBodyBytes`），请压缩图片或调整上限                                                                                                                       |
| 想看每次请求耗时                          | 看服务端 `[REQ] ... elapsed=...ms` 日志，或响应头 `X-Process-Ms`                                                                                                                     |
| 停止服务                                  | 服务端控制台 `Ctrl+C`                                                                                                                                                                |

## 10. 文件清单

| 文件               | 说明                                                                                                                                  |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| `server.py`        | 服务端（接口 + mock/YOLO 引擎 + `config.json` 加载）                                                                                  |
| `client.py`        | 测试客户端                                                                                                                            |
| `config.json`      | 服务端唯一配置（工作目录下读取；无命令行参数）                                                                                        |
| `examples/`        | 报文示例：`request_base64.json`、`request_share_dir.json`、`response_success.json`、`response_failure.json`（仅对照用，运行时不读取） |
| `requirements.txt` | 依赖：`pillow`（必装）、`ultralytics`（可选）                                                                                         |
| `README.md`        | 完整代码逻辑说明                                                                                                                      |
| `USAGE.md`         | 本手册                                                                                                                                |
| `shared/`          | 服务端共享盘根目录（运行时生成）                                                                                                      |
| `test_images/`     | 客户端测试图输出目录（运行时生成）                                                                                                    |
