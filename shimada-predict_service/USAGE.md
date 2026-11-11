# USAGE — ShimadaPredictService 使用手册

面向产线联调与对接测试的完整操作手册。协议依据《软件算法通讯协议》：`POST http://127.0.0.1:9000/image_defect`，JSON + UTF-8。

---

## 1. 环境要求

| 项 | 要求 |
|---|---|
| Python | ≥ 3.8（开发验证环境 3.10） |
| pillow | **必装**（读取图片尺寸 / 生成测试图） |
| ultralytics | 可选，仅 `--model` 真实推理模式需要（会自动带上 torch、opencv 等依赖） |
| 操作系统 | Windows / Linux 均可（路径处理已做跨平台） |

## 2. 安装

```bash
# 仅通讯联调（mock 模式）
pip install pillow

# 真实模型推理模式
pip install ultralytics
```

仓库根目录已有完整 ultralytics 环境时，服务端可直接运行，无需额外安装。

## 3. 快速开始（3 步测通）

**终端 1 — 启动服务（默认 mock 模式，无需模型）：**

```bash
cd E:\m_code\ultralytics-main\shimada-predict_service
python server.py
```

启动输出：

```
ShimadaPredictService started
  endpoint   : http://127.0.0.1:9000/image_defect
  mode       : mock
  share root : E:\m_code\ultralytics-main\shimada-predict_service\shared
Press Ctrl+C to stop.
```

**终端 2 — 客户端测试：**

```bash
cd E:\m_code\ultralytics-main\shimada-predict_service

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

## 4. 命令行参数

### 4.1 服务端 server.py

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--host` | `127.0.0.1` | 监听地址；对外服务改为 `0.0.0.0` |
| `--port` | `9000` | 监听端口 |
| `--endpoint` | `/image_defect` | 接口路径 |
| `--model` | 无 | ultralytics `*.pt` 权重路径；**缺省即 mock 模式** |
| `--device` | 自动 | 推理设备，如 `cpu`、`0` |
| `--conf` | `0.25` | YOLO 置信度阈值 |
| `--share-root` | `./shared` | 共享盘根目录（客户端 `relative_dir` 的基准路径），启动时自动创建 |

示例：

```bash
# 真实模型 + 对外监听
python server.py --host 0.0.0.0 --model yolo11n.pt --device 0 --conf 0.3

# 共享盘指向映射盘
python server.py --share-root "Z:\OriginalImages"
```

### 4.2 客户端 client.py

全局参数（**必须写在子命令之前**）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--url` | `http://127.0.0.1:9000/image_defect` | 服务地址 |
| `--timeout` | `30.0` | 请求超时（秒） |
| `--job-id` | 自动生成 | 任务号 |
| `--sample-id` | `S00001` | 工件编号 |
| `--pose-id` | `P01` | 位姿编号 |

子命令：

| 子命令 | 参数 | 作用 |
|---|---|---|
| `gen-images` | `--out`（默认 `test_images`）、`--count`（默认 2）、`--width`（1920）、`--height`（1440） | 生成随机测试图，按协议风格命名 |
| `send-base64` | `--images <路径...>`；**省略则自动生成一张内存测试图** | base64 传图请求 |
| `send-path` | `--dir <相对目录>`（必填）、`--files <文件名...>`（必填） | 共享盘传图请求 |
| `test-errors` | 无 | 自动验证 13 个错误码用例，退出码 0=全过 / 1=有失败 |

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

共享盘模式中，**客户端只传相对路径**，服务端自行从 `--share-root` 读取图片。本地联调流程：

```bash
# 终端 1：启动服务（共享根 = ./shared）
python server.py

# 终端 2：把测试图直接生成到共享根下的 demo 子目录
python client.py gen-images --count 2 --out shared\demo

# 用相对目录 + 文件名发送
python client.py send-path --dir demo --files S00001_C01_P01_L0_20260211103015.jpg S00001_C01_P01_L1_20260211103015.jpg
```

`--dir` 写相对路径（如 `demo` 或 `20260211103015/Original/S00001`），服务端拼 `shareRoot/relative_dir/file_name` 读取。生产环境中 `--share-root` 指向网络映射盘（如 `Z:\`），客户端与算法服务共用同一块盘。

安全机制：`relative_dir` 含 `..` 逃出共享根会被拒绝（-1003），文件不存在返回 -1004。

## 6. 真实模型模式

```bash
# 首次运行会自动下载 yolo11n.pt 权重
python server.py --model yolo11n.pt
```

- 检出框 `xyxy` 直接映射协议 `box`；类别名映射 `code`；置信度映射 `score`。
- `area/length/volume` 目前是**框的占位值**（框面积/框长边），接入真实测量逻辑见 README §6。
- 无检出时 `results: []`，`error_code` 仍为 0 —— 这是正常业务结果。

## 7. 错误码

| error_code | error_msg | 含义 / 常见原因 |
|---|---|---|
| 0 | `OK` | 推理成功（`results` 可为空 = 无缺陷） |
| -1001 | 系统错误：未知错误 | 服务端内部异常，看服务端控制台 `[ERROR]` 日志 |
| -1002 | 参数错误：参数不合法 | JSON 非法、缺 job_id/sample_id/file_names、数组长度不匹配、两种传图方式并存或都缺失 |
| -1003 | 参数错误：图片名称错误 | 文件名空白、含非法字符（白名单：字母数字下划线点空格横线）、扩展名不在 jpg/jpeg/png/bmp/tif/tiff、relative_dir 越界 |
| -1004 | 参数错误：图片数据不存在 | base64 解码失败/为空、图片字节无法解码、共享盘文件不存在或不可读 |

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

| 现象 | 原因与处理 |
|---|---|
| `connection failed: ...` | 服务未启动 / 地址端口不对 / 防火墙拦截。先用 `python client.py test-errors` 快速排查 |
| 中文文件名发不出去 | 协议文件名白名单为 ASCII 字符集；请用协议命名风格（如 `S00001_C01_P01_L0_时间戳.jpg`），中文信息放 `ext_params` |
| 端口被占用 | `OSError: [WinError 10048]` → 换 `--port` 或释放 9000 端口 |
| 响应 404 + `-1002` unknown endpoint | URL 路径与 `--endpoint` 不一致 |
| 响应 -1004 但文件确实存在 | 服务端 `--share-root` 与客户端 `--dir` 拼出的绝对路径不对；共享盘模式要求**服务端**能访问该路径，不是客户端 |
| 大图请求被拒 -1002 invalid content length | 请求体超过 256MB 上限（`kMaxBodyBytes`），请压缩图片或调整上限 |
| 想看每次请求耗时 | 看服务端 `[REQ] ... elapsed=...ms` 日志，或响应头 `X-Process-Ms` |
| 停止服务 | 服务端控制台 `Ctrl+C` |

## 10. 文件清单

| 文件 | 说明 |
|---|---|
| `server.py` | 服务端（接口 + mock/YOLO 引擎） |
| `client.py` | 测试客户端 |
| `requirements.txt` | 依赖：`pillow`（必装）、`ultralytics`（可选） |
| `README.md` | 完整代码逻辑说明 |
| `USAGE.md` | 本手册 |
| `shared/` | 服务端共享盘根目录（运行时生成） |
| `test_images/` | 客户端测试图输出目录（运行时生成） |
