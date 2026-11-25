# USAGE — ShimadaPredictService 使用手册

面向产线联调与对接测试的完整操作手册。协议依据《软件算法通讯协议》：`POST http://127.0.0.1:9000/image_defect`，JSON + UTF-8。

---

## 1. 环境要求

| 项 | 要求 |
|---|---|
| Python | ≥ 3.8（开发验证环境 3.10） |
| pillow | **必装**（读取图片尺寸 / 生成测试图） |
| ultralytics | 可选，仅 `config.json` 里某个模型别名的 `path` 非 `null`（真实推理）时需要（会自动带上 torch、opencv 等依赖） |
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

**终端 1 — 启动服务：**

```bash
cd D:\m_code\dzy\ultralytics-main\shimada-predict_service
python server.py
```

服务端**不接收任何命令行参数**，启动时只读当前工作目录下的 `config.json`，所以必须在 `shimada-predict_service` 目录内执行（或把 `config.json` 放到你的工作目录）。仓库自带的 `config.json` 是 mock 模式，不装模型即可跑通。

启动输出（同时写入 `log/YYYYMMDD.log`）：

```
2026-11-23 11:38:28 [SERVICE] starting | config=D:\...\shimada-predict_service\config.json endpoint=http://127.0.0.1:9000/image_defect default_model=default device=None share_root=D:\...\shimada-predict_service\shared
2026-11-23 11:38:28 [MODEL] loaded | alias=default path=None engine=mock conf=0.25
2026-11-23 11:38:28 [WORKPIECE] A100 -> default
2026-11-23 11:38:28 [SERVICE] started | pid=7800 log=D:\...\shimada-predict_service\log
Press Ctrl+C to stop.
```

每个别名一行 `[MODEL] loaded`：`engine=mock`（`path` 为 `null`）或 `engine=yolo` + 权重路径；每个型号再一行 `[WORKPIECE] 型号 -> 别名, 别名`。配置多个模型时这里会列出全部。

配置有问题（文件缺失、键名拼错、类型/范围不对）会写一行 `[SERVICE] [错误] [ng] <原因>` 并以退出码 1 结束，**不会打开端口**，客户端这时只会看到 `connection failed`。

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

# ⑤ 服务端逻辑单测（校验/装图/配置/路由/日志，不需要模型和网络）
python -m pytest tests -q
```

请求里不含 `model` / `workpiece` 字段：服务端按文件名解析型号（规则未定，见 §4.1 的 `workpieces`），再查配对表决定跑哪些模型。

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

`server.py` **不接收命令行参数**，启动时只读工作目录（`os.getcwd()`）下的 `config.json`，必须是 UTF-8 编码的 JSON 对象。8 个键**全部可选**，缺省值如下：

| 键 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `host` | string | `"127.0.0.1"` | 监听地址；对外服务改为 `"0.0.0.0"` |
| `port` | int | `9000` | 监听端口，1~65535 |
| `endpoint` | string | `"/image_defect"` | 接口路径，必须以 `/` 开头且不含空白 |
| `device` | string / int / null | `null` | 推理设备，如 `"cpu"`、`0`；**所有模型统一使用**；`null` 交给 ultralytics 自选 |
| `models` | object | `{"default": {"path": null}}` | 模型别名表，键即别名；每个值只能有 `path` 和 `conf` 两个字段，见下 |
| `default_model` | string / null | `null` | 型号解析失败时的回落别名；省略时仅当 `models` 只有一项才自动取它，多项必须显式指定 |
| `workpieces` | object | `{}` | 工件型号 → 模型别名数组的**配对表**，决定某型号跑哪些模型；不影响模型加载，见下 |
| `share_root` | string | `"./shared"` | 共享盘根目录（客户端 `relative_dir` 的基准），启动时自动创建 |

`models` 的每个条目：

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `path` | string / null | 必填 | ultralytics `*.pt` 权重路径或模型名；**`null` = 该别名使用 mock 引擎** |
| `conf` | number | `0.25` | 该模型的置信度阈值，0~1；mock 引擎忽略此值 |

别名限制为 `^[A-Za-z0-9_.-]{1,64}$`（会进日志，所以不允许中文/空格/引号）。

`workpieces` 的每个条目是"型号 → 模型别名数组"：

| 字段 | 类型 | 说明 |
|---|---|---|
| 型号（键） | string | 工件型号编号，`^\S{1,64}$`（不含空白，最长 64 字符） |
| 模型别名数组（值） | string[] | 非空数组，元素必须是 `models` 里已存在的别名；**顺序即业务调用顺序**，同一型号内不能重复 |

配对表只决定"这个型号要跑哪些模型"，**不影响加载**：启动时仍然把 `models` 里所有别名全部加载。业务代码用 `workpieceModels(型号)` 取引擎句柄列表（未知型号返回 `None`）。

型号来源是**请求里的文件名**：`parseWorkpiece(file_names[0])` 从文件名解析型号。目前解析规则与型号清单都未定，函数留空返回 `None`（代码里有 `TODO`），此时服务端回落 `default_model` 跑单模型，行为与之前一致。规则确定后只改 `parseWorkpiece` 一个函数即可，配对表、多模型执行、日志都已就绪。

校验规则：**出现未知键直接报错退出**（顶层未知键、`models` 条目内未知键都拦，防止 `modle` 这类拼写错误或误以为支持按模型分卡）；类型/范围不对同样报错退出，错误信息带完整文件路径。

示例——两个模型 + 对外监听（`device` 统一，`conf` 按模型覆盖）：

```json
{
    "host": "0.0.0.0",
    "port": 9000,
    "endpoint": "/image_defect",
    "device": 0,
    "models": {
        "fd_v3": { "path": "Z:\\models\\fd_v3.pt", "conf": 0.35 },
        "seg_v1": { "path": "Z:\\models\\seg_v1.pt" }
    },
    "default_model": "fd_v3",
    "workpieces": {
        "A100": ["fd_v3", "seg_v1"],
        "B200": ["fd_v3"]
    },
    "share_root": "./shared"
}
```

请求里**不带** `model` 字段，型号由服务端从文件名解析：

```json
{ "job_id": "J001", "sample_id": "S00001", "file_names": ["A100_C01_P01_L0_20261123103015.jpg"], "relative_dir": "demo/" }
```

- 解析出型号后按 `workpieces` 跑该型号的全部模型，结果合并返回；响应按协议原样输出，**不额外回显型号/模型**。
- 型号在表里查不到 → `-1002`（`error_msg` 为 `unknown workpiece: xxx`）。
- 解析规则未定期间 `parseWorkpiece` 恒返回 `None`，走 `default_model` 单模型。想知道实际跑了哪些模型，看服务端 `[REQ] ... workpiece=... models=...` 日志。

示例——共享盘指向映射盘（只写要改的键，其余走默认值）：

```json
{
    "share_root": "Z:\\OriginalImages"
}
```

保存后**重启** `python server.py` 才生效：配置只在启动时读一次，运行期不重读，请求里没有任何字段能覆盖这些设置。

校验规则补充：`models` 为空、模型条目含未知键（如误写 `device`）、`default_model` 指向不存在的别名、`workpieces` 引用了不存在的别名或同一型号内重复、型号键含空白，都会启动即报错退出。

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

在 `config.json` 的 `models` 里给别名配权重路径（`path`），`device` 统一写在顶层；再重启服务：

```json
{
    "device": 0,
    "models": {
        "yolo11n": { "path": "yolo11n.pt", "conf": 0.3 }
    },
    "default_model": "yolo11n"
}
```

```bash
python server.py
```

- 启动时**一次性加载全部别名**（fail-fast）：首次运行会自动下载 `yolo11n.pt`；任一权重路径写错/下载失败会在启动阶段就崩（进程不监听端口）。
- 跑哪个模型由**文件名解析出的型号**决定：`workpieces` 把型号映射到 1~N 个别名，全部按顺序执行并合并结果。多个模型可以共存，启动日志会逐个列出。
- 解析规则未定期间 `parseWorkpiece` 返回 `None`，回落 `default_model`（示例里是 `yolo11n`），因此现在仍能正常单模型推理。
- 检出框 `xyxy` 直接映射协议 `box`；类别名映射 `code`；置信度映射 `score`。
- `area/length/volume` 目前是**框的占位值**（框面积/框长边），接入真实测量逻辑见 README §6。
- 无检出时 `results: []`，`error_code` 仍为 0 —— 这是正常业务结果。

## 7. 运行日志

服务端把启动、模型加载、每个请求与响应都写到**工作目录下的 `log/`**，控制台同步打印一份。

| 项 | 规则 |
|---|---|
| 文件命名 | `log/YYYYMMDD.log`，按写入当天切换（跨天自动新建，无需重启） |
| 数量上限 | 最多 30 个日期文件（`kLogRetention`）；跨天首次写入时删最旧的，同一天内重启不清理 |
| 编码 | UTF-8、LF，一行一条，行首为 `YYYY-MM-DD HH:MM:SS` |
| 并发 | 全局锁串行写入，多请求并发不会写乱行 |
| 目录不可写 | 只打印一次 `[WARN] log file disabled`，服务照常推理，只是不落文件 |

一次正常请求在日志里是 `[REQ]` + `[RESP]` 两行：

```
2026-11-23 11:38:28 [REQ] received | job_id=J20261123113828838 sample_id=S00001 pose_id=P01 files=S00001_C01_P01_L0_20261123113828.jpg mode=share_dir
2026-11-23 11:38:28 [RESP] [成功] [ok] job_id=J20261123113828838 sample_id=S00001 pose_id=P01 files=S00001_C01_P01_L0_20261123113828.jpg workpiece=- models=default defects=QJ:1,HS:1 total=2 elapsed=156.1ms
```

- `[成功] [ok]` 表示 `error_code=0`（`total=0` 也算成功，即无缺陷）；`[错误] [ng]` 表示任意非 0 错误码，后面跟 `error_code` / `error_msg`。
- `defects=CODE:数量,...` 是该次响应中各缺陷代码的条数，`total` 是总条数，无缺陷时为 `defects=-`。
- `workpiece` / `models` 是实际路由结果：解析规则未实现期间固定为 `workpiece=- models=<default_model>`；接入 `parseWorkpiece` 后显示解析出的型号及其全部别名。
- 启动阶段另有 `[SERVICE]`（启动/停止/配置错误）、`[MODEL]`（每个别名的 `path`，yolo 模型还有一行 `warmup`）、`[WORKPIECE]`（配对表）、`[IMG]`（某张图被拒的原因：base64 非法、超字节/像素上限、解不开）几类行。
- 请求过多时闸门会直接拒绝（`-1001`，`error_msg` 为"系统繁忙：并发请求过多，请稍后重试"），日志同样有 `[REQ]` / `[RESP]` 两行，可据此判断是过载而不是代码错误。
- 只记录文件名，不记录图片内容、base64 或 `relative_dir`。

排查建议：拿 `job_id` 去当天文件里搜；日志里的 `elapsed` 与响应头 `X-Process-Ms` 同源（都是服务端处理耗时，不含网络）。

## 8. 错误码

| error_code | error_msg | 含义 / 常见原因 |
|---|---|---|
| 0 | `OK` | 推理成功（`results` 可为空 = 无缺陷） |
| -1001 | 系统错误：未知错误 | 服务端内部异常，看 `log/` 当天文件里的 `[RESP] [错误]` 行；**并发超过 8 个时也会返回 -1001，`error_msg` 为"系统繁忙：并发请求过多，请稍后重试"** |
| -1002 | 参数错误：参数不合法 | JSON 非法、缺 job_id/sample_id/file_names、数组长度不匹配、两种传图方式并存或都缺失、**文件名解析出的型号不在 `workpieces` 中（error_msg 为 `unknown workpiece: xxx`）** |
| -1003 | 参数错误：图片名称错误 | 文件名空白、含非法字符（白名单：字母数字下划线点空格横线）、扩展名不在 jpg/jpeg/png/bmp/tif/tiff、relative_dir 越界 |
| -1004 | 参数错误：图片数据不存在 | base64 含非法字符/解码失败/为空、单图超过 128MB 或解码后超过 3000 万像素、图片字节无法解码、共享盘文件不存在或不可读 |

## 9. C# 对接参考

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

## 10. 常见问题（FAQ）

| 现象 | 原因与处理 |
|---|---|
| `connection failed: ...` | 服务未启动 / 地址端口不对 / 防火墙拦截。先确认服务端控制台有没有打印 `[SERVICE] started`（带 pid，配置不合法会在启动阶段直接退出，原因也写进 `log/`），再用 `python client.py test-errors` 快速排查 |
| 中文文件名能不能用 | 可以：白名单正则的 `\w` 在 Python 3 下包含中文，`中文名.jpg` 实测能通过；只需保证请求 body 是 UTF-8。仍建议用 ASCII 命名，避免对接方编码差异 |
| 端口被占用 | `OSError: [WinError 10048]` → 改 `config.json` 的 `port` 或释放 9000 端口 |
| 响应 404 + `-1002` unknown endpoint | URL 路径与 `config.json` 的 `endpoint` 不一致 |
| 响应 -1004 但文件确实存在 | 服务端 `share_root` 与客户端 `--dir` 拼出的绝对路径不对；共享盘模式要求**服务端**能访问该路径，不是客户端 |
| 启动即退出：`config file not found` | 工作目录不对（`config.json` 只从 `os.getcwd()` 读，不按脚本所在目录找）；`cd` 到 `shimada_predict_service` 再启动 |
| 启动即退出：`has unknown keys ['modle']` | 键名拼错，按提示里给出的允许键列表改 |
| 启动即退出：`default_model is required` | `models` 里有多个别名却没写 `default_model`，补一个（只有一个模型时才允许省略） |
| 启动即退出：`model 'x' has unknown keys ['device']` | `device` 是全局统一配置，不能写在模型条目里；模型条目只允许 `path` / `conf` |
| 启动即退出：`workpiece 'X' references unknown model alias ...` | `workpieces` 里引用的别名必须在 `models` 中存在（区分大小写） |
| 启动即退出：`workpiece 'X' lists the same model alias more than once` | 同一型号下不要重复写同一个别名 |
| 响应 `unknown workpiece: xxx` | 文件名解析出的型号不在 `workpieces` 表里。解析规则定下后才可能出现；补配对项或让解析函数回退 `None` |
| 改了 `models` / `workpieces` 但没生效 | 配置只在启动时读一次，必须重启 `python server.py`；新增模型、改配对同样要重启 |
| 大图请求被拒 -1002 invalid content length | 请求体超过 256MB 上限（`kMaxBodyBytes`），请压缩图片或调整上限 |
| 响应 -1001 `系统繁忙：并发请求过多` | 同时处理中的请求超过 8 个（`kMaxInflight`），服务端直接拒绝而不是排队；降低客户端并发、加重试退避，或按机器内存调大该常量 |
| 响应 -1004 但图片看起来正常 | 单图超过 128MB（`kMaxImageBytes`）或解码后超过 3000 万像素（`kMaxImagePixels`）；看日志里的 `[IMG]` 行确认是 bytes 还是 pixels 超限 |
| 改了代码但行为没变 | Windows 上两个实例可能同时占用 9000（`SO_REUSEADDR`），先确认没有残留 `python.exe` 再启动 |
| 想看每次请求耗时 | 看 `log/YYYYMMDD.log` 里 `[RESP] ... elapsed=...ms`，或响应头 `X-Process-Ms` |
| 日志在哪、能留多久 | 工作目录下 `log/`，按天一个文件，最多 30 个（见 §7）；同一天重启会续写同一个文件 |
| 磁盘满了 / 不想写日志 | 删旧文件即可（运行中删除安全）；目录设为只读只会告警一次，服务不中断 |
| 停止服务 | 服务端控制台 `Ctrl+C` |

## 11. 文件清单

| 文件 | 说明 |
|---|---|
| `server.py` | 服务端（接口 + mock/YOLO 引擎 + `config.json` 加载） |
| `client.py` | 测试客户端 |
| `config.json` | 服务端唯一配置（工作目录下读取；含 `models` 别名表 + `workpieces` 配对表 + `default_model`，无命令行参数） |
| `examples/` | 报文示例：`request_base64.json`、`request_share_dir.json`、`response_success.json`、`response_failure.json`（仅对照用，运行时不读取） |
| `requirements.txt` | 依赖：`pillow`（必装）、`ultralytics`（可选） |
| `tests/` | pytest 用例：请求校验、装图与大小限制、配置校验、型号路由、日志保留 |
| `README.md` | 完整代码逻辑说明 |
| `USAGE.md` | 本手册 |
| `shared/` | 服务端共享盘根目录（运行时生成） |
| `log/` | 运行日志目录（运行时生成，`YYYYMMDD.log`，最多 30 个文件） |
| `test_images/` | 客户端测试图输出目录（运行时生成） |
