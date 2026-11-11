# ShimadaPredictService — 图像缺陷推理服务与通讯测试

依据《软件算法通讯协议》实现的 **HTTP + JSON 缺陷检测推理服务**，以及配套的 **独立测试客户端**。两个程序完全自包含、互不依赖，可单独拷贝分发。

- `server.py` — 服务端。实现 `/image_defect` 推理接口，支持 base64 传图与共享盘路径传图两种方式；默认内置 mock 推理引擎（零模型依赖），可通过 `--model` 切换为 ultralytics YOLO 真实推理。
- `client.py` — 测试客户端。生成测试图、发送两种协议请求、自动验证全部错误码。

使用方法见 [USAGE.md](USAGE.md)，本文档描述完整代码逻辑。

---

## 1. 协议要点（实现依据）

| 项 | 约定 |
|---|---|
| 协议 | JSON + HTTP，UTF-8 编码 |
| 接口地址 | `http://127.0.0.1:9000/image_defect`（POST） |
| 传图方式 | ① `images_base64`：base64 字符串数组，与 `file_names` 一一对应；② `relative_dir`：共享盘相对目录 + `file_names` |
| 业务错误 | 通过 JSON 顶层 `error_code` / `error_msg` 表达 |

错误码表（两个程序中各自定义了同一份常量表，见 §4.1 与 §5.1）：

| error_code | error_msg | 触发条件 |
|---|---|---|
| 0 | `OK` | 推理成功 |
| -1001 | 系统错误：未知错误 | 服务端未捕获异常 |
| -1002 | 参数错误：参数不合法 | JSON 非法 / 缺字段 / 字段类型错 / 两种传图方式冲突 |
| -1003 | 参数错误：图片名称错误 | 文件名为空、含非法字符或扩展名不受支持 |
| -1004 | 参数错误：图片数据不存在 | base64 解码失败、解码结果为空、图片无法解码，或共享盘文件不存在 |

请求示例（base64 方式）：

```json
{
    "job_id": "20210403120516",
    "sample_id": "S00001",
    "pose_id": "P01",
    "file_names": ["S00001_C01_P01_L0_20201024100008.jpg"],
    "images_base64": ["<base64...>"],
    "ext_params": {"xxxx1": "value1"}
}
```

请求示例（共享盘方式）：把 `images_base64` 替换为 `"relative_dir": "2021111093056/Original/S00001/"`。

响应示例：

```json
{
    "error_code": 0,
    "error_msg": "OK",
    "data": {
        "job_id": "...", "sample_id": "S00001", "pose_id": "P01",
        "file_names": ["..."],
        "image_width": 1920, "image_height": 1440, "grayscale": 0,
        "results": [
            {
                "code": "ZM", "score": 0.4545, "area": 1596,
                "length": 1000, "volume": 1000,
                "box": [100, 100, 200, 200],
                "polygon": [], "defect_raw": []
            }
        ],
        "alg_raw": []
    }
}
```

### 1.1 关键设计决策

| 决策 | 说明 |
|---|---|
| 业务错误也返回 HTTP 200 | 协议文档未定义 HTTP 状态码语义，且其 C# 示例仅在 `HttpStatusCode.OK` 分支解析响应体。因此所有业务错误（-1002/-1003/-1004）一律返回 200 + 错误码；仅"路径不存在 / 无法路由"返回 404。 |
| `results[].file_name` 附加字段 | 协议响应的 `results` 条目无归属字段，但请求支持多图。为便于多图联调，每个缺陷条目附加 `file_name` 标识来源图片。这是**协议扩展字段**，C# 端按 `JObject` 解析不受影响；不需要时可在 `processRequest` 中删除该行。 |
| 多图的 `image_width/height` | 协议响应中分辨率是单值，多图请求取**第一张图**的尺寸。 |
| `ext_params` 不回传 | 协议响应结构无此字段，服务端仅接收、不校验内容、不回传。 |
| 双程序各自持有一份错误码常量 | 换取 `client.py` 可脱离 `server.py` 独立分发给对接方；常量表改动时两边需同步。 |
| `error_msg` 中文文案 | 与协议错误码表原文一致；`error_code` 才是对接方应判断的字段。 |

---

### 1.2 通讯时序：一次推理交互的完整先后

一次成功的推理交互分三个阶段：**客户端本地准备 → 线上一问一答 → 客户端本地收尾**。协议严格遵循"一请求一响应"，无流式、无分块、无进度推送，也**不存在服务端主动推送**。

```
     client.py / C# 对接方                                server.py
────────────────────────────────────────────────    ──────────────────────────────────────
阶段 0：发送前准备（纯本地，无网络交互）
 ① 取图      读本地图片字节（base64 方式）
             或确认图片已写完共享盘（共享盘方式）
 ② 编码      图片字节 → Base64 文本（仅 base64
             方式，体积膨胀约 4/3）
 ③ 组包      job_id/sample_id/pose_id/file_names
             + images_base64 或 relative_dir
             + ext_params 组成 JSON 对象
 ④ 序列化    JSON → UTF-8 字节流
             │
阶段 1：线上交互
             │──────── ⑤ TCP 连接 host:port ──────────►│ accept，每连接一个线程
             │                                          │
             │──────── ⑥ POST /image_defect ───────────►│ ⑦ 读满 body（按 Content-Length）
             │        Host / Content-Type /             │ ⑧ UTF-8 解码 → json.loads
             │        Content-Length / body             │ ⑨ 校验（结构→文件名，§4.3 表）
             │                                          │    └ 任一失败：跳到 ⑫ 立即回错误
             │                                          │ ⑩ 装图（解码 base64 / 读共享盘
             │                                          │    + 试解码验尺寸）
             │                                          │ ⑪ 抢 g_inferLock → 逐张推理
             │                                          │    → 合并 results → 组装 data
             │◄─── ⑫ HTTP 200 + JSON body ──────────────│    + X-Process-Ms + 日志
             │    （Content-Length 必带）                │
阶段 2：收尾（纯本地）
 ⑬ 判读     HTTP 状态（非 200 时 body 仍是 JSON）
             → error_code（0 取 data.results）
             → 逐条消费 results
 ⑭ 连接      归还连接池复用（keep-alive）
```

**服务端各步骤的提前退出点**（校验失败的请求不占推理锁、毫秒级返回）：

| 失败点 | 条件 | 客户端收到 | 推理资源是否消耗 |
|---|---|---|---|
| ⑥ 路径不匹配 | URL 打错 / 用 GET | 404 + -1002，**连接被服务端关闭**（Connection: close） | 否 |
| ⑥/⑦ 长度检查 | body 缺失、> 256MB、chunked 无 Content-Length | 200 + -1002，**连接被服务端关闭** | 否 |
| ⑧ JSON 解析 | 非 JSON / 非 UTF-8 | 200 + -1002 | 否 |
| ⑨ 结构校验 | 缺字段、类型错、两种传图方式冲突 | 200 + -1002 | 否 |
| ⑨ 文件名校验 | 空白名、非法字符、扩展名不支持 | 200 + -1003 | 否 |
| ⑩ 装图 | base64 解码失败、共享盘文件不存在、图片解不开 | 200 + -1004 | 否 |
| ⑪ 推理异常 | 引擎内部错误（整体失败，无部分成功） | 200 + -1001，查服务端 [ERROR] 日志 | 是 |

### 1.3 HTTP 报文级交互示例

以下报文均来自实测抓取（body 为示意缩略）。

**base64 方式请求：**

```http
POST /image_defect HTTP/1.1
Host: 127.0.0.1:9000
Content-Type: application/json
Content-Length: 418

{"job_id": "20210403120516", "sample_id": "S00001", "pose_id": "P01",
 "file_names": ["S00001_C01_P01_L0_20201024100008.jpg"],
 "images_base64": ["IJKDJJFJWHHFL/==TLGKWJGD/GDL..."],
 "ext_params": {"client": "shimada-test-client"}}
```

**共享盘方式请求**：body 中去掉 `images_base64`，换成 `"relative_dir": "2021111093056/Original/S00001/"`（相对服务端 `--share-root`）。

**成功响应（error_code = 0）：**

```http
HTTP/1.1 200 OK
Server: ShimadaPredictService/1.0 Python/3.10.18
Date: Wed, 11 Nov 2026 08:38:00 GMT
Content-Type: application/json; charset=utf-8
Content-Length: 487

{"error_code": 0, "error_msg": "OK", "data": {"job_id": "...", ..., "results": [...]}}
```

**业务错误响应（示例：文件名非法 → -1003）：** 状态码仍是 200，无 `data` 键。

```http
HTTP/1.1 200 OK
Server: ShimadaPredictService/1.0 Python/3.10.18
Content-Type: application/json; charset=utf-8
Content-Length: 78

{"error_code": -1003, "error_msg": "参数错误：图片名称错误"}
```

**路径错误响应（唯一返回非 200 的场景）：**

```http
HTTP/1.1 404 Not Found
Server: ShimadaPredictService/1.0 Python/3.10.18
Content-Type: application/json; charset=utf-8
Content-Length: 67
Connection: close

{"error_code": -1002, "error_msg": "unknown endpoint: /wrong_path"}
```

报文层规则：

- 请求头**全部不校验**（协议约定"HTTP 头部无特殊校验逻辑"）：`Content-Type`、`Authorization` 等仅为惯例，服务端只看 body；`Server` 头可用于识别本服务版本。
- body 是唯一有效载荷，必须为 UTF-8 编码的 JSON 对象；**必须带 `Content-Length`**（`Transfer-Encoding: chunked` 会被拒为 -1002）。
- `Expect: 100-continue` 由 http.server 默认逻辑自动应答 `100 Continue`，RestSharp/HttpClient 场景无需关心。
- 响应必带 `Content-Length`，客户端应按其精确读满一个响应——这也是连接复用安全的前提。

### 1.4 请求字段说明

| 字段 | 类型 | 必填 | 约束与说明 |
|---|---|---|---|
| `job_id` | string | 是 | 非空；任务唯一标识，服务端**原样回传**用于两端对账；不查重、不幂等，重复提交会重复推理 |
| `sample_id` | string | 是 | 非空；工件/样本编号 |
| `pose_id` | string | 建议 | 位姿编号；缺省时响应回传 `""` |
| `file_names` | string[] | 是 | 非空数组；每个元素仅是文件名（不含路径），需匹配白名单 `^[\w\-. ]+\.(jpg|jpeg|png|bmp|tif|tiff)$`（忽略大小写）；响应中按请求顺序回传 |
| `images_base64` | string[] | 二选一 | 与 `file_names` **等长且一一对应**；标准 Base64 文本（建议含 padding）；空串按 -1004 处理 |
| `relative_dir` | string | 二选一 | 相对服务端 `--share-root` 的目录，含 `..` 越界 → -1003；与 `images_base64` 互斥，并存 → -1002，均缺 → -1002 |
| `ext_params` | object | 否 | 业务扩展参数，服务端不解析、不回传 |

### 1.5 响应字段说明

顶层（错误响应只有前两个键）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `error_code` | int | **对接判断的唯一依据**：0 成功，非 0 见错误码表 |
| `error_msg` | string | 中文文案，仅提示用 |
| `data` | object | 仅 `error_code == 0` 时存在 |

data：

| 字段 | 类型 | 说明 |
|---|---|---|
| `job_id` / `sample_id` / `pose_id` | string | 回传请求值（pose_id 未传为 `""`） |
| `file_names` | string[] | 回传请求值（顺序不变） |
| `image_width` / `image_height` | int | 第一张图的像素分辨率（多图时其余图见 results 内框坐标范围） |
| `grayscale` | int | 恒 0（占位） |
| `results` | object[] | **全部图片的缺陷合并列表**；无缺陷时空数组（正常业务结果） |
| `alg_raw` | array | 恒空（预留） |

results 条目：

| 字段 | 类型 | 说明 |
|---|---|---|
| `file_name` | string | **协议扩展字段**：来源图片名（多图归因） |
| `code` | string | 缺陷类别码（mock：ZM/HS/QJ/WS；yolo：模型类别名） |
| `score` | float | 置信度 0~1，4 位小数 |
| `area` | int | 缺陷面积 px²（当前 = 框面积） |
| `length` | int | 缺陷长度 px（当前 = 框长边） |
| `volume` | int | 缺陷体积（当前 = area 占位） |
| `box` | int[4] | `[左顶点X, 左顶点Y, 右顶点X, 右顶点Y]` 像素坐标（即 x1,y1,x2,y2） |
| `polygon` / `defect_raw` | array | 恒空（预留，分割模型可填） |

### 1.6 交互规则清单

| 规则 | 细节 |
|---|---|
| 编码 | 请求与响应 body 均为 UTF-8；服务端 `ensure_ascii=False` 不转义中文，错误文案是中文；勿按字符数假设 body 长度（中文占多字节），以 `Content-Length` 为准 |
| 连接复用 | 服务端 HTTP/1.1 keep-alive：正常请求响应后连接保持，RestSharp/HttpClient 连接池可复用；仅 404 与 body 超限两个分支主动发 `Connection: close` 并断开（原因：这两分支未消费请求体，复用会污染下一请求） |
| 并发与排队 | 每连接一个线程，多连接可同时到达；但推理由 `g_inferLock` 串行 → 响应耗时 ≈ 前面积压请求的推理时间之和 + 本次推理时间；客户端超时必须覆盖排队时间 |
| 超时分工 | 客户端 timeout（C# `inspectTimeout`）覆盖"发送+排队+推理+响应"全程；服务端 socket 300s 超时用于踢掉挂死连接；两边应满足：客户端 timeout > 服务端最坏处理耗时 |
| 请求体上限 | 256MB（`kMaxBodyBytes`）；base64 体积膨胀约 4/3，即原始图约 ≤ 190MB；超限直接拒为 -1002 不读 body |
| 多图语义 | 逐张装图、逐张推理，结果按请求顺序合并进 `results`；**任一张失败则整个请求失败**，无部分成功 |
| 共享盘时序前置 | 必须先把图片**写完盘**（文件句柄已关闭）再发 POST；服务端收到请求即刻读取，不等待、不轮询、不重试；写盘与发请求之间无确认机制，由文件系统保证 |
| 时间口径 | 响应头 `X-Process-Ms` = 服务端处理耗时（校验+装图+推理，不含网络）；客户端 elapsed = 端到端（含建连与传输）；对账时以 X-Process-Ms 排除网络因素 |
| 实测量级参考 | mock：全流程 2~20ms；yolo26n CPU：首次含模型加载约 3.1s（warmup），此后单图约百毫秒级 |

### 1.7 异常场景交互表现

| 场景 | 服务端行为 | 客户端观测 | 处理 |
|---|---|---|---|
| 服务未启动 / 地址端口错 | 不存在 | `URLError: connection failed`（client.py），HTTP 状态显示 N/A；C# 侧 `HttpRequestException` / `WebException` | 启动服务、核对地址与防火墙 |
| 客户端超时早于响应 | 推理照常完成，响应发往已断开的连接后被丢弃 | socket timeout / `TaskCanceledException` | 调大 timeout，或用 `X-Process-Ms` 与服务端日志比对定位耗时 |
| 请求路径错 | 404 + -1002，连接关闭 | HTTPError / `HttpStatusCode.NotFound`，**body 仍是 JSON 可解析** | 核对 URL 与 `--endpoint` |
| 请求体超限 / chunked | 不读 body，直接 200 + -1002，连接关闭 | `error_code=-1002`（invalid content length） | 压缩图片或调 `kMaxBodyBytes` |
| JSON / UTF-8 非法 | 200 + -1002 | `error_code=-1002` | 检查序列化（勿手动拼 JSON 字符串） |
| 图片数据问题 | 200 + -1004 | `error_code=-1004` | base64：核对编码与完整性；共享盘：核对服务端 `--share-root` 与落盘完成状态 |
| 推理异常 | 200 + -1001，日志 [ERROR] | `error_code=-1001` | 看服务端日志定位（模型缺失、显存不足等） |
| 服务端被 Ctrl+C | 监听关闭，在途请求响应丢失 | connection failed / 中断 | 客户端重试；产线建议由进程管理器守护 |

---

## 2. 目录结构

```
shimada-predict_service/
├── server.py        # 服务端程序（接口 + mock/YOLO 推理引擎）
├── client.py        # 测试客户端程序（生成图 / 两种请求 / 错误码用例）
├── requirements.txt # 依赖清单
├── README.md        # 本文档：完整代码逻辑
├── USAGE.md         # 使用手册
├── shared/          # 服务端共享盘根目录（启动时自动创建，--share-root 可改）
│   └── demo/        # 测试用相对目录（由客户端 gen-images 生成）
└── test_images/     # 客户端生成的本地测试图（gen-images 默认输出）
```

---

## 3. 总体架构

```
        ┌──────────────────────────────┐          ┌──────────────────────────────┐
        │  client.py                   │          │  server.py                   │
        │                              │          │                              │
        │  gen-images ──► 本地/共享盘   │  HTTP     │  ThreadingHTTPServer         │
        │  send-base64 ────────────────┼─POST────►│  └─ DefectHandler.do_POST    │
        │  send-path ──────────────────┼─JSON────►│      └─ handleInfer          │
        │  test-errors (13 用例) ──────┼──────────►│          ├─ validateRequest │
        │                              │          │          ├─ loadImages      │
        │  postJson (urllib, UTF-8)    │◄─JSON────┤          ├─ engine.predict   │
        │  printResult                 │ 200+JSON │          └─ processRequest  │
        └──────────────────────────────┘          │  MockEngine / YoloEngine     │
                                                  └──────────────────────────────┘
```

- 服务端线程模型：`ThreadingHTTPServer` 为每个连接开一个线程；**推理由全局锁 `g_inferLock` 串行化**（YOLO 模型非线程安全，且产线推理通常单卡）。
- 服务端全局状态：`g_config`（argparse 配置）、`g_engine`（推理引擎实例）、`g_inferLock`。

---

## 4. server.py 代码逻辑

### 4.1 常量与工具

| 名称 | 说明 |
|---|---|
| `kOk / kSystemError / kInvalidParam / kInvalidImageName / kImageNotFound` | 协议错误码常量（k 前缀） |
| `kErrorMsg` | 错误码 → 错误文案映射，与协议表一致 |
| `kMaxBodyBytes` = 256 MB | 请求体上限，防止恶意超大包 |
| `kImageNamePattern` | 文件名白名单正则：`^[\w\-. ]+\.(jpg|jpeg|png|bmp|tif|tiff)$`（忽略大小写）。字符集内不含 `/`、`\`，天然阻断文件名携带路径 |
| `kHandlerTimeoutSec` = 300 | Handler 的 socket 超时，防慢客户端占死线程 |
| `makeError(code, msg=None)` | 构造 `{"error_code":..., "error_msg":...}`；msg 缺省时查 `kErrorMsg` |
| `readImageInfo(bytes)` | 用 PIL 从内存图片读取 `(width, height)`；PIL 缺失、解码失败均返回 `None`（后续映射为 -1004） |

### 4.2 推理引擎

两者实现同一签名 `predict(imageBytes, fileName, imageWidth, imageHeight) -> list[dict]`，输出即协议 `results[]` 条目（不含 `file_name`，由调用方补充）。

**MockEngine（默认）**

- 用 `fileName` 的 MD5 前 8 位十六进制作随机种子（`random.Random(seed)`），同一文件名永远产生相同缺陷 → **结果可复现**，便于两端联调比对。
- 每张图生成 1~3 个随机矩形缺陷：类别从 `["ZM", "HS", "QJ", "WS"]` 中选（按业务自行替换），`score` ∈ [0.3, 0.99]，框尺寸受图片分辨率约束（宽高各不超过 1/6），`area = w*h`、`length = max(w, h)`、`volume = area`（mock 占位值）。
- `polygon / defect_raw` 返回空数组，与协议示例一致。

**YoloEngine（`--model` 指定时）**

- 构造时 `from ultralytics import YOLO` 并加载权重（延迟导入，mock 模式无需安装 ultralytics）。
- `predict` 中 `cv2.imdecode(np.frombuffer(bytes), IMREAD_COLOR)` 将请求体图片字节解码为 BGR ndarray，交给 `model.predict(img, device, conf, verbose=False)`。
- 字段映射：

| 协议字段 | 来源 |
|---|---|
| `code` | `result.names[cls_id]`（类别名；无名字时回退为类别 id 字符串） |
| `score` | `boxes.conf[i]`，round 4 位小数 |
| `box` | `boxes.xyxy[i]` 取整 → `[x1, y1, x2, y2]`，与协议"左顶点/右顶点"定义一致 |
| `area` | `(x2-x1) * (y2-y1)`（框面积；业务如需真实缺陷面积在此替换） |
| `length` | `max(框宽, 框高)` |
| `volume` | 与 area 相同的占位值 |
| `polygon / defect_raw` | 空数组（YOLO 检测头无多边形输出） |

- 无检出（`boxes` 为空）时返回空列表 → 响应 `results: []`，`error_code` 仍为 0（"没有缺陷"是正常业务结果，不是错误）。

### 4.3 请求处理流水线

```
do_POST
  └─ handleInfer                          # 兜底 try/except → -1001
       ├─ 路径校验     path != endpoint   → 404 + -1002
       ├─ 长度校验     Content-Length ≤0 或 >256MB → -1002
       ├─ JSON 解析   非法 UTF-8/JSON     → -1002
       └─ processRequest
            ├─ validateRequest            # 结构校验 → -1002 / -1003
            ├─ loadImages                 # 装图 → -1003 / -1004
            ├─ for 每张图: g_inferLock + engine.predict
            │    └─ 缺陷条目补 "file_name"
            └─ 组装 data → {"error_code":0, "error_msg":"OK", "data":{...}}
  响应头额外带 X-Process-Ms（服务端处理耗时，仅诊断用）
```

`validateRequest` 校验规则（按序短路）：

| 步骤 | 条件 | 错误码 |
|---|---|---|
| 1 | 请求体不是 JSON 对象 | -1002 |
| 2 | `job_id` / `sample_id` 缺失、非字符串或空白 | -1002 |
| 3 | `file_names` 非列表或为空列表；元素非字符串 | -1002 |
| 4 | 文件名空白或不过 `kImageNamePattern` 白名单 | -1003 |
| 5 | `images_base64` 与 `relative_dir` 同时存在 / 都不存在 | -1002 |
| 6 | base64 模式：`images_base64` 非列表、长度 ≠ `file_names`、元素非字符串或空串（空串视为数据不存在） | -1002 / -1004 |
| 7 | 共享盘模式：`relative_dir` 缺失、非字符串或空白 | -1002 |

`loadImages` 装图规则：

- **base64 模式**：`base64.b64decode` 逐张解码；`binascii.Error / ValueError / 空数据` → -1004。
- **共享盘模式**：`fullPath = abspath(join(shareRoot, relative_dir, fileName))`，用 `os.path.commonpath([shareRoot, fullPath])` 做**路径穿越防护**——`relative_dir` 含 `..` 逃出共享根时返回 -1003；文件不存在或读取 OSError → -1004。
- 装图后统一用 `readImageInfo` 试解码读尺寸；任何一张解不开（坏图、假扩展名）→ -1004。
- 成功返回 `[(fileName, bytes, width, height), ...]`。

### 4.4 HTTP 层细节

- `protocol_version = "HTTP/1.1"`：所有响应显式携带 `Content-Length`，支持 keep-alive 连接复用（RestSharp/HttpClient 默认复用连接，减少产线握手开销）。注意这两个类属性必须沿用 `http.server` 父类的蛇形命名（`server_version` / `protocol_version`），写成驼峰会**静默失效**、回落到 HTTP/1.0 且丢失自定义 Server 头。
- 两个"未读 body 就提前返回"的分支（路径 404、body 超限）额外发送 `Connection: close` 并断开连接：这两分支没有消费请求体，若复用连接，socket 缓冲区里的残留 body 会污染下一个请求。
- `send_response / send_header` 由 `sendJson` 统一处理：`Content-Type: application/json; charset=utf-8`，可选附加头（当前为 `X-Process-Ms` 与 `Connection: close`）。
- `do_GET` 统一返回 404 + -1002（提示 use POST），方便排查用浏览器误访问的情况。
- 访问日志走 `log_message`（BaseHTTPRequestHandler 的标准访问行），业务日志 `[REQ] job_id=... files=N mode=... -> error_code=... elapsed=...ms` 与 `[ERROR]` 均打印到 stdout，全部英文。

### 4.5 启动流程（`main`）

1. `parseArgs` 解析 `--host/--port/--endpoint/--model/--device/--conf/--share-root`。
2. 有 `--model` → 构造 `YoloEngine`（此时才 import ultralytics 并加载权重）；否则 `MockEngine`。
3. `os.makedirs(shareRoot, exist_ok=True)` 保证共享根存在。
4. `ThreadingHTTPServer((host, port), DefectHandler)` 启动，打印 endpoint / mode / share root 三行配置。
5. `serve_forever()`；`Ctrl+C` 优雅退出（`server_close()`）。

---

## 5. client.py 代码逻辑

### 5.1 常量

与 server.py 相同的一份错误码常量（kOk/kSystemError/…），供 `test-errors` 断言期望值。

### 5.2 传输层 `postJson(url, payload, timeoutSec)`

1. `json.dumps(payload, ensure_ascii=False).encode("utf-8")` — 中文内容不转义、按协议用 UTF-8。
2. `urllib.request.Request(..., method="POST", headers={"Content-Type": "application/json"})` — 标准库实现，无第三方依赖。
3. 三类结果：
   - 正常 → `(status, 解析后的 dict, 耗时ms, None)`
   - `HTTPError`（如 404）→ 读取错误响应体并照常解析（服务端 404 也返回 JSON）
   - `URLError`（连不上、超时）→ `(None, None, 耗时ms, "connection failed: ...")`
4. 响应体非合法 JSON → `(status, None, 耗时ms, "response is not valid json")`。

### 5.3 测试图生成 `makeTestImage(width, height)`

PIL 在深灰底上撒 150 个随机灰度小方块（噪声纹理）+ 一个红色描边矩形（模拟缺陷目标），JPEG quality=90 编码后返回字节。用于 `gen-images` 落盘，以及 `send-base64`/`test-errors` 在内存中直接构造请求。

### 5.4 请求构造

- `buildBase64Payload`：读本地文件 → `base64.b64encode → ascii 字符串`，组装协议全部字段，`ext_params` 携带 `client/sent_at` 演示扩展参数用法。
- `buildPathPayload`：`file_names` 取 basename，`relative_dir` 由 `--dir` 传入（相对服务端共享盘根）。
- `defaultJobId`：`J + yyyyMMddHHmmss + 3位随机数`，模拟产线 job_id 形态；`--job-id` 可覆盖。

### 5.5 四个子命令

| 子命令 | 逻辑 |
|---|---|
| `gen-images` | `makeTestImage` × count 张，按协议命名风格 `{sampleId}_C01_{poseId}_L{i}_{时间戳}.jpg` 写入 `--out` 目录并打印绝对路径 |
| `send-base64` | 有 `--images` 读文件；无则内存生成一张 1920×1440 测试图（开箱即测）。构造 payload → `postJson` → `printResult` |
| `send-path` | 用 `--dir` + `--files` 构造共享盘 payload → `postJson` → `printResult`。要求文件已存在于服务端 `--share-root` 下（先 gen-images 到共享目录，见 USAGE §5） |
| `test-errors` | 13 个用例自动跑（见下表），每例打印 `[PASS]/[FAIL] expected=... actual=...`，汇总后以进程退出码 0/1 表示全部通过与否 |

`test-errors` 用例表：

| # | 用例 | 构造方式 | 期望 error_code |
|---|---|---|---|
| 1 | valid base64 request | 完整合法请求 | 0 |
| 2 | missing job_id | 删 `job_id` | -1002 |
| 3 | missing sample_id | 删 `sample_id` | -1002 |
| 4 | file_names/images_base64 length mismatch | 两名一图 | -1002 |
| 5 | empty file_names | 空数组 | -1002 |
| 6 | both images_base64 and relative_dir | 两种方式并存 | -1002 |
| 7 | neither images_base64 nor relative_dir | 都不带 | -1002 |
| 8 | empty file name | `" "`（空白名） | -1003 |
| 9 | illegal file name | `"a|b<img>.jpg"` | -1003 |
| 10 | unsupported file extension | `"ghost.exe"` | -1003 |
| 11 | corrupted base64 image | `"@@not-base64@@"` | -1004 |
| 12 | empty base64 image | `""` | -1004 |
| 13 | image not found on share | `relative_dir="no_such_dir/"` + `"ghost.jpg"`（无论服务端共享根指向哪，路径必不存在） | -1004 |

### 5.6 输出 `printResult`

逐行打印 HTTP 状态、耗时、`error_code/error_msg`；成功时再打印 `data` 摘要（job_id/sample_id/pose_id/分辨率/缺陷数）以及每个缺陷一行的 `file/code/score/area/length/box`。连接失败或响应非 JSON 时打印 `error` 行。

---

## 6. 扩展与对接真实业务

| 需求 | 改动点 |
|---|---|
| 真实缺陷面积/体积 | `YoloEngine.predict` 中 area/length/volume 当前是框的占位值；接入分割模型（mask 像素数）或业务测量逻辑替换 |
| 多边形输出 | 协议 `polygon` 字段；在引擎里把分割 mask 轮廓坐标填入即可，服务端其余逻辑不动 |
| mock 类别 | `MockEngine.kDefectCodes` |
| 并发推理 | 当前 `g_inferLock` 全局串行；多卡部署可为每卡建独立引擎+锁，或换推理批处理队列 |
| 接口路径/端口 | 全部是服务端启动参数，无需改代码 |
| HTTPS / 鉴权 | 协议声明"HTTP 头部无特殊校验逻辑"，如需在标准库 `ThreadingHTTPServer` 外再套 Nginx 反代加 TLS 即可，服务端不用改 |
