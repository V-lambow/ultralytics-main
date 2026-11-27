import base64
import binascii
import hashlib
import io
import json
import math
import os
import random
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

kOk = 0
kSystemError = -1001
kInvalidParam = -1002
kInvalidImageName = -1003
kImageNotFound = -1004

kErrorMsg = {
    kOk: "OK",
    kSystemError: "系统错误：未知错误",
    kInvalidParam: "参数错误：参数不合法",
    kInvalidImageName: "参数错误：图片名称错误",
    kImageNotFound: "参数错误：图片数据不存在",
}

kMaxBodyBytes = 256 * 1024 * 1024
kMaxImageBytes = 128 * 1024 * 1024
kMaxImagePixels = 30_000_000
kMaxInflight = 8
kBusyMsg = "系统繁忙：并发请求过多，请稍后重试"
kImageNamePattern = re.compile(r"^[\w\-. ]+\.(jpg|jpeg|png|bmp|tif|tiff)$", re.IGNORECASE)
kHandlerTimeoutSec = 300
kConfigFileName = "config.json"
kModelAliasPattern = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
kWorkpiecePattern = re.compile(r"^\S{1,64}$")
kDefaultModelConf = 0.25
kLogDirName = "log"
kLogRetention = 30
kLogFilePattern = re.compile(r"^\d{8}\.log$")
kCountErrorCode = "count_error"
kIntervalErrorCode = "interval_error"
kCheckKeys = {
    kCountErrorCode: ("label_name", "proper_count"),
    kIntervalErrorCode: ("label_name", "proper_interval", "edgedistORcenterdist"),
}

kDefaultConfig = {
    "host": "127.0.0.1",
    "port": 9000,
    "endpoint": "/image_defect",
    "device": None,
    "models": {"default": {"path": None}},
    "default_model": None,
    "workpieces": {},
    "share_root": "./shared",
    kCountErrorCode: None,
    kIntervalErrorCode: None,
}

g_config = None
g_models = {}
g_defaultModel = None
g_inferLock = threading.Lock()
g_inferSlots = threading.BoundedSemaphore(kMaxInflight)
g_logLock = threading.Lock()
g_logDay = None
g_logErrorReported = False


def makeError(errorCode, msg=None):
    return {
        "error_code": errorCode,
        "error_msg": kErrorMsg.get(errorCode, "unknown error") if msg is None else msg,
    }


def logDir():
    return os.path.join(os.getcwd(), kLogDirName)


def pruneLogs():
    directory = logDir()
    try:
        names = sorted(n for n in os.listdir(directory) if kLogFilePattern.match(n))
    except OSError:
        return
    for name in names[: max(0, len(names) - kLogRetention)]:
        try:
            os.remove(os.path.join(directory, name))
        except OSError:
            pass


def logLine(message):
    """Append one timestamped line to log/YYYYMMDD.log, keep at most kLogRetention day files, and echo to stdout."""
    global g_logDay, g_logErrorReported
    day = time.strftime("%Y%m%d")
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    with g_logLock:
        try:
            print(line, flush=True)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode("ascii"), flush=True)
        try:
            os.makedirs(logDir(), exist_ok=True)
            with open(os.path.join(logDir(), f"{day}.log"), "a", encoding="utf-8", newline="\n") as f:
                f.write(line + "\n")
        except OSError as exc:
            if not g_logErrorReported:
                g_logErrorReported = True
                print(f"[WARN] log file disabled: {exc}", flush=True)
            return
        if day != g_logDay:
            g_logDay = day
            pruneLogs()


def readImageInfo(imageBytes):
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(io.BytesIO(imageBytes)) as im:
            return int(im.size[0]), int(im.size[1])
    except Exception:
        return None


def decodeImage(imageBytes):
    """Decode image bytes once to (bgr, width, height); bgr is None when only PIL is available."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        cv2 = None
    if cv2 is not None:
        img = cv2.imdecode(np.frombuffer(imageBytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            height, width = img.shape[:2]
            return img, int(width), int(height)
    size = readImageInfo(imageBytes)
    return None if size is None else (None, size[0], size[1])


class MockEngine:
    kDefectCodes = ["ZM", "HS", "QJ", "WS"]

    def __init__(self, alias):
        self.alias = alias

    def predict(self, pixels, fileName, imageWidth, imageHeight):
        seed = int(hashlib.md5(fileName.encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed)
        defects = []
        for _ in range(rng.randint(1, 3)):
            boxWidth = rng.randint(80, max(81, imageWidth // 6))
            boxHeight = rng.randint(80, max(81, imageHeight // 6))
            x1 = rng.randint(0, max(0, imageWidth - boxWidth))
            y1 = rng.randint(0, max(0, imageHeight - boxHeight))
            area = boxWidth * boxHeight
            defects.append({
                "code": rng.choice(self.kDefectCodes),
                "score": round(rng.uniform(0.3, 0.99), 4),
                "area": area,
                "length": max(boxWidth, boxHeight),
                "volume": area,
                "box": [x1, y1, x1 + boxWidth, y1 + boxHeight],
                "polygon": [],
                "defect_raw": [],
            })
        return defects


class YoloEngine:
    def __init__(self, alias, modelPath, device, conf):
        from ultralytics import YOLO
        self.alias = alias
        self.model = YOLO(modelPath)
        self.device = device
        self.conf = conf

    def predict(self, pixels, fileName, imageWidth, imageHeight):
        result = self.model.predict(pixels, device=self.device, conf=self.conf, verbose=False)[0]
        defects = []
        boxes = result.boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = (int(v) for v in boxes.xyxy[i].tolist())
            boxWidth = max(0, x2 - x1)
            boxHeight = max(0, y2 - y1)
            clsId = int(boxes.cls[i])
            defects.append({
                "code": result.names.get(clsId, str(clsId)),
                "score": round(float(boxes.conf[i]), 4),
                "area": boxWidth * boxHeight,
                "length": max(boxWidth, boxHeight),
                "volume": boxWidth * boxHeight,
                "box": [x1, y1, x2, y2],
                "polygon": [],
                "defect_raw": [],
            })
        return defects

    def warmup(self):
        """Run one dummy inference so the first real request does not pay lazy init/compile cost."""
        import numpy as np
        self.predict(np.zeros((640, 640, 3), dtype=np.uint8), "warmup.jpg", 640, 640)


def validateRequest(req):
    if not isinstance(req, dict):
        return kInvalidParam
    for key in ("job_id", "sample_id"):
        value = req.get(key)
        if not isinstance(value, str) or not value.strip():
            return kInvalidParam
    fileNames = req.get("file_names")
    if not isinstance(fileNames, list) or len(fileNames) == 0:
        return kInvalidParam
    if any(not isinstance(n, str) for n in fileNames):
        return kInvalidParam
    if any(not n.strip() or not kImageNamePattern.match(n) for n in fileNames):
        return kInvalidImageName
    hasBase64 = "images_base64" in req
    hasDir = "relative_dir" in req
    if hasBase64 and hasDir:
        return kInvalidParam
    if not hasBase64 and not hasDir:
        return kInvalidParam
    if hasBase64:
        images = req.get("images_base64")
        if not isinstance(images, list) or len(images) != len(fileNames):
            return kInvalidParam
        if any(not isinstance(b, str) or not b for b in images):
            return kImageNotFound
    else:
        relDir = req.get("relative_dir")
        if not isinstance(relDir, str) or not relDir.strip():
            return kInvalidParam
    return kOk


def loadImages(req):
    fileNames = req["file_names"]
    loaded = []
    if "images_base64" in req:
        for name, b64Data in zip(fileNames, req["images_base64"]):
            try:
                data = base64.b64decode(b64Data, validate=True)
            except (binascii.Error, ValueError):
                logLine(f"[IMG] [错误] [ng] file={name} reason=invalid base64")
                return None, kImageNotFound
            if not data:
                logLine(f"[IMG] [错误] [ng] file={name} reason=empty image data")
                return None, kImageNotFound
            if len(data) > kMaxImageBytes:
                logLine(f"[IMG] [错误] [ng] file={name} bytes={len(data)} limit={kMaxImageBytes}")
                return None, kImageNotFound
            loaded.append((name, data))
    else:
        shareRoot = os.path.abspath(g_config.share_root)
        for name in fileNames:
            fullPath = os.path.abspath(os.path.join(shareRoot, req["relative_dir"], name))
            try:
                if os.path.commonpath([shareRoot, fullPath]) != shareRoot:
                    return None, kInvalidImageName
            except ValueError:
                return None, kInvalidImageName
            if not os.path.isfile(fullPath):
                return None, kImageNotFound
            try:
                size = os.path.getsize(fullPath)
                if size > kMaxImageBytes:
                    logLine(f"[IMG] [错误] [ng] file={name} bytes={size} limit={kMaxImageBytes}")
                    return None, kImageNotFound
                with open(fullPath, "rb") as f:
                    loaded.append((name, f.read()))
            except OSError:
                return None, kImageNotFound
    images = []
    for name, data in loaded:
        decoded = decodeImage(data)
        if decoded is None:
            logLine(f"[IMG] [错误] [ng] file={name} reason=decode failed")
            return None, kImageNotFound
        pixels, width, height = decoded
        if width * height > kMaxImagePixels:
            logLine(f"[IMG] [错误] [ng] file={name} pixels={width * height} limit={kMaxImagePixels}")
            return None, kImageNotFound
        images.append((name, pixels, width, height))
    return images, kOk


def parseWorkpiece(fileName):
    """Return the workpiece code parsed from an image file name, or None when no rule matches.

    TODO: 解析规则与型号清单均未定，先留空返回 None。
    例：'S00001_C01_P01_L0_20261119103015.jpg' 中按约定取某一段作为型号。
    规则确定后只改这个函数，processRequest / 配对表 / 日志都不用动。
    """
    return None


def workpieceModels(workpiece):
    """Return the engine handles configured for a workpiece code, or None when the code is not in the table."""
    aliases = g_config.workpieces.get(workpiece) if isinstance(workpiece, str) else None
    return None if aliases is None else [g_models[alias] for alias in aliases]


def resolveHandles(fileNames):
    """Return (workpiece, engine handles); fall back to default_model while parseWorkpiece returns None."""
    workpiece = parseWorkpiece(fileNames[0])
    if workpiece is None:
        return None, [g_models[g_defaultModel]]
    return workpiece, workpieceModels(workpiece)


def selectionSummary(fileNames):
    """Format the workpiece / model selection for the request log."""
    if not isinstance(fileNames, list) or not fileNames or not isinstance(fileNames[0], str):
        return "workpiece=- models=-"
    workpiece, handles = resolveHandles(fileNames)
    aliases = ",".join(handle.alias for handle in handles) if handles else "-"
    return f"workpiece={workpiece or '-'} models={aliases}"


def defectCounts(results):
    counts = {}
    for item in results:
        code = item.get("code", "?")
        counts[code] = counts.get(code, 0) + 1
    return ",".join(f"{code}:{count}" for code, count in counts.items()) or "-"


def formatFileNames(fileNames):
    names = [str(name) for name in fileNames] if isinstance(fileNames, list) else []
    return ",".join(names) or "-"


def boxInterval(boxA, boxB, useEdgeDist):
    """Return the edge distance (useEdgeDist) or center distance between two [x1, y1, x2, y2] boxes, in pixels."""
    if useEdgeDist:
        dx = max(0, max(boxA[0], boxB[0]) - min(boxA[2], boxB[2]))
        dy = max(0, max(boxA[1], boxB[1]) - min(boxA[3], boxB[3]))
    else:
        dx = (boxA[0] + boxA[2] - boxB[0] - boxB[2]) / 2
        dy = (boxA[1] + boxA[3] - boxB[1] - boxB[3]) / 2
    return math.hypot(dx, dy)


def countErrorDefect():
    """Rule-based count defect: no target region, so every geometry field is empty/zero."""
    return {
        "code": kCountErrorCode,
        "score": 1.0,
        "area": 0,
        "length": 0,
        "volume": 0,
        "box": [],
        "polygon": [],
        "defect_raw": [],
    }


def intervalErrorDefect(targetA, targetB):
    """Rule-based spacing defect: box is the pair's outer box, defect_raw carries both target boxes."""
    x1 = min(targetA["box"][0], targetB["box"][0])
    y1 = min(targetA["box"][1], targetB["box"][1])
    x2 = max(targetA["box"][2], targetB["box"][2])
    y2 = max(targetA["box"][3], targetB["box"][3])
    area = (x2 - x1) * (y2 - y1)
    return {
        "code": kIntervalErrorCode,
        "score": 1.0,
        "area": area,
        "length": max(x2 - x1, y2 - y1),
        "volume": area,
        "box": [x1, y1, x2, y2],
        "polygon": [],
        "defect_raw": [
            {"code": target["code"], "score": target["score"], "box": list(target["box"])}
            for target in (targetA, targetB)
        ],
    }


def applyChecks(defects, fileName):
    """Append count_error / interval_error entries based on one image's merged model output."""
    countRule = g_config.count_error
    if countRule is not None:
        count = sum(1 for d in defects if d.get("code") == countRule["label_name"])
        if not countRule["low"] <= count <= countRule["high"]:
            logLine(
                f"[CHECK] [错误] [ng] file={fileName} count_error label={countRule['label_name']} "
                f"count={count} proper=[{countRule['low']},{countRule['high']}]"
            )
            defects.append(countErrorDefect())
    intervalRule = g_config.interval_error
    if intervalRule is not None:
        targets = [d for d in defects if d.get("code") == intervalRule["label_name"]]
        for i, targetA in enumerate(targets):
            for targetB in targets[i + 1 :]:
                interval = boxInterval(targetA["box"], targetB["box"], intervalRule["edgedistORcenterdist"])
                if not intervalRule["low"] <= interval <= intervalRule["high"]:
                    logLine(
                        f"[CHECK] [错误] [ng] file={fileName} interval_error label={intervalRule['label_name']} "
                        f"interval={round(interval, 2)} proper=[{intervalRule['low']},{intervalRule['high']}] "
                        f"edgedistORcenterdist={intervalRule['edgedistORcenterdist']}"
                    )
                    defects.append(intervalErrorDefect(targetA, targetB))
    return defects


def logResponse(resp, req, elapsedMs):
    jobInfo = req.get("job_id", "?") if isinstance(req, dict) else "?"
    sampleInfo = req.get("sample_id", "?") if isinstance(req, dict) else "?"
    poseInfo = req.get("pose_id", "") if isinstance(req, dict) else ""
    fileNames = req.get("file_names") if isinstance(req, dict) else None
    common = (
        f"job_id={jobInfo} sample_id={sampleInfo} pose_id={poseInfo} files={formatFileNames(fileNames)} "
        f"{selectionSummary(fileNames)}"
    )
    if resp["error_code"] == kOk:
        results = (resp.get("data") or {}).get("results") or []
        logLine(
            f"[RESP] [成功] [ok] {common} defects={defectCounts(results)} total={len(results)} elapsed={elapsedMs}ms"
        )
    else:
        logLine(
            f"[RESP] [错误] [ng] {common} error_code={resp['error_code']} error_msg={resp['error_msg']} "
            f"elapsed={elapsedMs}ms"
        )


def processRequest(req):
    startMs = time.perf_counter()
    errorCode = validateRequest(req)
    if errorCode != kOk:
        return makeError(errorCode), 0.0
    workpiece, handles = resolveHandles(req["file_names"])
    if handles is None:
        return makeError(kInvalidParam, f"unknown workpiece: {workpiece}"), 0.0
    images, errorCode = loadImages(req)
    if errorCode != kOk:
        return makeError(errorCode), 0.0
    allResults = []
    firstWidth, firstHeight = 0, 0
    for name, pixels, width, height in images:
        imageResults = []
        for handle in handles:
            with g_inferLock:
                imageResults.extend(handle.predict(pixels, name, width, height))
        applyChecks(imageResults, name)
        for d in imageResults:
            d["file_name"] = name
        allResults.extend(imageResults)
        if firstWidth == 0:
            firstWidth, firstHeight = width, height
    data = {
        "job_id": req["job_id"],
        "sample_id": req["sample_id"],
        "pose_id": req.get("pose_id", ""),
        "file_names": req["file_names"],
        "image_width": firstWidth,
        "image_height": firstHeight,
        "grayscale": 0,
        "results": allResults,
        "alg_raw": [],
    }
    elapsedMs = round((time.perf_counter() - startMs) * 1000, 1)
    return {"error_code": kOk, "error_msg": "OK", "data": data}, elapsedMs


class DefectHandler(BaseHTTPRequestHandler):
    server_version = "ShimadaPredictService/1.0"
    protocol_version = "HTTP/1.1"
    timeout = kHandlerTimeoutSec

    def log_message(self, fmt, *args):
        """Drop BaseHTTPRequestHandler access lines; [REQ]/[RESP] already cover every request."""

    def do_GET(self):
        logLine(f"[RESP] [错误] [ng] GET {self.path} rejected, use POST")
        self.sendJson(makeError(kInvalidParam, "use POST with a json body"), httpStatus=404)

    def do_POST(self):
        try:
            self.handleInfer()
        except Exception as exc:
            logLine(f"[RESP] [错误] [ng] unhandled exception: {exc}")
            try:
                self.sendJson(makeError(kSystemError))
            except Exception:
                pass

    def handleInfer(self):
        pathOnly = self.path.split("?")[0].rstrip("/")
        if pathOnly != g_config.endpoint.rstrip("/"):
            logLine(f"[RESP] [错误] [ng] unknown endpoint={self.path}")
            self.sendJson(makeError(kInvalidParam, f"unknown endpoint: {self.path}"),
                          httpStatus=404, extraHeaders={"Connection": "close"})
            return
        contentLength = int(self.headers.get("Content-Length") or 0)
        if contentLength <= 0 or contentLength > kMaxBodyBytes:
            logLine(f"[RESP] [错误] [ng] invalid content length={contentLength}")
            self.sendJson(makeError(kInvalidParam, f"invalid content length: {contentLength}"),
                          extraHeaders={"Connection": "close"})
            return
        rawBody = self.rfile.read(contentLength)
        try:
            req = json.loads(rawBody.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            logLine("[RESP] [错误] [ng] request body is not valid utf-8 json")
            self.sendJson(makeError(kInvalidParam, "request body is not valid utf-8 json"))
            return
        mode = "base64" if isinstance(req, dict) and "images_base64" in req else "share_dir"
        jobInfo = req.get("job_id", "?") if isinstance(req, dict) else "?"
        sampleInfo = req.get("sample_id", "?") if isinstance(req, dict) else "?"
        poseInfo = req.get("pose_id", "") if isinstance(req, dict) else ""
        fileNames = req.get("file_names") if isinstance(req, dict) else None
        logLine(
            f"[REQ] received | job_id={jobInfo} sample_id={sampleInfo} pose_id={poseInfo} "
            f"files={formatFileNames(fileNames)} mode={mode}"
        )
        if not g_inferSlots.acquire(blocking=False):
            resp = makeError(kSystemError, kBusyMsg)
            logResponse(resp, req, 0.0)
            self.sendJson(resp, extraHeaders={"X-Process-Ms": "0.0"})
            return
        try:
            resp, elapsedMs = processRequest(req)
        finally:
            g_inferSlots.release()
        logResponse(resp, req, elapsedMs)
        self.sendJson(resp, extraHeaders={"X-Process-Ms": str(elapsedMs)})

    def sendJson(self, payload, httpStatus=200, extraHeaders=None):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(httpStatus)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extraHeaders or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)


class Config:
    def __init__(
        self, host, port, endpoint, device, models, default_model, workpieces, share_root, count_error, interval_error
    ):
        self.host = host
        self.port = port
        self.endpoint = endpoint
        self.device = device
        self.models = models
        self.default_model = default_model
        self.workpieces = workpieces
        self.share_root = share_root
        self.count_error = count_error
        self.interval_error = interval_error


def loadModels(path, models):
    if not isinstance(models, dict) or not models:
        raise SystemExit(f"config file {path}: models must be a non-empty json object of alias -> {{path, conf}}")
    loaded = {}
    for alias, entry in models.items():
        if not kModelAliasPattern.match(alias):
            raise SystemExit(f"config file {path}: model alias {alias!r} must match {kModelAliasPattern.pattern}")
        if not isinstance(entry, dict):
            raise SystemExit(f"config file {path}: model {alias!r} must be a json object")
        unknown = sorted(set(entry) - {"path", "conf"})
        if unknown:
            raise SystemExit(
                f"config file {path}: model {alias!r} has unknown keys {unknown}, allowed: ['conf', 'path']"
            )
        modelPath = entry.get("path")
        if modelPath is not None and (not isinstance(modelPath, str) or not modelPath.strip()):
            raise SystemExit(f"config file {path}: model {alias!r} path must be null or a non-empty string")
        conf = entry.get("conf", kDefaultModelConf)
        if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not 0.0 <= conf <= 1.0:
            raise SystemExit(f"config file {path}: model {alias!r} conf must be a number in 0..1")
        loaded[alias] = {"path": modelPath, "conf": conf}
    return loaded


def loadWorkpieces(path, workpieces, models):
    if not isinstance(workpieces, dict):
        raise SystemExit(f"config file {path}: workpieces must be a json object of workpiece -> [model alias]")
    loaded = {}
    for workpiece, aliases in workpieces.items():
        if not kWorkpiecePattern.match(workpiece):
            raise SystemExit(
                f"config file {path}: workpiece {workpiece!r} must match {kWorkpiecePattern.pattern}"
            )
        if not isinstance(aliases, list) or not aliases:
            raise SystemExit(
                f"config file {path}: workpiece {workpiece!r} must map to a non-empty json array of model aliases"
            )
        for alias in aliases:
            if not isinstance(alias, str) or alias not in models:
                raise SystemExit(
                    f"config file {path}: workpiece {workpiece!r} references unknown model alias {alias!r}, "
                    f"known: {sorted(models)}"
                )
        if len(set(aliases)) != len(aliases):
            raise SystemExit(f"config file {path}: workpiece {workpiece!r} lists the same model alias more than once")
        loaded[workpiece] = list(aliases)
    return loaded


def loadCheckRule(path, key, rule):
    """Validate a count/interval check rule, returning {label_name, low, high[, edgedistORcenterdist]} or None."""
    if rule is None:
        return None
    if not isinstance(rule, dict):
        raise SystemExit(f"config file {path}: {key} must be null or a json object")
    allowed = list(kCheckKeys[key])
    unknown = sorted(set(rule) - set(allowed))
    if unknown:
        raise SystemExit(f"config file {path}: {key} has unknown keys {unknown}, allowed: {allowed}")
    label = rule.get("label_name")
    if not isinstance(label, str) or not label.strip():
        raise SystemExit(f"config file {path}: {key}.label_name must be a non-empty string")
    rangeKey = allowed[1]
    bounds = rule.get(rangeKey)
    if (
        not isinstance(bounds, list)
        or len(bounds) != 2
        or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in bounds)
    ):
        raise SystemExit(f"config file {path}: {key}.{rangeKey} must be [min, max] with two numbers")
    low, high = bounds
    if low > high:
        raise SystemExit(f"config file {path}: {key}.{rangeKey} min must not exceed max")
    loaded = {"label_name": label, "low": low, "high": high}
    if key == kIntervalErrorCode:
        edgeDist = rule.get("edgedistORcenterdist")
        if not isinstance(edgeDist, bool):
            raise SystemExit(
                f"config file {path}: {key}.edgedistORcenterdist must be true (edge distance) "
                "or false (center distance)"
            )
        loaded["edgedistORcenterdist"] = edgeDist
    return loaded


def loadConfig():
    path = os.path.join(os.getcwd(), kConfigFileName)
    try:
        with open(path, "rb") as f:
            raw = json.loads(f.read().decode("utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"config file not found: {path}")
    except OSError as exc:
        raise SystemExit(f"cannot read config file {path}: {exc}")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"config file {path} is not valid utf-8 json: {exc}")
    if not isinstance(raw, dict):
        raise SystemExit(f"config file {path} must be a json object")
    unknown = sorted(set(raw) - set(kDefaultConfig))
    if unknown:
        raise SystemExit(f"config file {path} has unknown keys {unknown}, allowed: {sorted(kDefaultConfig)}")
    cfg = {**kDefaultConfig, **raw}
    host = cfg["host"]
    if not isinstance(host, str) or not host.strip():
        raise SystemExit(f"config file {path}: host must be a non-empty string")
    port = cfg["port"]
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise SystemExit(f"config file {path}: port must be an integer in 1..65535")
    endpoint = cfg["endpoint"]
    if not isinstance(endpoint, str) or not endpoint.startswith("/") or any(c.isspace() for c in endpoint):
        raise SystemExit(f"config file {path}: endpoint must start with '/' and contain no whitespace")
    device = cfg["device"]
    if device is not None and (isinstance(device, bool) or not isinstance(device, (str, int))):
        raise SystemExit(f"config file {path}: device must be null, a string or an integer")
    models = loadModels(path, cfg["models"])
    defaultModel = cfg["default_model"]
    if defaultModel is None:
        if len(models) != 1:
            raise SystemExit(f"config file {path}: default_model is required when models has more than one entry")
        defaultModel = next(iter(models))
    elif not isinstance(defaultModel, str) or defaultModel not in models:
        raise SystemExit(f"config file {path}: default_model {defaultModel!r} is not one of {sorted(models)}")
    workpieces = loadWorkpieces(path, cfg["workpieces"], models)
    shareRoot = cfg["share_root"]
    if not isinstance(shareRoot, str) or not shareRoot.strip():
        raise SystemExit(f"config file {path}: share_root must be a non-empty string")
    countError = loadCheckRule(path, kCountErrorCode, cfg[kCountErrorCode])
    intervalError = loadCheckRule(path, kIntervalErrorCode, cfg[kIntervalErrorCode])
    return Config(
        host, port, endpoint, device, models, defaultModel, workpieces, shareRoot, countError, intervalError
    )


def buildModels(config):
    models = {}
    for alias, entry in config.models.items():
        kind = "mock" if entry["path"] is None else "yolo"
        try:
            if entry["path"] is None:
                models[alias] = MockEngine(alias)
            else:
                models[alias] = YoloEngine(alias, entry["path"], config.device, entry["conf"])
            logLine(f"[MODEL] loaded | alias={alias} path={entry['path']} engine={kind} conf={entry['conf']}")
            if entry["path"] is not None:
                startMs = time.perf_counter()
                models[alias].warmup()
                logLine(
                    f"[MODEL] warmup | alias={alias} elapsed={round((time.perf_counter() - startMs) * 1000, 1)}ms"
                )
        except Exception as exc:
            logLine(f"[MODEL] [错误] [ng] alias={alias} path={entry['path']} error={exc}")
            raise
    return models


def main():
    global g_config, g_models, g_defaultModel
    try:
        g_config = loadConfig()
    except SystemExit as exc:
        logLine(f"[SERVICE] [错误] [ng] {exc}")
        raise
    logLine(
        f"[SERVICE] starting | config={os.path.join(os.getcwd(), kConfigFileName)} "
        f"endpoint=http://{g_config.host}:{g_config.port}{g_config.endpoint} "
        f"default_model={g_config.default_model} device={g_config.device} "
        f"share_root={os.path.abspath(g_config.share_root)}"
    )
    g_defaultModel = g_config.default_model
    g_models = buildModels(g_config)
    for workpiece, aliases in g_config.workpieces.items():
        logLine(f"[WORKPIECE] {workpiece} -> {', '.join(aliases)}")
    for key, rule in ((kCountErrorCode, g_config.count_error), (kIntervalErrorCode, g_config.interval_error)):
        if rule is not None:
            extra = ""
            if "edgedistORcenterdist" in rule:
                extra = f" edgedistORcenterdist={rule['edgedistORcenterdist']}"
            logLine(f"[CHECK] rule | {key} label={rule['label_name']} proper=[{rule['low']},{rule['high']}]{extra}")
    os.makedirs(g_config.share_root, exist_ok=True)
    httpd = ThreadingHTTPServer((g_config.host, g_config.port), DefectHandler)
    logLine(f"[SERVICE] started | pid={os.getpid()} log={logDir()}")
    print("Press Ctrl+C to stop.", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logLine("[SERVICE] stopped")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
