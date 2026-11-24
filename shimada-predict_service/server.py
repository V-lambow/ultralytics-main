import base64
import binascii
import hashlib
import io
import json
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
kImageNamePattern = re.compile(r"^[\w\-. ]+\.(jpg|jpeg|png|bmp|tif|tiff)$", re.IGNORECASE)
kHandlerTimeoutSec = 300
kConfigFileName = "config.json"
kModelAliasPattern = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
kWorkpiecePattern = re.compile(r"^\S{1,64}$")
kDefaultModelConf = 0.25
kLogDirName = "log"
kLogRetention = 30
kLogFilePattern = re.compile(r"^\d{8}\.log$")

kDefaultConfig = {
    "host": "127.0.0.1",
    "port": 9000,
    "endpoint": "/image_defect",
    "device": None,
    "models": {"default": {"path": None}},
    "default_model": None,
    "workpieces": {},
    "share_root": "./shared",
}

g_config = None
g_models = {}
g_defaultModel = None
g_inferLock = threading.Lock()
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


class MockEngine:
    kDefectCodes = ["ZM", "HS", "QJ", "WS"]

    def __init__(self, alias):
        self.alias = alias

    def predict(self, imageBytes, fileName, imageWidth, imageHeight):
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

    def predict(self, imageBytes, fileName, imageWidth, imageHeight):
        import cv2
        import numpy as np
        img = cv2.imdecode(np.frombuffer(imageBytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return []
        result = self.model.predict(img, device=self.device, conf=self.conf, verbose=False)[0]
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
                data = base64.b64decode(b64Data)
            except (binascii.Error, ValueError):
                return None, kImageNotFound
            if not data:
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
                with open(fullPath, "rb") as f:
                    loaded.append((name, f.read()))
            except OSError:
                return None, kImageNotFound
    images = []
    for name, data in loaded:
        size = readImageInfo(data)
        if size is None:
            return None, kImageNotFound
        images.append((name, data, size[0], size[1]))
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
    for name, data, width, height in images:
        for handle in handles:
            with g_inferLock:
                defects = handle.predict(data, name, width, height)
            for d in defects:
                d["file_name"] = name
            allResults.extend(defects)
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
        resp, elapsedMs = processRequest(req)
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
    def __init__(self, host, port, endpoint, device, models, default_model, workpieces, share_root):
        self.host = host
        self.port = port
        self.endpoint = endpoint
        self.device = device
        self.models = models
        self.default_model = default_model
        self.workpieces = workpieces
        self.share_root = share_root


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
    return Config(host, port, endpoint, device, models, defaultModel, workpieces, shareRoot)


def buildModels(config):
    models = {}
    for alias, entry in config.models.items():
        kind = "mock" if entry["path"] is None else "yolo"
        try:
            if entry["path"] is None:
                models[alias] = MockEngine(alias)
            else:
                models[alias] = YoloEngine(alias, entry["path"], config.device, entry["conf"])
        except Exception as exc:
            logLine(f"[MODEL] [错误] [ng] alias={alias} path={entry['path']} error={exc}")
            raise
        logLine(f"[MODEL] loaded | alias={alias} path={entry['path']} engine={kind} conf={entry['conf']}")
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
