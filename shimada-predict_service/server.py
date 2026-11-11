import argparse
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

g_config = None
g_engine = None
g_inferLock = threading.Lock()


def makeError(errorCode, msg=None):
    return {
        "error_code": errorCode,
        "error_msg": kErrorMsg.get(errorCode, "unknown error") if msg is None else msg,
    }


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
    def __init__(self, modelPath, device, conf):
        from ultralytics import YOLO
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
        shareRoot = os.path.abspath(g_config.shareRoot)
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


def processRequest(req):
    startMs = time.perf_counter()
    errorCode = validateRequest(req)
    if errorCode != kOk:
        return makeError(errorCode), 0.0
    images, errorCode = loadImages(req)
    if errorCode != kOk:
        return makeError(errorCode), 0.0
    allResults = []
    firstWidth, firstHeight = 0, 0
    for name, data, width, height in images:
        with g_inferLock:
            defects = g_engine.predict(data, name, width, height)
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
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {fmt % args}", flush=True)

    def do_GET(self):
        self.sendJson(makeError(kInvalidParam, "use POST with a json body"), httpStatus=404)

    def do_POST(self):
        try:
            self.handleInfer()
        except Exception as exc:
            print(f"[ERROR] unhandled exception: {exc}", flush=True)
            try:
                self.sendJson(makeError(kSystemError))
            except Exception:
                pass

    def handleInfer(self):
        pathOnly = self.path.split("?")[0].rstrip("/")
        if pathOnly != g_config.endpoint.rstrip("/"):
            self.sendJson(makeError(kInvalidParam, f"unknown endpoint: {self.path}"),
                          httpStatus=404, extraHeaders={"Connection": "close"})
            return
        contentLength = int(self.headers.get("Content-Length") or 0)
        if contentLength <= 0 or contentLength > kMaxBodyBytes:
            self.sendJson(makeError(kInvalidParam, f"invalid content length: {contentLength}"),
                          extraHeaders={"Connection": "close"})
            return
        rawBody = self.rfile.read(contentLength)
        try:
            req = json.loads(rawBody.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.sendJson(makeError(kInvalidParam, "request body is not valid utf-8 json"))
            return
        resp, elapsedMs = processRequest(req)
        mode = "base64" if isinstance(req, dict) and "images_base64" in req else "share_dir"
        jobInfo = req.get("job_id", "?") if isinstance(req, dict) else "?"
        fileCount = len(req.get("file_names") or []) if isinstance(req, dict) else 0
        print(
            f"[REQ] job_id={jobInfo} files={fileCount} mode={mode} "
            f"-> error_code={resp['error_code']} elapsed={elapsedMs}ms",
            flush=True,
        )
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


def parseArgs():
    parser = argparse.ArgumentParser(description="Shimada image defect predict service (JSON over HTTP)")
    parser.add_argument("--host", default="127.0.0.1", help="bind address, default 127.0.0.1")
    parser.add_argument("--port", type=int, default=9000, help="listen port, default 9000")
    parser.add_argument("--endpoint", default="/image_defect", help="service path, default /image_defect")
    parser.add_argument("--model", default=None, help="ultralytics *.pt weights; omit to run mock inference")
    parser.add_argument("--device", default=None, help="inference device, e.g. cpu or 0")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold in yolo mode")
    parser.add_argument("--share-root", dest="shareRoot", default="./shared",
                        help="root dir of the shared image folder, default ./shared")
    return parser.parse_args()


def main():
    global g_config, g_engine
    g_config = parseArgs()
    if g_config.model:
        g_engine = YoloEngine(g_config.model, g_config.device, g_config.conf)
        mode = f"yolo({g_config.model})"
    else:
        g_engine = MockEngine()
        mode = "mock"
    os.makedirs(g_config.shareRoot, exist_ok=True)
    httpd = ThreadingHTTPServer((g_config.host, g_config.port), DefectHandler)
    print("ShimadaPredictService started")
    print(f"  endpoint   : http://{g_config.host}:{g_config.port}{g_config.endpoint}")
    print(f"  mode       : {mode}")
    print(f"  share root : {os.path.abspath(g_config.shareRoot)}")
    print("Press Ctrl+C to stop.", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nService stopped.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
