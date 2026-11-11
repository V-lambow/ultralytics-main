import argparse
import base64
import io
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

kOk = 0
kSystemError = -1001
kInvalidParam = -1002
kInvalidImageName = -1003
kImageNotFound = -1004


def makeTestImage(width, height):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (width, height), (32, 32, 40))
    draw = ImageDraw.Draw(img)
    for _ in range(150):
        x = random.randint(0, width - 2)
        y = random.randint(0, height - 2)
        gray = random.randint(60, 220)
        size = random.randint(1, 5)
        draw.rectangle([x, y, min(x + size, width - 1), min(y + size, height - 1)], fill=(gray, gray, gray))
    boxW = random.randint(width // 8, width // 3)
    boxH = random.randint(height // 8, height // 3)
    x0 = random.randint(0, width - boxW)
    y0 = random.randint(0, height - boxH)
    draw.rectangle([x0, y0, x0 + boxW, y0 + boxH], outline=(220, 30, 30), width=3)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def defaultJobId():
    return f"J{time.strftime('%Y%m%d%H%M%S')}{random.randint(0, 999):03d}"


def buildBase64Payload(imageBytesList, fileNames, args):
    return {
        "job_id": args.jobId,
        "sample_id": args.sampleId,
        "pose_id": args.poseId,
        "file_names": fileNames,
        "images_base64": [base64.b64encode(b).decode("ascii") for b in imageBytesList],
        "ext_params": {"client": "shimada-test-client", "sent_at": time.strftime("%Y-%m-%d %H:%M:%S")},
    }


def buildPathPayload(fileNames, relativeDir, args):
    return {
        "job_id": args.jobId,
        "sample_id": args.sampleId,
        "pose_id": args.poseId,
        "file_names": fileNames,
        "relative_dir": relativeDir,
        "ext_params": {"client": "shimada-test-client"},
    }


def postJson(url, payload, timeoutSec):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    startMs = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeoutSec) as resp:
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        status = exc.code
    except urllib.error.URLError as exc:
        return None, None, round((time.perf_counter() - startMs) * 1000, 1), f"connection failed: {exc.reason}"
    elapsedMs = round((time.perf_counter() - startMs) * 1000, 1)
    try:
        return status, json.loads(raw.decode("utf-8")), elapsedMs, None
    except (UnicodeDecodeError, json.JSONDecodeError):
        return status, None, elapsedMs, "response is not valid json"


def printResult(status, resp, elapsedMs, err):
    print(f"HTTP status : {status if status is not None else 'N/A'}")
    print(f"elapsed     : {elapsedMs} ms")
    if err:
        print(f"error       : {err}")
        return
    print(f"error_code  : {resp.get('error_code')}")
    print(f"error_msg   : {resp.get('error_msg')}")
    data = resp.get("data") or {}
    if data:
        print(f"job_id      : {data.get('job_id')}")
        print(f"sample_id   : {data.get('sample_id')}")
        print(f"pose_id     : {data.get('pose_id')}")
        print(f"resolution  : {data.get('image_width')} x {data.get('image_height')}")
        results = data.get("results") or []
        print(f"defects     : {len(results)}")
        for r in results:
            print(f"  - file={r.get('file_name')} code={r.get('code')} score={r.get('score')} "
                  f"area={r.get('area')} length={r.get('length')} box={r.get('box')}")


def cmdGenImages(args):
    os.makedirs(args.out, exist_ok=True)
    for i in range(args.count):
        data = makeTestImage(args.width, args.height)
        name = f"{args.sampleId}_C01_{args.poseId}_L{i}_{time.strftime('%Y%m%d%H%M%S')}.jpg"
        path = os.path.join(args.out, name)
        with open(path, "wb") as f:
            f.write(data)
        print(f"generated: {os.path.abspath(path)}")
    print(f"total: {args.count}")


def cmdSendBase64(args):
    if args.images:
        imageBytesList = []
        fileNames = []
        for p in args.images:
            with open(p, "rb") as f:
                imageBytesList.append(f.read())
            fileNames.append(os.path.basename(p))
    else:
        imageBytesList = [makeTestImage(1920, 1440)]
        fileNames = [f"{args.sampleId}_C01_{args.poseId}_L0_{time.strftime('%Y%m%d%H%M%S')}.jpg"]
        print("no --images given, using an auto-generated test image")
    payload = buildBase64Payload(imageBytesList, fileNames, args)
    print(f"POST {args.url}")
    print(f"job_id      : {args.jobId}")
    print(f"images      : {fileNames}")
    status, resp, elapsedMs, err = postJson(args.url, payload, args.timeout)
    printResult(status, resp, elapsedMs, err)


def cmdSendPath(args):
    fileNames = [os.path.basename(p) for p in args.files]
    payload = buildPathPayload(fileNames, args.dir, args)
    print(f"POST {args.url}")
    print(f"job_id      : {args.jobId}")
    print(f"relative_dir: {args.dir}")
    print(f"files       : {fileNames}")
    status, resp, elapsedMs, err = postJson(args.url, payload, args.timeout)
    printResult(status, resp, elapsedMs, err)


def cmdTestErrors(args):
    imageBytes = makeTestImage(640, 480)
    b64Image = base64.b64encode(imageBytes).decode("ascii")
    name = f"{args.sampleId}_C01_{args.poseId}_L0_test.jpg"

    def basePayload():
        return {
            "job_id": defaultJobId(),
            "sample_id": args.sampleId,
            "pose_id": args.poseId,
        }

    cases = [
        ("valid base64 request",
         {**basePayload(), "file_names": [name], "images_base64": [b64Image]}, kOk),
        ("missing job_id",
         {"sample_id": args.sampleId, "pose_id": args.poseId,
          "file_names": [name], "images_base64": [b64Image]},
         kInvalidParam),
        ("missing sample_id",
         {"job_id": defaultJobId(), "pose_id": args.poseId,
          "file_names": [name], "images_base64": [b64Image]},
         kInvalidParam),
        ("file_names/images_base64 length mismatch",
         {**basePayload(), "job_id": defaultJobId(), "file_names": [name, name], "images_base64": [b64Image]},
         kInvalidParam),
        ("empty file_names",
         {**basePayload(), "job_id": defaultJobId(), "file_names": [], "images_base64": []},
         kInvalidParam),
        ("images_base64 and relative_dir both present",
         {**basePayload(), "job_id": defaultJobId(), "file_names": [name],
          "images_base64": [b64Image], "relative_dir": "demo/"},
         kInvalidParam),
        ("neither images_base64 nor relative_dir",
         {**basePayload(), "job_id": defaultJobId(), "file_names": [name]},
         kInvalidParam),
        ("empty file name",
         {**basePayload(), "job_id": defaultJobId(), "file_names": [" "], "images_base64": [b64Image]},
         kInvalidImageName),
        ("illegal file name",
         {**basePayload(), "job_id": defaultJobId(), "file_names": ["a|b<img>.jpg"], "images_base64": [b64Image]},
         kInvalidImageName),
        ("unsupported file extension",
         {**basePayload(), "job_id": defaultJobId(), "file_names": ["ghost.exe"], "images_base64": [b64Image]},
         kInvalidImageName),
        ("corrupted base64 image",
         {**basePayload(), "job_id": defaultJobId(), "file_names": [name], "images_base64": ["@@not-base64@@"]},
         kImageNotFound),
        ("empty base64 image",
         {**basePayload(), "job_id": defaultJobId(), "file_names": [name], "images_base64": [""]},
         kImageNotFound),
        ("image not found on share",
         {**basePayload(), "job_id": defaultJobId(),
          "file_names": ["ghost.jpg"], "relative_dir": "no_such_dir/"},
         kImageNotFound),
    ]
    passed = 0
    failed = 0
    for title, payload, expected in cases:
        status, resp, elapsedMs, err = postJson(args.url, payload, args.timeout)
        actual = None if (err or resp is None) else resp.get("error_code")
        ok = actual == expected
        passed += 1 if ok else 0
        failed += 0 if ok else 1
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {title}: expected={expected} actual={actual} ({elapsedMs} ms)")
    print(f"summary: {passed} passed, {failed} failed, total {len(cases)}")
    return 0 if failed == 0 else 1


def parseArgs():
    parser = argparse.ArgumentParser(description="Test client for the shimada image defect service")
    parser.add_argument("--url", default="http://127.0.0.1:9000/image_defect", help="service endpoint")
    parser.add_argument("--timeout", type=float, default=30.0, help="request timeout in seconds")
    parser.add_argument("--job-id", dest="jobId", default=None, help="job_id; auto-generated when omitted")
    parser.add_argument("--sample-id", dest="sampleId", default="S00001", help="sample_id")
    parser.add_argument("--pose-id", dest="poseId", default="P01", help="pose_id")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("gen-images", help="generate local test images")
    gen.add_argument("--out", default="test_images", help="output dir, default ./test_images")
    gen.add_argument("--count", type=int, default=2, help="number of images")
    gen.add_argument("--width", type=int, default=1920, help="image width")
    gen.add_argument("--height", type=int, default=1440, help="image height")

    sendB64 = sub.add_parser("send-base64", help="send request with base64-encoded images")
    sendB64.add_argument("--images", nargs="+", default=None,
                         help="local image paths; omit to auto-generate one test image")

    sendPath = sub.add_parser("send-path", help="send request with shared-folder relative paths")
    sendPath.add_argument("--dir", required=True, help="relative dir under the server share root")
    sendPath.add_argument("--files", nargs="+", required=True, help="image file names inside --dir")

    sub.add_parser("test-errors", help="run the protocol error-code test suite")

    args = parser.parse_args()
    if args.jobId is None:
        args.jobId = defaultJobId()
    return args


def main():
    args = parseArgs()
    handlers = {
        "gen-images": cmdGenImages,
        "send-base64": cmdSendBase64,
        "send-path": cmdSendPath,
        "test-errors": cmdTestErrors,
    }
    sys.exit(handlers[args.command](args) or 0)


if __name__ == "__main__":
    main()
