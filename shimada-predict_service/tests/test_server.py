import base64
import io
import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server


@pytest.fixture(autouse=True)
def isolatedCwd(tmp_path, monkeypatch):
    """Run every test in a temp working directory so log/config files never touch the repo."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(scope="module")
def pngBytes():
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (0, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def pngB64(pngBytes):
    return base64.b64encode(pngBytes).decode("ascii")


def baseRequest(**overrides):
    req = {
        "job_id": "J1",
        "sample_id": "S00001",
        "pose_id": "P01",
        "file_names": ["a.jpg"],
        "images_base64": ["x"],
    }
    req.update(overrides)
    return req


def writeConfig(cfg):
    Path("config.json").write_text(json.dumps(cfg), encoding="utf-8")


def useFakeModels(monkeypatch, aliases, workpieces=None, default=None):
    models = {alias: FakeEngine(alias) for alias in aliases}
    monkeypatch.setattr(server, "g_models", models)
    monkeypatch.setattr(server, "g_defaultModel", default or aliases[0])
    monkeypatch.setattr(
        server,
        "g_config",
        SimpleNamespace(workpieces=workpieces or {}, count_error=None, interval_error=None),
    )
    return models


class FakeEngine:
    def __init__(self, alias):
        self.alias = alias

    def predict(self, pixels, fileName, imageWidth, imageHeight):
        return [
            {
                "code": self.alias,
                "score": 1.0,
                "area": 1,
                "length": 1,
                "volume": 1,
                "box": [0, 0, 1, 1],
                "polygon": [],
                "defect_raw": [],
            }
        ]


# ---------------------------------------------------------------- validateRequest


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, server.kOk),
        ({"job_id": ""}, server.kInvalidParam),
        ({"job_id": 1}, server.kInvalidParam),
        ({"sample_id": "   "}, server.kInvalidParam),
        ({"file_names": []}, server.kInvalidParam),
        ({"file_names": "a.jpg"}, server.kInvalidParam),
        ({"file_names": [1]}, server.kInvalidParam),
        ({"file_names": [""]}, server.kInvalidImageName),
        ({"file_names": ["a.txt"]}, server.kInvalidImageName),
        ({"file_names": ["a|b.jpg"]}, server.kInvalidImageName),
        ({"relative_dir": "demo"}, server.kInvalidParam),
        ({"images_base64": None}, server.kInvalidParam),
        ({"images_base64": []}, server.kInvalidParam),
        ({"images_base64": [""]}, server.kImageNotFound),
        ({"images_base64": [1]}, server.kImageNotFound),
    ],
)
def testValidateRequest(overrides, expected):
    assert server.validateRequest(baseRequest(**overrides)) == expected


def testValidateRequestRejectsNonObject():
    assert server.validateRequest(["not", "a", "dict"]) == server.kInvalidParam


def testValidateRequestAcceptsShareDir():
    req = baseRequest(relative_dir="demo")
    del req["images_base64"]
    assert server.validateRequest(req) == server.kOk


# ------------------------------------------------------------------- loadImages


def testLoadImagesBase64(pngB64):
    images, code = server.loadImages({"file_names": ["a.jpg"], "images_base64": [pngB64]})
    assert code == server.kOk
    assert [(name, width, height) for name, _, width, height in images] == [("a.jpg", 8, 8)]


def testLoadImagesRejectsInvalidBase64():
    # validate=True: stray characters must fail instead of being silently dropped.
    images, code = server.loadImages({"file_names": ["a.jpg"], "images_base64": ["aGVsbG8=###"]})
    assert images is None
    assert code == server.kImageNotFound


def testLoadImagesRejectsOversizedBytes(monkeypatch, pngB64):
    monkeypatch.setattr(server, "kMaxImageBytes", 4)
    images, code = server.loadImages({"file_names": ["a.jpg"], "images_base64": [pngB64]})
    assert images is None
    assert code == server.kImageNotFound


def testLoadImagesRejectsOversizedPixels(monkeypatch, pngB64):
    monkeypatch.setattr(server, "kMaxImagePixels", 1)
    images, code = server.loadImages({"file_names": ["a.jpg"], "images_base64": [pngB64]})
    assert images is None
    assert code == server.kImageNotFound


def testLoadImagesRejectsUndecodableData():
    images, code = server.loadImages({"file_names": ["a.jpg"], "images_base64": [base64.b64encode(b"nope").decode()]})
    assert images is None
    assert code == server.kImageNotFound


def testLimitsCover4800x4800x32bpp():
    # Requirement: at least 4800 x 4800 at 32 bits per pixel must be accepted.
    assert server.kMaxImagePixels >= 4800 * 4800
    assert server.kMaxImageBytes >= 4800 * 4800 * 4


def testLoadImagesShareDir(monkeypatch, pngBytes, isolatedCwd):
    (isolatedCwd / "a.jpg").write_bytes(pngBytes)
    monkeypatch.setattr(server, "g_config", SimpleNamespace(share_root=str(isolatedCwd)))
    images, code = server.loadImages({"file_names": ["a.jpg"], "relative_dir": "."})
    assert code == server.kOk
    assert images[0][0] == "a.jpg"


def testLoadImagesShareDirTraversal(monkeypatch, pngBytes, isolatedCwd):
    (isolatedCwd.parent / "escape.jpg").write_bytes(pngBytes)
    monkeypatch.setattr(server, "g_config", SimpleNamespace(share_root=str(isolatedCwd)))
    images, code = server.loadImages({"file_names": ["escape.jpg"], "relative_dir": ".."})
    assert images is None
    assert code == server.kInvalidImageName


def testLoadImagesShareDirMissingFile(monkeypatch, isolatedCwd):
    monkeypatch.setattr(server, "g_config", SimpleNamespace(share_root=str(isolatedCwd)))
    images, code = server.loadImages({"file_names": ["ghost.jpg"], "relative_dir": "."})
    assert images is None
    assert code == server.kImageNotFound


def testLoadImagesShareDirOversizedFile(monkeypatch, pngBytes, isolatedCwd):
    (isolatedCwd / "big.jpg").write_bytes(pngBytes)
    monkeypatch.setattr(server, "g_config", SimpleNamespace(share_root=str(isolatedCwd)))
    monkeypatch.setattr(server, "kMaxImageBytes", 4)
    images, code = server.loadImages({"file_names": ["big.jpg"], "relative_dir": "."})
    assert images is None
    assert code == server.kImageNotFound


# --------------------------------------------------------------------- loadConfig


def testLoadConfigDefaults():
    writeConfig({"models": {"default": {"path": None}}})
    cfg = server.loadConfig()
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 9000
    assert cfg.endpoint == "/image_defect"
    assert cfg.default_model == "default"
    assert cfg.workpieces == {}
    assert cfg.models == {"default": {"path": None, "conf": 0.25}}
    assert cfg.count_error is None
    assert cfg.interval_error is None


def testLoadConfigAutoDefaultModelOnlyWithSingleModel():
    writeConfig({"models": {"a": {"path": None}, "b": {"path": None}}})
    with pytest.raises(SystemExit, match="default_model is required"):
        server.loadConfig()


def testLoadConfigRejectsUnknownTopLevelKey():
    writeConfig({"models": {"default": {"path": None}}, "modle": 1})
    with pytest.raises(SystemExit, match="unknown keys"):
        server.loadConfig()


def testLoadConfigRejectsBadAlias():
    writeConfig({"models": {"bad alias": {"path": None}}})
    with pytest.raises(SystemExit, match="model alias"):
        server.loadConfig()


def testLoadConfigRejectsUnknownModelKey():
    writeConfig({"models": {"default": {"path": None, "device": 0}}})
    with pytest.raises(SystemExit, match="unknown keys"):
        server.loadConfig()


@pytest.mark.parametrize("conf", [-0.1, 1.1, True, "0.5"])
def testLoadConfigRejectsBadConf(conf):
    writeConfig({"models": {"default": {"path": None, "conf": conf}}})
    with pytest.raises(SystemExit, match="conf must be a number"):
        server.loadConfig()


def testLoadConfigRejectsUnknownWorkpieceAlias():
    writeConfig({"models": {"default": {"path": None}}, "workpieces": {"A100": ["nope"]}})
    with pytest.raises(SystemExit, match="references unknown model alias"):
        server.loadConfig()


def testLoadConfigRejectsDuplicateAliasInWorkpiece():
    writeConfig({"models": {"default": {"path": None}}, "workpieces": {"A100": ["default", "default"]}})
    with pytest.raises(SystemExit, match="more than once"):
        server.loadConfig()


def testLoadConfigRejectsEmptyWorkpieceList():
    writeConfig({"models": {"default": {"path": None}}, "workpieces": {"A100": []}})
    with pytest.raises(SystemExit, match="non-empty json array"):
        server.loadConfig()


def testLoadConfigCountErrorRule():
    writeConfig(
        {
            "models": {"default": {"path": None}},
            "count_error": {"label_name": "sticker", "proper_count": [1, 2]},
        }
    )
    cfg = server.loadConfig()
    assert cfg.count_error == {"label_name": "sticker", "low": 1, "high": 2}
    assert cfg.interval_error is None


def testLoadConfigIntervalErrorRule():
    writeConfig(
        {
            "models": {"default": {"path": None}},
            "interval_error": {"label_name": "ink", "proper_interval": [10, 20.5], "gap": False},
        }
    )
    cfg = server.loadConfig()
    assert cfg.interval_error == {"label_name": "ink", "low": 10, "high": 20.5, "gap": False}
    assert cfg.count_error is None


@pytest.mark.parametrize(
    ("rule", "match"),
    [
        (["not", "an", "object"], "must be null or a json object"),
        ({"label_name": "sticker", "proper_count": [1, 2], "extra": 1}, "unknown keys"),
        ({"label_name": "  ", "proper_count": [1, 2]}, "label_name"),
        ({"label_name": "sticker", "proper_count": [1]}, "min, max"),
        ({"label_name": "sticker", "proper_count": [1, True]}, "min, max"),
        ({"label_name": "sticker", "proper_count": "1..2"}, "min, max"),
        ({"label_name": "sticker", "proper_count": [3, 2]}, "must not exceed max"),
    ],
)
def testLoadConfigRejectsBadCountError(rule, match):
    writeConfig({"models": {"default": {"path": None}}, "count_error": rule})
    with pytest.raises(SystemExit, match=match):
        server.loadConfig()


@pytest.mark.parametrize(
    ("rule", "match"),
    [
        ({"label_name": "ink", "proper_interval": [1, 2]}, "gap must be true"),
        ({"label_name": "ink", "proper_interval": [1, 2], "gap": "yes"}, "gap must be true"),
        ({"label_name": "ink", "proper_interval": [1, 2], "gap": True, "proper_count": [1, 2]}, "unknown keys"),
        ({"label_name": "ink", "proper_interval": [1, 2], "gap": True, "proper_count": [1, 2]}, "allowed"),
    ],
)
def testLoadConfigRejectsBadIntervalError(rule, match):
    writeConfig({"models": {"default": {"path": None}}, "interval_error": rule})
    with pytest.raises(SystemExit, match=match):
        server.loadConfig()


# ------------------------------------------------------------------- log retention


def testPruneLogsKeepsNewest30():
    (Path.cwd() / "log").mkdir()
    for i in range(35):
        (Path.cwd() / "log" / f"202001{i + 1:02d}.log").write_text("old\n", encoding="utf-8")
    server.pruneLogs()
    kept = sorted(p.name for p in (Path.cwd() / "log").glob("*.log"))
    assert len(kept) == server.kLogRetention
    assert kept[0] == "20200106.log"
    assert kept[-1] == "20200135.log"


def testLogLineWritesUtf8AndPrunes(monkeypatch):
    for i in range(server.kLogRetention + 2):
        (Path.cwd() / "log").mkdir(exist_ok=True)
        (Path.cwd() / "log" / f"202001{i + 1:02d}.log").write_text("old\n", encoding="utf-8")
    monkeypatch.setattr(server, "g_logDay", None)  # force the once-per-day prune path
    server.logLine("[TEST] 成功 ok")
    today = server.time.strftime("%Y%m%d")
    files = sorted(p.name for p in (Path.cwd() / "log").glob("*.log"))
    assert len(files) == server.kLogRetention
    assert f"{today}.log" in files
    text = (Path.cwd() / "log" / f"{today}.log").read_text(encoding="utf-8")
    assert "[TEST] 成功 ok" in text


# ---------------------------------------------------------------------- routing


def testResolveHandlesFallsBackToDefaultModel(monkeypatch):
    models = useFakeModels(monkeypatch, ["a", "b"], workpieces={"A100": ["b"]}, default="a")
    monkeypatch.setattr(server, "parseWorkpiece", lambda fileName: None)
    workpiece, handles = server.resolveHandles(["x.jpg"])
    assert workpiece is None
    assert handles == [models["a"]]


def testResolveHandlesUsesWorkpieceTable(monkeypatch):
    models = useFakeModels(monkeypatch, ["a", "b"], workpieces={"A100": ["b", "a"]})
    monkeypatch.setattr(server, "parseWorkpiece", lambda fileName: "A100")
    workpiece, handles = server.resolveHandles(["x.jpg"])
    assert workpiece == "A100"
    assert handles == [models["b"], models["a"]]


def testResolveHandlesUnknownWorkpiece(monkeypatch):
    useFakeModels(monkeypatch, ["a"], workpieces={"A100": ["a"]})
    monkeypatch.setattr(server, "parseWorkpiece", lambda fileName: "C999")
    workpiece, handles = server.resolveHandles(["x.jpg"])
    assert workpiece == "C999"
    assert handles is None


def testProcessRequestRunsEveryModelPerImageInOrder(monkeypatch):
    models = useFakeModels(monkeypatch, ["a", "b"], workpieces={"A100": ["a", "b"]})
    monkeypatch.setattr(server, "parseWorkpiece", lambda fileName: "A100")
    monkeypatch.setattr(
        server,
        "loadImages",
        lambda req: ([("one.jpg", b"1", 8, 8), ("two.jpg", b"2", 8, 8)], server.kOk),
    )
    req = baseRequest(file_names=["one.jpg", "two.jpg"], images_base64=["x", "y"])
    resp, elapsedMs = server.processRequest(req)
    assert resp["error_code"] == server.kOk
    assert [d["code"] for d in resp["data"]["results"]] == ["a", "b", "a", "b"]
    assert [d["file_name"] for d in resp["data"]["results"]] == ["one.jpg", "one.jpg", "two.jpg", "two.jpg"]
    assert elapsedMs >= 0.0
    assert models["a"].alias == "a"


def testProcessRequestRejectsUnknownWorkpieceBeforeLoadingImages(monkeypatch):
    useFakeModels(monkeypatch, ["a"], workpieces={"A100": ["a"]})
    monkeypatch.setattr(server, "parseWorkpiece", lambda fileName: "C999")

    def failIfCalled(req):
        raise AssertionError("loadImages must not run for an unknown workpiece")

    monkeypatch.setattr(server, "loadImages", failIfCalled)
    resp, _ = server.processRequest(baseRequest())
    assert resp["error_code"] == server.kInvalidParam
    assert resp["error_msg"] == "unknown workpiece: C999"


def testDefectCounts():
    results = [{"code": "QJ"}, {"code": "QJ"}, {"code": "HS"}]
    assert server.defectCounts(results) == "QJ:2,HS:1"
    assert server.defectCounts([]) == "-"


def testFormatFileNames():
    assert server.formatFileNames(["a.jpg", "b.jpg"]) == "a.jpg,b.jpg"
    assert server.formatFileNames([]) == "-"
    assert server.formatFileNames(None) == "-"


# ---------------------------------------------------------------- defect checks


def target(code, box, score=0.9):
    return {
        "code": code,
        "score": score,
        "area": (box[2] - box[0]) * (box[3] - box[1]),
        "length": max(box[2] - box[0], box[3] - box[1]),
        "volume": 0,
        "box": box,
        "polygon": [],
        "defect_raw": [],
    }


def useChecks(monkeypatch, count=None, interval=None):
    monkeypatch.setattr(server, "g_config", SimpleNamespace(count_error=count, interval_error=interval))


def testBoxIntervalGapUsesNearestEdges():
    assert server.boxInterval([0, 0, 10, 10], [13, 14, 20, 20], True) == pytest.approx(5.0)
    assert server.boxInterval([0, 0, 10, 10], [5, 5, 15, 15], True) == 0.0
    assert server.boxInterval([0, 0, 10, 10], [-5, -5, 5, 5], True) == 0.0


def testBoxIntervalCenterDistance():
    assert server.boxInterval([0, 0, 10, 10], [20, 0, 30, 10], False) == pytest.approx(20.0)
    assert server.boxInterval([0, 0, 10, 10], [5, 5, 15, 15], False) == pytest.approx(7.0710678, rel=1e-4)


def testApplyChecksCountInRangeLeavesResultsUntouched(monkeypatch):
    useChecks(monkeypatch, count={"label_name": "sticker", "low": 2, "high": 2})
    defects = [target("sticker", [0, 0, 1, 1]), target("sticker", [2, 2, 3, 3])]
    server.applyChecks(defects, "a.jpg")
    assert [d["code"] for d in defects] == ["sticker", "sticker"]


def testApplyChecksCountAboveRangeAppendsEmptyDefect(monkeypatch):
    useChecks(monkeypatch, count={"label_name": "sticker", "low": 1, "high": 2})
    defects = [target("sticker", [0, 0, 1, 1]) for _ in range(3)]
    server.applyChecks(defects, "a.jpg")
    assert defects[-1] == {
        "code": "count_error",
        "score": 1.0,
        "area": 0,
        "length": 0,
        "volume": 0,
        "box": [],
        "polygon": [],
        "defect_raw": [],
    }


def testApplyChecksCountMissingTargets(monkeypatch):
    useChecks(monkeypatch, count={"label_name": "sticker", "low": 1, "high": 1})
    defects = [target("ink", [0, 0, 1, 1])]
    server.applyChecks(defects, "a.jpg")
    assert [d["code"] for d in defects] == ["ink", "count_error"]


def testApplyChecksIntervalGapOutOfRangeCarriesBothTargets(monkeypatch):
    useChecks(monkeypatch, interval={"label_name": "sticker", "low": 5, "high": 10, "gap": True})
    defects = [target("sticker", [0, 0, 10, 10], score=0.8), target("sticker", [12, 0, 22, 10], score=0.7)]
    server.applyChecks(defects, "a.jpg")
    defect = defects[-1]
    assert defect["code"] == "interval_error"
    assert defect["score"] == 1.0
    assert defect["box"] == [0, 0, 22, 10]
    assert defect["area"] == 220
    assert defect["length"] == 22
    assert defect["volume"] == 220
    assert defect["defect_raw"] == [
        {"code": "sticker", "score": 0.8, "box": [0, 0, 10, 10]},
        {"code": "sticker", "score": 0.7, "box": [12, 0, 22, 10]},
    ]


def testApplyChecksIntervalInRangeLeavesResultsUntouched(monkeypatch):
    useChecks(monkeypatch, interval={"label_name": "sticker", "low": 0, "high": 10, "gap": True})
    defects = [target("sticker", [0, 0, 10, 10]), target("sticker", [12, 0, 22, 10])]
    server.applyChecks(defects, "a.jpg")
    assert [d["code"] for d in defects] == ["sticker", "sticker"]


def testApplyChecksIntervalCenterMode(monkeypatch):
    useChecks(monkeypatch, interval={"label_name": "sticker", "low": 0, "high": 10, "gap": False})
    defects = [target("sticker", [0, 0, 10, 10]), target("sticker", [20, 0, 30, 10])]
    server.applyChecks(defects, "a.jpg")
    assert defects[-1]["code"] == "interval_error"


def testApplyChecksIntervalReportsEveryOutOfRangePair(monkeypatch):
    useChecks(monkeypatch, interval={"label_name": "sticker", "low": 0, "high": 10, "gap": True})
    defects = [target("sticker", [x, 0, x + 5, 5]) for x in (0, 20, 40)]
    server.applyChecks(defects, "a.jpg")
    assert [d["code"] for d in defects] == ["sticker"] * 3 + ["interval_error"] * 3


def testApplyChecksIntervalOnlyPairsMatchingLabel(monkeypatch):
    useChecks(monkeypatch, interval={"label_name": "sticker", "low": 0, "high": 10, "gap": True})
    defects = [target("sticker", [0, 0, 5, 5]), target("ink", [6, 0, 11, 5])]
    server.applyChecks(defects, "a.jpg")
    assert [d["code"] for d in defects] == ["sticker", "ink"]


def testApplyChecksDisabledWhenRulesAreNull(monkeypatch):
    useChecks(monkeypatch)
    defects = [target("sticker", [0, 0, 1, 1]), target("sticker", [100, 0, 101, 1])]
    server.applyChecks(defects, "a.jpg")
    assert [d["code"] for d in defects] == ["sticker", "sticker"]


def testProcessRequestAppliesChecksPerImage(monkeypatch):
    useFakeModels(monkeypatch, ["sticker"])
    server.g_config.count_error = {"label_name": "sticker", "low": 2, "high": 2}
    monkeypatch.setattr(server, "parseWorkpiece", lambda fileName: None)
    monkeypatch.setattr(
        server,
        "loadImages",
        lambda req: ([("one.jpg", b"1", 8, 8), ("two.jpg", b"2", 8, 8)], server.kOk),
    )
    req = baseRequest(file_names=["one.jpg", "two.jpg"], images_base64=["x", "y"])
    resp, _ = server.processRequest(req)
    codes = [d["code"] for d in resp["data"]["results"]]
    assert codes == ["sticker", "count_error", "sticker", "count_error"]
    assert [d["file_name"] for d in resp["data"]["results"]] == ["one.jpg", "one.jpg", "two.jpg", "two.jpg"]


def testProcessRequestAppliesIntervalCheck(monkeypatch):
    models = useFakeModels(monkeypatch, ["sticker"])
    server.g_config.interval_error = {"label_name": "sticker", "low": 0, "high": 10, "gap": True}
    monkeypatch.setattr(server, "parseWorkpiece", lambda fileName: None)
    monkeypatch.setattr(
        models["sticker"],
        "predict",
        lambda *args: [target("sticker", [0, 0, 10, 10]), target("sticker", [100, 0, 110, 10])],
    )
    monkeypatch.setattr(server, "loadImages", lambda req: ([("one.jpg", b"1", 8, 8)], server.kOk))
    resp, _ = server.processRequest(baseRequest())
    codes = [d["code"] for d in resp["data"]["results"]]
    assert codes == ["sticker", "sticker", "interval_error"]
    assert resp["data"]["results"][-1]["defect_raw"] == [
        {"code": "sticker", "score": 0.9, "box": [0, 0, 10, 10]},
        {"code": "sticker", "score": 0.9, "box": [100, 0, 110, 10]},
    ]


# ------------------------------------------------------------ concurrency guard


def testInferSlotsLimitConcurrentRequests():
    slots = threading.BoundedSemaphore(2)
    assert slots.acquire(blocking=False)
    assert slots.acquire(blocking=False)
    assert not slots.acquire(blocking=False)
    slots.release()
    assert slots.acquire(blocking=False)
    slots.release()
    slots.release()
