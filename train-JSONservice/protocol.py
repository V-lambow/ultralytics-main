"""Payload 与 stdout 事件协议：子进程进、父进程出的唯一通道。"""

import json
import os
import sys
import tempfile

EVENT_PREFIX = "__TRAIN_SERVICE_EVENT__="
PROTOCOL_VERSION = 1
SUPPORTED_ACTIONS = ("train",)
TERMINAL_EVENTS = frozenset({"training_completed", "training_error", "training_stopped"})

TRAIN_ARGS_ALLOWED = frozenset(
    {
        "model",
        "data",
        "project",
        "name",
        "exist_ok",
        "task",
        "device",
        "epochs",
        "batch",
        "imgsz",
        "workers",
        "time",
        "patience",
        "resume",
        "cache",
        "fraction",
        "classes",
        "single_cls",
        "rect",
        "pretrained",
        "seed",
        "deterministic",
        "verbose",
        "plots",
        "val",
        "save",
        "save_period",
        "optimizer",
        "cos_lr",
        "amp",
        "multi_scale",
        "close_mosaic",
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "warmup_epochs",
        "warmup_momentum",
        "warmup_bias_lr",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "dropout",
        "box",
        "cls",
        "dfl",
        "pose",
        "kobj",
    }
)

SERVICE_ONLY_ARGS = frozenset({"useRoi", "roi"})

LEGACY_FIELD_ALIASES = {"modelPath": "model", "dataYaml": "data", "alias": "name"}


def create_payload(action, args, paths=None, options=None):
    payload = {
        "version": PROTOCOL_VERSION,
        "action": action,
        "args": args,
        "paths": paths or {},
        "options": options or {},
    }
    fd, payload_path = tempfile.mkstemp(prefix="train-json-", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False)
    return payload_path


def load_payload(payload_path):
    with open(payload_path, encoding="utf-8") as stream:
        payload = json.load(stream)

    if payload.get("version") != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported payload version: {payload.get('version')}")
    if payload.get("action") not in SUPPORTED_ACTIONS:
        raise ValueError(f"Unsupported payload action: {payload.get('action')}")

    unexpected = set(payload) - {"version", "action", "args", "paths", "options"}
    if unexpected:
        raise ValueError(f"Unsupported payload fields: {sorted(unexpected)}")
    for field in ("args", "paths", "options"):
        if not isinstance(payload.get(field), dict):
            raise ValueError(f"Payload field '{field}' must be an object")

    unsupported = set(payload["args"]) - TRAIN_ARGS_ALLOWED - SERVICE_ONLY_ARGS
    if unsupported:
        raise ValueError(f"Unsupported training arguments: {sorted(unsupported)}")
    missing = {key for key in ("model", "data") if not payload["args"].get(key)}
    if missing:
        raise ValueError(f"Missing required training arguments: {sorted(missing)}")
    if any(not isinstance(value, str) for value in payload["paths"].values()):
        raise ValueError("Payload paths must be strings")
    return payload


def emit(event, **data):
    payload = {"version": PROTOCOL_VERSION, "event": event, **data}
    stream = sys.__stdout__ or sys.stdout
    stream.write(f"{EVENT_PREFIX}{json.dumps(payload, ensure_ascii=False)}\n")
    stream.flush()


def parse_line(line):
    cleaned = line.strip()
    if not cleaned:
        return None
    if not cleaned.startswith(EVENT_PREFIX):
        return "training_log", {"message": cleaned}

    try:
        payload = json.loads(cleaned[len(EVENT_PREFIX) :])
    except json.JSONDecodeError:
        return "training_log", {"message": cleaned}

    if payload.get("version") != PROTOCOL_VERSION:
        return "protocol_error", {"error": f"Unsupported event version: {payload.get('version')}"}
    event = payload.pop("event", None)
    payload.pop("version", None)
    if not event:
        return "protocol_error", {"error": "Event payload missing 'event'"}
    return event, payload


class EventLogStream:
    """把子进程 stdout 的文本切成 training_log 事件，换行与回车都作为分隔符。"""

    def __init__(self, event="training_log"):
        self._event = event
        self._buffer = ""

    def write(self, text):
        if not text:
            return
        self._buffer += text
        while True:
            indexes = [i for i in (self._buffer.find("\n"), self._buffer.find("\r")) if i >= 0]
            if not indexes:
                break
            index = min(indexes)
            line, self._buffer = self._buffer[:index], self._buffer[index + 1 :]
            if line.strip():
                emit(self._event, message=line.strip())

    def flush(self):
        line = self._buffer.strip()
        if line:
            emit(self._event, message=line)
        self._buffer = ""
