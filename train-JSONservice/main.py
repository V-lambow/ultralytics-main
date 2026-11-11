"""CLI：按配置跑一次训练并打印事件流，Ctrl+C 停止训练。"""

import argparse
import json
import multiprocessing
import re
import signal
import sys
from pathlib import Path

from manager import TrainingManager, read_state
from protocol import (
    LEGACY_FIELD_ALIASES,
    SERVICE_ONLY_ARGS,
    TERMINAL_EVENTS,
    TRAIN_ARGS_ALLOWED,
    emit,
)

ANSI_RE = re.compile(r"\x1B(?:[@-Z\-_]|\[[0-?]*[ -/]*[@-~])")
PROGRESS_RE = re.compile(r"\d{1,3}%|\bit/s\b|\bs/it\b")


def load_train_config(path):
    with open(path, encoding="utf-8") as stream:
        config = json.load(stream)
    train_cfg = config.get("train")
    if not isinstance(train_cfg, dict):
        raise ValueError("Config must contain a 'train' object")

    resolved = dict(train_cfg)
    for legacy, current in LEGACY_FIELD_ALIASES.items():
        if legacy in resolved:
            resolved.setdefault(current, resolved.pop(legacy))
    return resolved


def build_train_args(train_cfg):
    train_args = {key: value for key, value in train_cfg.items() if key in TRAIN_ARGS_ALLOWED}
    for key in SERVICE_ONLY_ARGS:
        if key in train_cfg:
            train_args[key] = train_cfg[key]
    return train_args


def resolve_resume(train_cfg, resume):
    if resume is None:
        return None
    if resume:
        return resume

    state = read_state(train_cfg.get("project"), train_cfg.get("name"))
    save_dir = (state or {}).get("save_dir")
    if not save_dir:
        raise ValueError("Cannot resolve checkpoint: no previous save_dir in state file")
    checkpoint = Path(save_dir) / "weights" / "last.pt"
    if not checkpoint.exists():
        raise ValueError(f"Checkpoint not found: {checkpoint}")
    return str(checkpoint)


def print_event(event, data):
    if event == "training_log":
        message = ANSI_RE.sub("", str(data.get("message", "")))
        if not message:
            return
        if PROGRESS_RE.search(message):
            sys.stdout.write(f"\r{message}")
        else:
            sys.stdout.write(f"\r{message}\n")
        sys.stdout.flush()
        return

    if event == "training_started":
        print(f"[START] epochs={data.get('total_epochs')} save_dir={data.get('save_dir')}")
    elif event == "training_completed":
        print(f"[DONE] {data.get('results', '')}")
    elif event == "training_stopped":
        print("[STOPPED] stopped by user")
    elif event == "training_error":
        print(f"[ERROR] {data.get('error', '')}", file=sys.stderr)
        if data.get("traceback"):
            print(data["traceback"], file=sys.stderr)
    elif event == "protocol_error":
        print(f"[PROTOCOL] {data.get('error', '')}", file=sys.stderr)
    else:
        print(f"[{event}] {json.dumps(data, ensure_ascii=False)}")


def show_status(train_cfg):
    state = read_state(train_cfg.get("project"), train_cfg.get("name"))
    if state is None:
        print("No training state found.")
        return 0
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


def run_training(train_cfg, resume):
    train_args = build_train_args(train_cfg)
    if resume is not None:
        train_args["resume"] = resolve_resume(train_cfg, resume)

    outcome = {"event": None}

    def track(event, data):
        if event in TERMINAL_EVENTS:
            outcome["event"] = event

    def forward(event, data):
        emit(event, **data)

    manager = TrainingManager(callbacks=[print_event, forward, track])
    started, message = manager.start_training(train_args)
    if not started:
        print(f"[ERROR] {message}", file=sys.stderr)
        return 1

    def stop(signum, frame):
        print("\n[STOP] stopping training...", file=sys.stderr)
        manager.stop_training()

    signal.signal(signal.SIGINT, stop)
    manager.wait()

    event = outcome["event"]
    if event == "training_completed":
        return 0
    if event == "training_stopped":
        return 130
    return 1


def run_worker(argv):
    if "--worker" not in argv:
        return None
    index = argv.index("--worker")
    if index + 1 >= len(argv):
        raise SystemExit("Usage: <exe> --worker <payload.json>")
    from worker import main as run

    run(argv[index + 1])
    return 0


def main():
    worker_exit = run_worker(sys.argv)
    if worker_exit is not None:
        return worker_exit

    parser = argparse.ArgumentParser(description="YOLO train service (child process)")
    parser.add_argument("--config", required=True, help="Path to config JSON file")
    parser.add_argument("--status", action="store_true", help="Print last training state and exit")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="",
        default=None,
        metavar="CKPT",
        help="Resume from a checkpoint; omit the path to reuse the last save_dir",
    )
    args = parser.parse_args()

    try:
        train_cfg = load_train_config(args.config)
        if args.status:
            return show_status(train_cfg)
        return run_training(train_cfg, args.resume)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    from multiprocessing.spawn import is_forking

    if is_forking(sys.argv):
        from worker import install_gray_shim

        install_gray_shim()
    multiprocessing.freeze_support()
    sys.exit(main())
