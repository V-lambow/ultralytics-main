"""父进程管理器：拉起训练子进程、解析 stdout 事件、杀进程树、状态落盘。"""

import json
import os
import signal
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

from protocol import TERMINAL_EVENTS, create_payload, parse_line

STATE_FILENAME = "service_state.json"
IGNORED_STATE_EVENTS = frozenset({"training_log", "training_metrics"})


def build_worker_command(payload_path):
    if getattr(sys, "frozen", False):
        return [sys.executable, "--worker", str(payload_path)]
    worker = Path(__file__).with_name("worker.py")
    return [sys.executable, "-u", str(worker), str(payload_path)]


def build_worker_env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def worker_creation_flags():
    if os.name != "nt":
        return 0
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def reap(process, timeout):
    try:
        process.wait(timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        return False


def terminate_process_tree(process):
    if process is None or process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/T", "/PID", str(process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if reap(process, 5):
                return
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if reap(process, 5):
                return
            process.kill()
            reap(process, 5)
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            if reap(process, 5):
                return
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            reap(process, 5)
    except OSError:
        process.kill()
        reap(process, 5)


def state_path(project, name):
    return Path(project or ".") / (name or "runs") / STATE_FILENAME


def read_state(project=None, name=None):
    path = state_path(project, name)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


class TrainingManager:
    def __init__(self, callbacks=None):
        self.callbacks = list(callbacks or [])
        self.is_training = False
        self.save_dir = None

        self._process = None
        self._stop_event = threading.Event()
        self._done = threading.Event()
        self._done.set()
        self._state_file = None
        self._terminal_sent = False

    def notify_callbacks(self, event, data):
        for callback in self.callbacks:
            try:
                callback(event, data)
            except Exception:
                pass

    def start_training(self, train_args, options=None):
        if self.is_training:
            return False, "Training is already in progress"

        try:
            payload_path = create_payload("train", train_args, options=options)
        except Exception as exc:
            return False, f"Failed to create payload: {exc}"

        self._state_file = state_path(train_args.get("project"), train_args.get("name"))
        self._stop_event.clear()
        self._done.clear()
        self._terminal_sent = False
        self.is_training = True

        thread = threading.Thread(target=self._run_training, args=(payload_path,), daemon=True)
        thread.start()
        return True, "Training started"

    def wait(self, timeout=None):
        return self._done.wait(timeout)

    def stop_training(self):
        if not self.is_training:
            return False
        self._stop_event.set()
        process = self._process
        if process is not None:
            threading.Thread(target=terminate_process_tree, args=(process,), daemon=True).start()
        return True

    def _run_training(self, payload_path):
        try:
            if self._stop_event.is_set():
                self.is_training = False
                self._dispatch("training_stopped", {})
                return

            process = subprocess.Popen(
                build_worker_command(payload_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=build_worker_env(),
                start_new_session=os.name != "nt",
                creationflags=worker_creation_flags(),
            )
            self._process = process

            while True:
                if self._stop_event.is_set():
                    terminate_process_tree(process)
                    self.is_training = False
                    self._dispatch("training_stopped", {})
                    return

                line = process.stdout.readline()

                if self._stop_event.is_set():
                    terminate_process_tree(process)
                    self.is_training = False
                    self._dispatch("training_stopped", {})
                    return
                if line == "" and process.poll() is not None:
                    break
                if not line:
                    continue

                parsed = parse_line(line)
                if parsed is None:
                    continue
                event, data = parsed
                self._dispatch(event, data)

            return_code = process.poll()
            self.is_training = False
            if self._terminal_sent:
                return
            if return_code == 0:
                self._dispatch("training_completed", {"results": "Training completed successfully"})
            else:
                self._dispatch("training_error", {"error": f"Training process exited with code {return_code}"})
        except Exception as exc:
            self.is_training = False
            if self._stop_event.is_set():
                self._dispatch("training_stopped", {})
            else:
                self._dispatch("training_error", {"error": str(exc)})
        finally:
            self._process = None
            self._done.set()
            try:
                os.remove(payload_path)
            except OSError:
                pass

    def _dispatch(self, event, data):
        if event in TERMINAL_EVENTS:
            if self._terminal_sent:
                return
            self._terminal_sent = True
        self.notify_callbacks(event, data)
        self._record(event, data)

    def _record(self, event, data):
        if self._state_file is None:
            return
        if event in IGNORED_STATE_EVENTS:
            return
        if data.get("save_dir"):
            self.save_dir = data["save_dir"]
        record = {
            "event": event,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "save_dir": self.save_dir,
        }
        for key in ("error", "results", "total_epochs"):
            if key in data:
                record[key] = data[key]
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            self._state_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass
