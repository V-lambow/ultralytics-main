import argparse
import json
import logging
import signal
import sys

import zmq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


def server_mode(config: dict, config_path: str):
    from train_service import TrainService

    service = TrainService(config, config_path)

    def on_signal(sig, frame):
        logger.info("Shutting down...")
        service.stop_server()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    service.start_server()


def client_mode(config: dict, cmd: str):
    port = config.get("train", {}).get("port", 9257)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(f"tcp://localhost:{port + 1}")

    sock.send_json({"cmd": cmd})
    resp = sock.recv_json()

    print(json.dumps(resp, indent=2, ensure_ascii=False))

    sock.close()
    ctx.term()


def subscribe_mode(config: dict):
    port = config.get("train", {}).get("port", 9257)

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.connect(f"tcp://localhost:{port}")
    sub.setsockopt(zmq.SUBSCRIBE, b"")

    logger.info(f"Subscribing on tcp://localhost:{port}  (Ctrl+C to quit)\n")

    try:
        while True:
            topic, data = sub.recv_multipart()
            topic_str = topic.decode()
            msg = json.loads(data)

            if topic_str == "metrics":
                e = msg["epoch"]
                n = msg["total_epochs"]
                print(
                    f"Epoch {e:>4}/{n} | "
                    f"loss: {msg['loss']:.4f} "
                    f"(box: {msg['box_loss']:.4f} cls: {msg['cls_loss']:.4f} dfl: {msg['dfl_loss']:.4f}) | "
                    f"P: {msg['precision']:.4f}  R: {msg['recall']:.4f}  "
                    f"mAP50: {msg['mAP50']:.4f}  mAP50-95: {msg['mAP50_95']:.4f}"
                )
            elif topic_str == "error":
                print(f"[ERROR] {msg.get('message', '')}")
            elif topic_str == "train_end":
                print(f"[DONE] Results saved to: {msg.get('save_dir', '')}")
            else:
                print(f"[{topic_str}] {json.dumps(msg, ensure_ascii=False)}")
    except KeyboardInterrupt:
        logger.info("Unsubscribed.")
    finally:
        sub.close()
        ctx.term()


def main():
    parser = argparse.ArgumentParser(description="YOLO Train Service")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument(
        "--cmd",
        choices=["train", "pause", "holdon", "stop", "status"],
        help="Send command to running service",
    )
    parser.add_argument(
        "--subscribe", action="store_true", help="Subscribe to training metrics"
    )
    args = parser.parse_args()

    config = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)

    if args.subscribe:
        subscribe_mode(config)
    elif args.cmd:
        if not args.config:
            parser.error("--config is required with --cmd")
        client_mode(config, args.cmd)
    else:
        if not args.config:
            parser.error("--config is required for server mode")
        server_mode(config, args.config)


if __name__ == "__main__":
    main()
