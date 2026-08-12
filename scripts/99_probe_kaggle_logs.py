"""Probe a running Kaggle kernel until it reaches a decisive startup marker."""

import argparse
import threading
import time

from kaggle.api.kaggle_api_extended import KaggleApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel")
    parser.add_argument("--require-training", action="store_true")
    parser.add_argument("--snapshot-seconds", type=float)
    args = parser.parse_args()

    api = KaggleApi()
    api.authenticate()

    if args.snapshot_seconds:
        lines: list[str] = []
        last_event_at = [0.0]

        def collect() -> None:
            for event in api.kernels_logs_stream(args.kernel):
                data = (event.get("data") or "").strip()
                if data:
                    lines.append(data)
                    last_event_at[0] = time.monotonic()

        threading.Thread(target=collect, daemon=True).start()
        deadline = time.monotonic() + args.snapshot_seconds
        while time.monotonic() < deadline:
            if lines and time.monotonic() - last_event_at[0] >= 1.5:
                break
            time.sleep(0.1)
        print("\n".join(lines[-12:]) if lines else "NO_LOG_EVENTS")
        return

    install_count = 0
    for event in api.kernels_logs_stream(args.kernel):
        data = event.get("data") or ""
        if "installing torch 2.2.2" in data:
            install_count += 1
        if (
            "P100-compatible torch 2.2.2 already active" in data
            and not args.require_training
        ):
            print("STARTUP_OK: compatible torch survived restart")
            return
        if "开始训练" in data:
            print("TRAINING_OK: training loop started")
            return
        if "Traceback (most recent call last)" in data:
            print("STARTUP_ERROR: traceback detected")
            raise SystemExit(2)
        if install_count >= 3:
            print("STARTUP_LOOP: torch install repeated at least three times")
            raise SystemExit(3)


if __name__ == "__main__":
    main()
