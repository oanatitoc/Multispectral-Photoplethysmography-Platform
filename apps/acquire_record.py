from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import sys

import serial

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppg_suite.io_utils import create_run_dir, save_json, subject_dir
from ppg_suite.live_preview import LivePreview


def open_serial_reset(port: str, baud: int) -> serial.Serial:
    s = serial.Serial(port, baud, timeout=1)
    s.setDTR(False)
    time.sleep(0.1)
    s.setDTR(True)
    time.sleep(1.0)
    return s


def read_header(ser: serial.Serial, timeout_s: float = 5.0) -> Optional[list[str]]:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        line = ser.readline().decode(errors="ignore").strip()
        if line.startswith("ms,"):
            return [c.strip() for c in line.split(",")]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Record one run from the ESP32 into the dataset layout.")
    parser.add_argument("--dataset-dir", default=str(ROOT / "dataset"))
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--duration", type=float, default=180.0, help="Seconds. Use 0 for manual stop with Ctrl+C.")
    parser.add_argument("--preview-channel", default="NIR_diff")
    parser.add_argument("--operator", default="")
    parser.add_argument("--protocol", default="")
    parser.add_argument("--live", action="store_true", help="Show live waveform + HR preview while recording.")
    parser.add_argument("--max-preview-samples", type=int, default=700)
    args = parser.parse_args()

    sdir = subject_dir(args.dataset_dir, args.subject_id)
    if not sdir.exists():
        raise FileNotFoundError(f"Subject not found: {sdir}. Create it first with create_subject.py")

    ser = open_serial_reset(args.port, args.baud)
    header = read_header(ser)
    if header is None:
        raise RuntimeError("No CSV header received from firmware.")

    run_dir = create_run_dir(args.dataset_dir, args.subject_id, datetime.now())
    raw_path = run_dir / "raw" / "tcs3448_raw.csv"
    meta_path = run_dir / "meta" / "run_metadata.json"

    preview = LivePreview(
        header=header,
        channel=args.preview_channel,
        max_samples=args.max_preview_samples,
        enabled=args.live,
    )

    run_meta = {
        "subject_id": args.subject_id,
        "start_time": datetime.now().isoformat(timespec="seconds"),
        "operator": args.operator,
        "protocol": args.protocol,
        "port": args.port,
        "baud": args.baud,
        "preview_channel": args.preview_channel,
        "columns": header,
    }
    save_json(meta_path, run_meta)

    print(f"Recording to: {raw_path}")
    print("Press Ctrl+C to stop." if args.duration == 0 else f"Recording for {args.duration:.1f} seconds")

    t0 = time.time()
    rows_written = 0

    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pc_time_s", *header])

        try:
            while True:
                raw = ser.readline().decode(errors="ignore").strip()
                if not raw or not raw[0].isdigit():
                    continue
                parts = raw.split(",")
                if len(parts) != len(header):
                    continue

                now_s = time.time()
                writer.writerow([now_s, *parts])
                rows_written += 1

                if args.live:
                    preview.update(parts, now_s)

                if rows_written % 100 == 0:
                    elapsed = time.time() - t0
                    print(f"rows={rows_written} elapsed={elapsed:.1f}s", end="\r")

                if args.duration > 0 and (time.time() - t0) >= args.duration:
                    break
        except KeyboardInterrupt:
            pass

    preview.close()

    run_meta["stop_time"] = datetime.now().isoformat(timespec="seconds")
    run_meta["rows_written"] = rows_written
    run_meta["duration_s"] = time.time() - t0
    save_json(meta_path, run_meta)

    print(f"\nDone. Saved run in: {run_dir}")


if __name__ == "__main__":
    main()