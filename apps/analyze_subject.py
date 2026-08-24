from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze all runs for one subject.")
    parser.add_argument("--dataset-dir", default=str(ROOT / "dataset"))
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--channel", default="auto_fz_nir")
    parser.add_argument("--drop-start-sec", type=float, default=3.0)
    parser.add_argument("--drop-end-sec", type=float, default=3.0)
    parser.add_argument("--segments", default="baseline:0:60,task:60:120,recovery:120:180")
    args = parser.parse_args()

    subject_dir = Path(args.dataset_dir) / args.subject_id
    raw_files = sorted(subject_dir.glob("run_*/raw/tcs3448_raw.csv"))
    if not raw_files:
        print(f"No runs found for {args.subject_id}")
        return

    cmd_base = [
        sys.executable,
        str(ROOT / "apps" / "analyze_run.py"),
        "--channel", args.channel,
        "--drop-start-sec", str(args.drop_start_sec),
        "--drop-end-sec", str(args.drop_end_sec),
        "--segments", args.segments,
    ]

    for raw in raw_files:
        print(f"Analyzing {raw} ...")
        subprocess.run([*cmd_base, "--input", str(raw)], check=True)

    print(f"Done. Analyzed {len(raw_files)} run(s).")


if __name__ == "__main__":
    main()
