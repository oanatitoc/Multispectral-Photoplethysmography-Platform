from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppg_suite.io_utils import create_subject


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or update a subject in the dataset.")
    parser.add_argument("--dataset-dir", default=str(ROOT / "dataset"))
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--name", default="")
    parser.add_argument("--age", type=float, default=None)
    parser.add_argument("--sex", default="")
    parser.add_argument("--height-cm", type=float, default=None)
    parser.add_argument("--weight-kg", type=float, default=None)
    parser.add_argument("--skin-tone", default="")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    bmi = None
    if args.height_cm and args.weight_kg and args.height_cm > 0:
        bmi = args.weight_kg / ((args.height_cm / 100.0) ** 2)

    metadata = {
        "name": args.name,
        "age": args.age,
        "sex": args.sex,
        "height_cm": args.height_cm,
        "weight_kg": args.weight_kg,
        "bmi": bmi,
        "skin_tone": args.skin_tone,
        "notes": args.notes,
    }
    metadata = {k: v for k, v in metadata.items() if v not in ("", None)}

    subject_path = create_subject(args.dataset_dir, args.subject_id, metadata)
    print(f"Subject saved: {subject_path}")


if __name__ == "__main__":
    main()
