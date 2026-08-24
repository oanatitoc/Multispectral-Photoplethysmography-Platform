from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppg_suite.io_utils import create_subject, import_legacy_csv


def is_raw_measurement_csv(path: Path) -> bool:
    name = path.name
    if not name.endswith(".csv"):
        return False
    if any(token in name for token in [
        "_beats", "_summary", "_segments", "_metrics", "_traces",
        "_perfusion", "_vasomotion", "_resp", "_hrv"
    ]):
        return False
    return bool(re.search(r"\d{8}_\d{6}\.csv$", name))


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Import legacy raw CSV files into the dataset layout.")
    parser.add_argument("--source", required=True, help="Folder with old CSV files.")
    parser.add_argument("--dataset-dir", default=str(ROOT / "dataset"))
    parser.add_argument("--subject-id", required=True)
    args = parser.parse_args()

    source_dir = Path(args.source)
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)

    create_subject(args.dataset_dir, args.subject_id, {"import_note": "legacy_imported_subject"})

    candidates = sorted(
        (p for p in source_dir.rglob("*.csv") if is_raw_measurement_csv(p)),
        key=lambda p: (len(str(p)), str(p)),
    )
    if not candidates:
        print("No raw measurement CSV files found.")
        return

    unique = []
    seen_hashes = set()
    for path in candidates:
        sha1 = file_sha1(path)
        if sha1 in seen_hashes:
            print(f"Skipping duplicate content: {path}")
            continue
        seen_hashes.add(sha1)
        unique.append(path)

    unique = sorted(unique, key=lambda p: p.name)

    for path in unique:
        _, raw_out = import_legacy_csv(path, args.dataset_dir, args.subject_id, copy=True)
        print(f"Imported {path.name} -> {raw_out}")

    print(f"Imported {len(unique)} unique run(s).")


if __name__ == "__main__":
    main()
