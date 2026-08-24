from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd


def load_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def resolve_summary_path(run_dir: Path, channel: str) -> Path | None:
    unified = run_dir / "analysis" / "summary.json"
    if channel == "auto" and unified.exists():
        return unified

    legacy = run_dir / "analysis" / channel / "summary.json"
    if legacy.exists():
        return legacy

    if unified.exists():
        return unified
    return None


def flatten_dict(prefix: str, value, row: dict) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            flatten_dict(f"{prefix}__{key}" if prefix else str(key), nested, row)
        return
    if isinstance(value, list):
        row[prefix] = json.dumps(value, ensure_ascii=False)
        return
    row[prefix] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one flat validation table for all analyzed runs of a subject.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--channel", default="auto", help="auto uses analysis/summary.json when available; otherwise use a channel folder such as auto_fz_nir.")
    args = parser.parse_args()

    subject_dir = Path(args.dataset_dir) / args.subject_id
    rows = []
    snapshot_rows = []

    for run_dir in sorted(subject_dir.glob("run_*")):
        summary_path = resolve_summary_path(run_dir, args.channel)
        meta_path = run_dir / "meta" / "run_metadata.json"
        raw_path = run_dir / "raw" / "tcs3448_raw.csv"
        if summary_path is None or not summary_path.exists():
            continue

        summary = load_json(summary_path)
        meta = load_json(meta_path) if meta_path.exists() else {}
        row = {
            "run_name": run_dir.name,
            "raw_csv": str(raw_path),
            "summary_path": str(summary_path),
            "channel": summary.get("channel", args.channel),
            "data_mode": summary.get("data_mode"),
            "selected_channel": summary.get("selected_channel", args.channel),
            "source_file": meta.get("source_file"),
            "start_time": meta.get("start_time"),
            "run_kind": meta.get("run_kind"),
            "protocol": meta.get("protocol"),
        }

        for module_name, module_info in summary.get("modules", {}).items():
            status = module_info.get("status")
            row[f"{module_name}_status"] = status
            flatten_dict(module_name, module_info.get("summary", {}), row)

        rows.append(row)

        snapshots_path = run_dir / "analysis" / "red_nir_12ch" / "validation_snapshots.csv"
        if snapshots_path.exists():
            snapshots_df = pd.read_csv(snapshots_path)
            for _, snap in snapshots_df.iterrows():
                snap_row = {
                    "run_name": run_dir.name,
                    "run_metadata_protocol": row.get("protocol"),
                    "data_mode": row.get("data_mode"),
                    "selected_channel": row.get("selected_channel"),
                    "summary_path": str(summary_path),
                    "validation_snapshots_path": str(snapshots_path),
                }
                snap_row.update(snap.to_dict())
                snapshot_rows.append(snap_row)

    df = pd.DataFrame(rows)
    out_path = subject_dir / f"{args.subject_id}_{args.channel}_validation_table.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    if snapshot_rows:
        snapshots_out_path = subject_dir / f"{args.subject_id}_{args.channel}_validation_snapshots_table.csv"
        pd.DataFrame(snapshot_rows).to_csv(snapshots_out_path, index=False)
        print(f"Saved: {snapshots_out_path}")


if __name__ == "__main__":
    main()
