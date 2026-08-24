from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


RAW_NAME = "tcs3448_raw.csv"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_signal_csv(csv_path: str | Path, channel: str, drop_start_sec: float = 0.0, drop_end_sec: float = 0.0) -> Dict[str, Any]:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if channel not in df.columns:
        raise ValueError(f"{channel} not found in CSV columns: {list(df.columns)}")

    if "us" in df.columns:
        t_s = (df["us"].to_numpy(dtype=float) - float(df["us"].iloc[0])) / 1_000_000.0
    elif "ms" in df.columns:
        t_s = (df["ms"].to_numpy(dtype=float) - float(df["ms"].iloc[0])) / 1000.0
    else:
        raise ValueError("CSV must contain 'us' or 'ms' column.")

    raw = df[channel].to_numpy(dtype=float)
    if len(t_s) == 0:
        raise ValueError("Empty CSV after loading.")

    trim_mask = np.ones(len(t_s), dtype=bool)
    if drop_start_sec > 0 or drop_end_sec > 0:
        trim_mask = (t_s >= drop_start_sec) & (t_s <= (t_s[-1] - drop_end_sec))
        t_s = t_s[trim_mask]
        raw = raw[trim_mask]

    if len(t_s) < 10:
        raise ValueError("Too few samples after trimming.")

    return {
        "df": df,
        "df_trimmed": df.loc[trim_mask].reset_index(drop=True),
        "trim_mask": trim_mask,
        "t_s": t_s,
        "raw": raw,
        "channel": channel,
        "csv_path": csv_path,
    }

def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_subject_metadata(run_dir: str | Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    meta_path = run_dir.parent / "subject_metadata.json"
    if meta_path.exists():
        return load_json(meta_path)
    return {}

def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_json_safe(data), f, indent=2, ensure_ascii=False)


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    return value


def write_df(path: str | Path, df: pd.DataFrame) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def parse_timestamp_from_name(name: str) -> Optional[datetime]:
    m = re.search(r"(\d{8})_(\d{6})", name)
    if not m:
        return None
    return datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M%S")


def subject_dir(dataset_dir: str | Path, subject_id: str) -> Path:
    return ensure_dir(Path(dataset_dir) / subject_id)


def create_subject(dataset_dir: str | Path, subject_id: str, metadata: Dict[str, Any]) -> Path:
    sdir = subject_dir(dataset_dir, subject_id)
    meta_path = sdir / "subject_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            current = json.load(f)
    else:
        current = {}
    current.update(metadata)
    current.setdefault("subject_id", subject_id)
    current.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
    save_json(meta_path, current)
    return sdir


def next_run_index(subject_path: str | Path) -> int:
    subject_path = Path(subject_path)
    existing = sorted(p for p in subject_path.glob("run_*") if p.is_dir())
    max_idx = 0
    for path in existing:
        m = re.match(r"run_(\d{4})_", path.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def create_run_dir(dataset_dir: str | Path, subject_id: str, timestamp: datetime | None = None) -> Path:
    sdir = subject_dir(dataset_dir, subject_id)
    run_idx = next_run_index(sdir)
    if timestamp is None:
        timestamp = datetime.now()
    run_name = f"run_{run_idx:04d}_{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir = ensure_dir(sdir / run_name)
    ensure_dir(run_dir / "raw")
    ensure_dir(run_dir / "meta")
    ensure_dir(run_dir / "analysis")
    return run_dir


def import_legacy_csv(csv_path: str | Path, dataset_dir: str | Path, subject_id: str, copy: bool = True) -> Tuple[Path, Path]:
    csv_path = Path(csv_path)
    ts = parse_timestamp_from_name(csv_path.name)
    run_dir = create_run_dir(dataset_dir, subject_id, ts)
    raw_out = run_dir / "raw" / RAW_NAME
    if copy:
        shutil.copy2(csv_path, raw_out)
    else:
        raw_out.write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")
    meta = {
        "subject_id": subject_id,
        "source_file": str(csv_path),
        "imported_at": datetime.now().isoformat(timespec="seconds"),
        "run_kind": "legacy_import",
    }
    save_json(run_dir / "meta" / "run_metadata.json", meta)
    return run_dir, raw_out


def analysis_dir_for_run(run_dir: str | Path, channel: str) -> Path:
    return ensure_dir(Path(run_dir) / "analysis" / channel)


def resolve_run_dir_from_input(csv_path: str | Path, output_dir: str | Path | None, channel: str) -> Tuple[Path, Path]:
    csv_path = Path(csv_path)
    if output_dir is not None:
        outdir = ensure_dir(Path(output_dir))
        ensure_dir(outdir / "analysis")
        ensure_dir(outdir / "raw")
        return outdir, ensure_dir(outdir / "analysis" / channel)

    if csv_path.name == RAW_NAME and csv_path.parent.name == "raw":
        run_dir = csv_path.parent.parent
        return run_dir, analysis_dir_for_run(run_dir, channel)

    run_dir = ensure_dir(csv_path.parent / f"{csv_path.stem}_analysis")
    ensure_dir(run_dir / "raw")
    ensure_dir(run_dir / "analysis")
    return run_dir, analysis_dir_for_run(run_dir, channel)
