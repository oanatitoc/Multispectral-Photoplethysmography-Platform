from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from ppg_suite.calibration import (
    apply_pi_calibration,
    apply_rr_calibration,
    load_red_nir_12ch_calibration,
    resolve_spo2_params,
)
from ppg_suite.config import AnalysisConfig
from ppg_suite.modules.respiration import run_respiration
from ppg_suite.modules.stiffness import run_stiffness
from ppg_suite.modules.tissue_oxygenation_trend import run_tissue_oxygenation_trend


HEART_BAND = (0.7, 4.0)
DEFAULT_RED_CHANNEL = "F6"
DEFAULT_IR_CHANNEL = "NIR"
SATURATION_THRESHOLD = 3580.0
MIN_DC_COUNTS = 20.0
MIN_ACDC = 1e-5


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
        value = float(value)
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
    return value


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_json_safe(data), f, indent=2, ensure_ascii=False)


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def infer_run_dir(input_path: Path) -> Path | None:
    if input_path.name == "tcs3448_raw.csv" and input_path.parent.name == "raw":
        return input_path.parent.parent
    return None


def summarize_numeric_column(df: pd.DataFrame, column: str) -> dict[str, Any]:
    if column not in df.columns:
        return {}
    values = pd.to_numeric(df[column], errors="coerce")
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {}
    return {
        f"{column}_median": float(values.median()),
        f"{column}_mean": float(values.mean()),
        f"{column}_last": float(values.iloc[-1]),
    }


def summarize_live_metrics(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {}
    metrics_path = run_dir / "metrics_live.csv"
    if not metrics_path.exists():
        return {}
    df = pd.read_csv(metrics_path)
    summary: dict[str, Any] = {
        "metrics_file": str(metrics_path),
        "num_rows": int(len(df)),
    }
    for column in (
        "hr_bpm",
        "hr_fft_bpm",
        "mean_ibi_ms",
        "rmssd_ms",
        "sdnn_ms",
        "respiratory_rate_brpm",
        "spo2_estimated_pct",
        "spo2_ratio",
        "perfusion_proxy_pct",
        "perfusion_index_pct",
        "pulse_amplitude",
        "signal_quality",
        "saturation_fraction",
    ):
        summary.update(summarize_numeric_column(df, column))
    for column in ("best_channel", "selected_channel", "red_channel", "ir_channel", "spo2_status"):
        if column in df.columns and len(df[column].dropna()):
            summary[f"{column}_last"] = str(df[column].dropna().iloc[-1])
    if "artifact_flag" in df.columns and len(df):
        flags = pd.to_numeric(df["artifact_flag"], errors="coerce")
        summary["artifact_fraction"] = float(np.nanmean(flags))
    return summary


def summarize_ground_truth(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {}
    gt_path = run_dir / "ground_truth.csv"
    if not gt_path.exists():
        return {}
    df = pd.read_csv(gt_path)
    summary: dict[str, Any] = {
        "ground_truth_file": str(gt_path),
        "num_snapshots": int(len(df)),
    }
    if len(df) == 0:
        return summary
    for column in ("pulseox_hr_bpm", "pulseox_spo2_pct", "pulseox_pi_pct", "rr_ref_bpm", "ecg_hr_bpm", "ecg_hrv_metric"):
        summary.update(summarize_numeric_column(df, column))
    last = df.iloc[-1].to_dict()
    summary["last_snapshot"] = {k: v for k, v in last.items() if pd.notna(v) and v != ""}
    return summary


def summarize_events(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {}
    events_path = run_dir / "events.csv"
    if not events_path.exists():
        return {}
    df = pd.read_csv(events_path)
    return {
        "events_file": str(events_path),
        "num_events": int(len(df)),
        "events": df.to_dict(orient="records"),
    }


def numeric_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        out = float(value)
        return out if np.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def summarize_validation_snapshots(run_dir: Path | None, output_dir: Path, window_s: float = 10.0) -> dict[str, Any]:
    if run_dir is None:
        return {}
    gt_path = run_dir / "ground_truth.csv"
    metrics_path = run_dir / "metrics_live.csv"
    if not gt_path.exists() or not metrics_path.exists():
        return {}

    gt_df = pd.read_csv(gt_path)
    metrics_df = pd.read_csv(metrics_path)
    if len(gt_df) == 0 or len(metrics_df) == 0 or "timestamp_rel_s" not in gt_df.columns or "timestamp_rel_s" not in metrics_df.columns:
        return {}

    metrics_df = metrics_df.copy()
    metrics_df["timestamp_rel_s"] = pd.to_numeric(metrics_df["timestamp_rel_s"], errors="coerce")
    metrics_df = metrics_df[np.isfinite(metrics_df["timestamp_rel_s"])]
    if len(metrics_df) == 0:
        return {}

    metric_columns = [
        "hr_bpm",
        "hr_fft_bpm",
        "respiratory_rate_brpm",
        "spo2_estimated_pct",
        "spo2_ratio",
        "perfusion_index_pct",
        "pulse_amplitude",
        "signal_quality",
        "artifact_flag",
    ]
    rows: list[dict[str, Any]] = []

    for snapshot_idx, gt_row in gt_df.iterrows():
        snapshot_t = numeric_or_none(gt_row.get("timestamp_rel_s"))
        if snapshot_t is None:
            continue

        in_window = metrics_df[
            (metrics_df["timestamp_rel_s"] >= snapshot_t - window_s)
            & (metrics_df["timestamp_rel_s"] <= snapshot_t + window_s)
        ]
        if len(in_window) == 0:
            nearest_idx = (metrics_df["timestamp_rel_s"] - snapshot_t).abs().idxmin()
            in_window = metrics_df.loc[[nearest_idx]]

        out: dict[str, Any] = {
            "snapshot_index": int(snapshot_idx),
            "timestamp_iso": gt_row.get("timestamp_iso"),
            "timestamp_rel_s": snapshot_t,
            "match_window_s": float(window_s),
            "matched_live_rows": int(len(in_window)),
            "matched_live_time_min_s": float(in_window["timestamp_rel_s"].min()),
            "matched_live_time_max_s": float(in_window["timestamp_rel_s"].max()),
        }

        for col in gt_df.columns:
            if col in {"timestamp_iso", "timestamp_rel_s"}:
                continue
            out[f"gt_{col}"] = gt_row.get(col)

        for column in metric_columns:
            if column not in in_window.columns:
                continue
            values = pd.to_numeric(in_window[column], errors="coerce")
            values = values[np.isfinite(values)]
            out[f"live_{column}_median"] = None if len(values) == 0 else float(values.median())

        gt_hr = numeric_or_none(gt_row.get("pulseox_hr_bpm"))
        gt_spo2 = numeric_or_none(gt_row.get("pulseox_spo2_pct"))
        gt_pi = numeric_or_none(gt_row.get("pulseox_pi_pct"))
        gt_rr = numeric_or_none(gt_row.get("rr_ref_bpm"))
        live_hr = numeric_or_none(out.get("live_hr_bpm_median"))
        live_spo2 = numeric_or_none(out.get("live_spo2_estimated_pct_median"))
        live_pi = numeric_or_none(out.get("live_perfusion_index_pct_median"))
        live_rr = numeric_or_none(out.get("live_respiratory_rate_brpm_median"))

        out["hr_error_vs_pulseox_bpm"] = None if gt_hr is None or live_hr is None else float(live_hr - gt_hr)
        out["spo2_error_vs_pulseox_pct"] = None if gt_spo2 is None or live_spo2 is None else float(live_spo2 - gt_spo2)
        out["pi_proxy_minus_pulseox_pct_points"] = None if gt_pi is None or live_pi is None else float(live_pi - gt_pi)
        out["rr_error_vs_reference_brpm"] = None if gt_rr is None or live_rr is None else float(live_rr - gt_rr)
        rows.append(out)

    if not rows:
        return {}

    out_df = pd.DataFrame(rows)
    out_path = output_dir / "validation_snapshots.csv"
    out_df.to_csv(out_path, index=False)

    summary: dict[str, Any] = {
        "validation_snapshots_file": str(out_path),
        "num_snapshots": int(len(out_df)),
        "match_window_s": float(window_s),
        "comparison_note": (
            "Each ground-truth snapshot is compared against the median live metrics in a centered time window; "
            "this reduces manual-entry delay/noise and is better than a single instant comparison."
        ),
    }
    for column in (
        "hr_error_vs_pulseox_bpm",
        "spo2_error_vs_pulseox_pct",
        "pi_proxy_minus_pulseox_pct_points",
        "rr_error_vs_reference_brpm",
        "postrun_rr_error_vs_reference_brpm",
    ):
        summary.update(summarize_numeric_column(out_df, column))
    return summary


def add_postrun_respiration_to_validation_snapshots(output_dir: Path, respiration_module: dict[str, Any] | None) -> dict[str, Any]:
    path = output_dir / "validation_snapshots.csv"
    if not path.exists() or not respiration_module or respiration_module.get("status") != "ok":
        return {}

    resp_summary = respiration_module.get("summary", {})
    final_rr = numeric_or_none(resp_summary.get("final_respiratory_rate_brpm"))
    if final_rr is None:
        return {}

    df = pd.read_csv(path)
    if len(df) == 0:
        return {}

    df["postrun_respiration_final_brpm"] = float(final_rr)
    df["postrun_respiration_chosen_method"] = resp_summary.get("chosen_method")
    df["postrun_respiration_chosen_band_label"] = resp_summary.get("chosen_band_label")
    df["postrun_respiration_fast_candidate_brpm"] = resp_summary.get("fast_respiration_candidate_brpm")

    if "gt_rr_ref_bpm" in df.columns:
        gt_rr = pd.to_numeric(df["gt_rr_ref_bpm"], errors="coerce")
        df["postrun_rr_error_vs_reference_brpm"] = np.where(np.isfinite(gt_rr), final_rr - gt_rr, np.nan)

    df.to_csv(path, index=False)
    summary = {
        "postrun_respiration_final_brpm": float(final_rr),
        "postrun_respiration_chosen_method": resp_summary.get("chosen_method"),
        "postrun_respiration_chosen_band_label": resp_summary.get("chosen_band_label"),
    }
    if "postrun_rr_error_vs_reference_brpm" in df.columns:
        summary.update(summarize_numeric_column(df, "postrun_rr_error_vs_reference_brpm"))
    return summary


def build_analysis_config(
    reference_channel: str,
    subject_meta: dict[str, Any],
    run_meta: dict[str, Any],
    spo2_anchor_ratio: float | None,
    spo2_anchor_pct: float | None,
    spo2_slope: float,
    resp_target_brpm: float | None = None,
) -> AnalysisConfig:
    protocol = run_meta.get("protocol")
    target = resp_target_brpm
    if target is None:
        target = numeric_or_none(run_meta.get("respiration_target_brpm"))

    cfg = AnalysisConfig(
        channel=reference_channel,
        drop_start_sec=0.0,
        drop_end_sec=0.0,
        spo2_anchor_ratio=spo2_anchor_ratio,
        spo2_anchor_pct=spo2_anchor_pct,
        spo2_linear_slope=spo2_slope,
        protocol=protocol,
        resp_target_brpm=target,
    )
    if protocol == "post_exercise_recovery":
        cfg.resp_band = (0.10, 0.70)
    elif protocol == "controlled_breathing" and target is not None and target > 0:
        center = float(target) / 60.0
        cfg.resp_band = (max(0.05, center - 4.0 / 60.0), min(0.95, center + 4.0 / 60.0))

    height_cm = subject_meta.get("height_cm")
    if height_cm is not None:
        try:
            cfg.subject_height_m = float(height_cm) / 100.0
        except (TypeError, ValueError):
            pass
    return cfg


def build_tissue_loaded_alias(df_trim: pd.DataFrame, t_trim: np.ndarray, input_path: Path) -> dict[str, Any]:
    tissue_df = df_trim.copy()
    aliases = {
        "NIR": "NIR_diff",
        "F6": "F6_diff",
        "FXL": "FXL_diff",
        "FY": "FY_diff",
        "F4": "F4_diff",
        "F3": "F3_diff",
    }
    for source, alias in aliases.items():
        if source in tissue_df.columns and alias not in tissue_df.columns:
            tissue_df[alias] = tissue_df[source]
    return {
        "df": tissue_df,
        "df_trimmed": tissue_df,
        "trim_mask": np.ones(len(tissue_df), dtype=bool),
        "t_s": t_trim,
        "raw": tissue_df["NIR_diff"].to_numpy(dtype=float) if "NIR_diff" in tissue_df.columns else np.array([], dtype=float),
        "channel": "NIR_diff",
        "csv_path": input_path,
        "channel_aliases": {alias: source for source, alias in aliases.items() if alias in tissue_df.columns},
    }


def save_module_result(output_dir: Path, module_name: str, result: dict[str, Any]) -> dict[str, Any]:
    module_summary = result.get("summary", {})
    save_json(output_dir / f"{module_name}_summary.json", module_summary)
    for table_name, df in result.get("tables", {}).items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(output_dir / f"{table_name}.csv", index=False)
    return {
        "status": "ok",
        "summary_file": str(output_dir / f"{module_name}_summary.json"),
        "summary": module_summary,
    }


def error_module(exc: Exception) -> dict[str, Any]:
    return {
        "status": "error",
        "error": str(exc),
    }


def estimate_fs(t_s: np.ndarray) -> Optional[float]:
    if len(t_s) < 5:
        return None
    dt = np.diff(t_s)
    dt = dt[(dt > 0.005) & (dt < 1.0)]
    if len(dt) < 3:
        return None
    return float(1.0 / np.median(dt))


def bandpass(x: np.ndarray, fs: float, band: tuple[float, float] = HEART_BAND) -> Optional[np.ndarray]:
    if fs is None or len(x) < max(20, int(fs * 2)):
        return None
    nyq = 0.5 * fs
    lo = band[0] / nyq
    hi = band[1] / nyq
    if hi >= 1.0:
        hi = 0.99
    if lo <= 0.0 or lo >= hi:
        return None
    b, a = butter(2, [lo, hi], btype="band")
    return filtfilt(b, a, np.asarray(x, dtype=float))


def robust_scale(x: np.ndarray, center: float, floor: float) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float(floor)
    mad = float(np.median(np.abs(x - center)))
    return float(max(1.4826 * mad, floor))


def robust_inlier_mask(x: np.ndarray, zmax: float, min_scale: float) -> np.ndarray:
    x = np.asarray(x, float)
    finite = np.isfinite(x)
    out = np.zeros(len(x), dtype=bool)
    if not np.any(finite):
        return out
    center = float(np.median(x[finite]))
    scale = robust_scale(x[finite], center, min_scale)
    out[finite] = np.abs(x[finite] - center) <= zmax * scale
    return out


def safe_median(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else np.nan


def safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else np.nan


def saturation_fraction(x: np.ndarray, threshold: float = SATURATION_THRESHOLD) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.mean(x >= threshold))


def fft_bpm_quality(x: np.ndarray, fs: float) -> tuple[float, float]:
    yf = bandpass(x, fs)
    if yf is None or len(yf) < max(30, int(6 * fs)):
        return np.nan, 0.0
    y = yf - np.mean(yf)
    win = np.hanning(len(y))
    spec = np.abs(np.fft.rfft(y * win))
    freqs = np.fft.rfftfreq(len(y), d=1.0 / fs)
    mask = (freqs >= HEART_BAND[0]) & (freqs <= HEART_BAND[1])
    if not np.any(mask):
        return np.nan, 0.0
    fb = freqs[mask]
    sb = spec[mask]
    k = int(np.argmax(sb))
    q = float(sb[k] / (np.median(sb) + 1e-9))
    bpm = float(fb[k] * 60.0)
    return bpm, q


def detect_reference_peaks(raw: np.ndarray, t_s: np.ndarray, fs: float) -> dict[str, Any]:
    yf = bandpass(raw, fs)
    if yf is None:
        raise RuntimeError("Could not bandpass reference signal.")

    candidates = []
    for polarity_name, signal in (("positive", yf), ("negative", -yf)):
        prominence = max(0.5, 0.30 * float(np.std(signal)))
        min_distance = max(1, int(0.35 * fs))
        peaks, props = find_peaks(signal, distance=min_distance, prominence=prominence)
        if len(peaks) < 3:
            score = 0.0
            bpm = np.nan
            cv = np.nan
        else:
            ibi = np.diff(t_s[peaks])
            valid_ibi = ibi[(ibi >= 60.0 / 180.0) & (ibi <= 60.0 / 40.0)]
            if len(valid_ibi) < 2:
                score = 0.0
                bpm = np.nan
                cv = np.nan
            else:
                bpm = float(60.0 / np.median(valid_ibi))
                cv = float(np.std(valid_ibi) / max(np.mean(valid_ibi), 1e-9))
                prom = props.get("prominences", np.array([0.0]))
                median_prom = safe_median(prom)
                score = float(len(valid_ibi) * median_prom / (1.0 + 5.0 * cv))
        candidates.append({
            "polarity": polarity_name,
            "signal": signal,
            "peaks": peaks,
            "score": score,
            "bpm": bpm,
            "ibi_cv": cv,
        })

    best = max(candidates, key=lambda item: float(item["score"]))
    if len(best["peaks"]) < 3 or float(best["score"]) <= 0.0:
        raise RuntimeError("Could not detect enough regular reference peaks.")
    return best


def score_reference_channel(raw: np.ndarray, t_s: np.ndarray, fs: float) -> dict[str, Any]:
    try:
        detected = detect_reference_peaks(raw, t_s, fs)
        bpm_fft, q_fft = fft_bpm_quality(raw, fs)
        sat = saturation_fraction(raw)
        raw_range = float(np.nanpercentile(raw, 95) - np.nanpercentile(raw, 5))
        sat_penalty = max(0.0, 1.0 - 2.0 * sat) if np.isfinite(sat) else 1.0
        score = float(detected["score"] * max(0.25, q_fft) * sat_penalty)
        return {
            "status": "ok",
            "score": score,
            "bpm_peaks": detected["bpm"],
            "bpm_fft": bpm_fft,
            "fft_quality": q_fft,
            "saturation_fraction": sat,
            "raw_range": raw_range,
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "score": 0.0,
        }


def pick_reference_channel(df: pd.DataFrame, t_s: np.ndarray, fs: float, candidates: list[str]) -> tuple[str, dict[str, Any]]:
    details = {}
    ranked = []
    for channel in candidates:
        if channel not in df.columns:
            continue
        raw = df[channel].to_numpy(dtype=float)
        info = score_reference_channel(raw, t_s, fs)
        details[channel] = info
        if info["status"] == "ok":
            ranked.append((float(info["score"]), channel))
    if not ranked:
        raise RuntimeError(f"Could not score any reference channel from {candidates}.")
    ranked.sort(reverse=True)
    return ranked[0][1], details


def beatwise_channel_features(
    raw: np.ndarray,
    peak_idx: np.ndarray,
    t_s: np.ndarray,
    saturation_threshold: float,
) -> dict[str, np.ndarray]:
    n_beats = max(0, len(peak_idx) - 1)
    beat_time = np.full(n_beats, np.nan, dtype=float)
    ac = np.full(n_beats, np.nan, dtype=float)
    dc = np.full(n_beats, np.nan, dtype=float)
    acdc = np.full(n_beats, np.nan, dtype=float)
    sat = np.full(n_beats, np.nan, dtype=float)
    p5 = np.full(n_beats, np.nan, dtype=float)
    p95 = np.full(n_beats, np.nan, dtype=float)

    for i in range(n_beats):
        a = int(peak_idx[i])
        b = int(peak_idx[i + 1])
        if b <= a + 2:
            continue
        window = np.asarray(raw[a:b], dtype=float)
        finite = window[np.isfinite(window)]
        if len(finite) < 3:
            continue
        beat_time[i] = float(0.5 * (t_s[a] + t_s[b]))
        p5[i] = float(np.percentile(finite, 5))
        p95[i] = float(np.percentile(finite, 95))
        ac[i] = p95[i] - p5[i]
        dc[i] = float(np.median(finite))
        acdc[i] = ac[i] / max(dc[i], 1e-9)
        sat[i] = saturation_fraction(finite, saturation_threshold)

    return {
        "beat_time_s": beat_time,
        "ac": ac,
        "dc": dc,
        "acdc": acdc,
        "saturation_fraction": sat,
        "p5": p5,
        "p95": p95,
    }


def heuristic_spo2_from_ratio(
    ratio: float,
    anchor_ratio: float | None,
    anchor_pct: float | None,
    slope: float,
    clip_min: float,
    clip_max: float,
) -> tuple[float | None, str | None]:
    if anchor_ratio is None or anchor_pct is None or not np.isfinite(ratio):
        return None, None
    spo2 = float(np.clip(anchor_pct - slope * (ratio - anchor_ratio), clip_min, clip_max))
    intercept = anchor_pct + slope * anchor_ratio
    formula = f"SpO2_est = clip({intercept:.6f} - {slope:.6f} * R, {clip_min:.1f}, {clip_max:.1f})"
    return spo2, formula


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze TCS3448 12-channel always-on F6/NIR lab CSV.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--red-channel", default=DEFAULT_RED_CHANNEL)
    parser.add_argument("--ir-channel", default=DEFAULT_IR_CHANNEL)
    parser.add_argument("--reference-channel", default="auto", help="auto, F6, NIR, or another channel from the CSV.")
    parser.add_argument("--drop-start-sec", type=float, default=3.0)
    parser.add_argument("--drop-end-sec", type=float, default=3.0)
    parser.add_argument("--saturation-threshold", type=float, default=SATURATION_THRESHOLD)
    parser.add_argument("--calibration-file", default=None, help="Optional cohort calibration JSON produced by calibrate_red_nir_12ch_cohort.py")
    parser.add_argument("--spo2-anchor-ratio", type=float, default=None)
    parser.add_argument("--spo2-anchor-pct", type=float, default=None)
    parser.add_argument("--spo2-slope", type=float, default=None)
    parser.add_argument("--spo2-clip-min", type=float, default=70.0)
    parser.add_argument("--spo2-clip-max", type=float, default=100.0)
    parser.add_argument("--validation-window-s", type=float, default=10.0, help="Centered window around each ground-truth snapshot used for live-metric validation.")
    parser.add_argument("--resp-target-brpm", type=float, default=None, help="Optional expected respiratory rate for controlled-breathing runs.")
    parser.add_argument("--spo2-reference-pct-for-this-run", type=float, default=None, help="If provided, saves the current median R as a suggested one-point anchor for this reference SpO2.")
    args = parser.parse_args()
    calibration = load_red_nir_12ch_calibration(args.calibration_file)
    (
        args.spo2_anchor_ratio,
        args.spo2_anchor_pct,
        args.spo2_slope,
        args.spo2_clip_min,
        args.spo2_clip_max,
    ) = resolve_spo2_params(
        calibration,
        anchor_ratio=args.spo2_anchor_ratio,
        anchor_pct=args.spo2_anchor_pct,
        slope=args.spo2_slope,
    )

    input_path = Path(args.input)
    df = pd.read_csv(input_path)
    for channel in (args.red_channel, args.ir_channel):
        if channel not in df.columns:
            raise RuntimeError(f"{channel} not found in CSV columns: {list(df.columns)}")

    if "us" in df.columns:
        t_s = (df["us"].to_numpy(dtype=float) - float(df["us"].iloc[0])) / 1_000_000.0
    elif "ms" in df.columns:
        t_s = (df["ms"].to_numpy(dtype=float) - float(df["ms"].iloc[0])) / 1000.0
    else:
        raise RuntimeError("CSV must contain us or ms.")

    trim = np.ones(len(t_s), dtype=bool)
    if args.drop_start_sec > 0 or args.drop_end_sec > 0:
        trim = (t_s >= args.drop_start_sec) & (t_s <= (t_s[-1] - args.drop_end_sec))
    df_trim = df.loc[trim].reset_index(drop=True)
    t_trim = t_s[trim]
    t_trim = t_trim - t_trim[0]

    if len(t_trim) < 30:
        raise RuntimeError("Too few samples after trimming.")

    fs = estimate_fs(t_trim)
    if fs is None:
        raise RuntimeError("Could not estimate sampling rate.")

    signal_cols = [c for c in df_trim.columns if c not in {"pc_time_s", "ms", "us", "astatus"}]
    if args.reference_channel == "auto":
        candidates = [args.red_channel, args.ir_channel, *signal_cols]
        seen = set()
        candidates = [c for c in candidates if not (c in seen or seen.add(c))]
        reference_channel, reference_details = pick_reference_channel(df_trim, t_trim, fs, candidates)
    else:
        reference_channel = args.reference_channel
        if reference_channel not in df_trim.columns:
            raise RuntimeError(f"Reference channel {reference_channel} not found in CSV.")
        reference_details = {
            reference_channel: score_reference_channel(df_trim[reference_channel].to_numpy(dtype=float), t_trim, fs)
        }

    reference_raw = df_trim[reference_channel].to_numpy(dtype=float)
    detected = detect_reference_peaks(reference_raw, t_trim, fs)
    peak_idx = np.asarray(detected["peaks"], dtype=int)

    red_raw = df_trim[args.red_channel].to_numpy(dtype=float)
    ir_raw = df_trim[args.ir_channel].to_numpy(dtype=float)
    red = beatwise_channel_features(red_raw, peak_idx, t_trim, args.saturation_threshold)
    ir = beatwise_channel_features(ir_raw, peak_idx, t_trim, args.saturation_threshold)

    ratio = red["acdc"] / np.maximum(ir["acdc"], 1e-12)
    basic_quality = (
        np.isfinite(red["acdc"])
        & np.isfinite(ir["acdc"])
        & np.isfinite(ratio)
        & (red["dc"] >= MIN_DC_COUNTS)
        & (ir["dc"] >= MIN_DC_COUNTS)
        & (red["acdc"] >= MIN_ACDC)
        & (ir["acdc"] >= MIN_ACDC)
        & (red["saturation_fraction"] <= 0.10)
        & (ir["saturation_fraction"] <= 0.10)
        & (ratio > 0.0)
    )
    ratio_med_pre = safe_median(ratio[basic_quality])
    ratio_inlier = robust_inlier_mask(
        ratio,
        zmax=4.0,
        min_scale=max(0.03, 0.10 * abs(ratio_med_pre) if np.isfinite(ratio_med_pre) else 0.03),
    )
    quality = basic_quality & ratio_inlier

    if int(np.sum(quality)) < 5:
        raise RuntimeError(f"Too few quality beats for F6/NIR ratio: {int(np.sum(quality))}")

    ratio_med = safe_median(ratio[quality])
    ratio_mean = safe_mean(ratio[quality])
    ratio_cv = robust_scale(ratio[quality], ratio_med, floor=max(0.01, 0.08 * abs(ratio_med))) / max(abs(ratio_med), 0.05)

    red_bpm_fft, red_fft_q = fft_bpm_quality(red_raw, fs)
    ir_bpm_fft, ir_fft_q = fft_bpm_quality(ir_raw, fs)
    spo2_est, spo2_formula = heuristic_spo2_from_ratio(
        ratio_med,
        args.spo2_anchor_ratio,
        args.spo2_anchor_pct,
        args.spo2_slope,
        args.spo2_clip_min,
        args.spo2_clip_max,
    )

    anchor_suggestion = None
    if args.spo2_reference_pct_for_this_run is not None:
        anchor_suggestion = {
            "spo2_anchor_ratio": ratio_med,
            "spo2_anchor_pct": float(args.spo2_reference_pct_for_this_run),
            "note": "This is a one-point personal calibration anchor from this run; it is not an independent validation.",
        }

    beat_table = pd.DataFrame({
        "beat_time_s": red["beat_time_s"],
        f"{args.red_channel}_ac": red["ac"],
        f"{args.red_channel}_dc": red["dc"],
        f"{args.red_channel}_acdc": red["acdc"],
        f"{args.red_channel}_saturation_fraction": red["saturation_fraction"],
        f"{args.ir_channel}_ac": ir["ac"],
        f"{args.ir_channel}_dc": ir["dc"],
        f"{args.ir_channel}_acdc": ir["acdc"],
        f"{args.ir_channel}_saturation_fraction": ir["saturation_fraction"],
        "ratio_of_ratios": ratio,
        "quality_ok": quality,
    })

    summary = {
        "input_csv": str(input_path.resolve()),
        "estimated_fs_hz": fs,
        "duration_s_after_trim": float(t_trim[-1] - t_trim[0]),
        "status_note": "F6/NIR always-on SpO2-style research analysis. Ratio-of-ratios is useful for calibration, but percent SpO2 requires device-specific calibration.",
        "red_channel": args.red_channel,
        "red_peak_wavelength_nm": 636 if args.red_channel == "F6" else None,
        "ir_channel": args.ir_channel,
        "ir_peak_wavelength_nm": 855 if args.ir_channel == "NIR" else None,
        "reference_channel": reference_channel,
        "reference_polarity": detected["polarity"],
        "reference_peak_bpm": detected["bpm"],
        "num_reference_peaks": int(len(peak_idx)),
        "num_beat_windows": int(len(ratio)),
        "num_quality_beats": int(np.sum(quality)),
        "quality_beat_fraction": float(np.sum(quality) / len(ratio)) if len(ratio) else np.nan,
        "red_saturation_fraction_total": saturation_fraction(red_raw, args.saturation_threshold),
        "ir_saturation_fraction_total": saturation_fraction(ir_raw, args.saturation_threshold),
        "red_raw_range_p95_p5": float(np.nanpercentile(red_raw, 95) - np.nanpercentile(red_raw, 5)),
        "ir_raw_range_p95_p5": float(np.nanpercentile(ir_raw, 95) - np.nanpercentile(ir_raw, 5)),
        "red_fft_bpm": red_bpm_fft,
        "red_fft_quality": red_fft_q,
        "ir_fft_bpm": ir_bpm_fft,
        "ir_fft_quality": ir_fft_q,
        "median_red_ac": safe_median(red["ac"][quality]),
        "median_red_dc": safe_median(red["dc"][quality]),
        "median_red_acdc": safe_median(red["acdc"][quality]),
        "median_ir_ac": safe_median(ir["ac"][quality]),
        "median_ir_dc": safe_median(ir["dc"][quality]),
        "median_ir_acdc": safe_median(ir["acdc"][quality]),
        "selected_ratio_of_ratios": ratio_med,
        "selected_ratio_of_ratios_mean": ratio_mean,
        "selected_ratio_dispersion_cv_robust": ratio_cv,
        "spo2_estimated_pct_available": spo2_est is not None,
        "spo2_estimated_pct": spo2_est,
        "spo2_formula": spo2_formula,
        "cohort_calibration_file": str(Path(args.calibration_file).resolve()) if args.calibration_file else None,
        "cohort_calibration_name": calibration.get("calibration_name"),
        "spo2_anchor_ratio": args.spo2_anchor_ratio,
        "spo2_anchor_pct": args.spo2_anchor_pct,
        "spo2_reference_anchor_suggestion": anchor_suggestion,
        "reference_channel_scores": reference_details,
    }

    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent / "analysis_red_nir_12ch"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "red_nir_12ch_summary.json", summary)
    beat_table.to_csv(output_dir / "red_nir_12ch_beats.csv", index=False)

    run_dir = infer_run_dir(input_path)
    if run_dir is not None:
        run_meta = load_json_if_exists(run_dir / "meta" / "run_metadata.json") or load_json_if_exists(run_dir / "run_metadata.json")
        subject_meta = load_json_if_exists(run_dir.parent / "subject_metadata.json")
        live_summary = summarize_live_metrics(run_dir)
        ground_truth_summary = summarize_ground_truth(run_dir)
        events_summary = summarize_events(run_dir)
        validation_snapshots_summary = summarize_validation_snapshots(run_dir, output_dir, window_s=args.validation_window_s)
        cfg = build_analysis_config(reference_channel, subject_meta, run_meta, args.spo2_anchor_ratio, args.spo2_anchor_pct, args.spo2_slope, args.resp_target_brpm)

        morphology_raw = reference_raw.copy()
        morphology_note = "reference channel used directly"
        if detected["polarity"] == "negative":
            morphology_raw = 2.0 * float(np.nanmedian(reference_raw)) - reference_raw
            morphology_note = "reference channel inverted around median because detected pulse polarity was negative"

        respiration_module = None
        stiffness_module = None
        tissue_module = None
        try:
            respiration_module = save_module_result(
                output_dir,
                "respiration",
                run_respiration(t_trim, morphology_raw, cfg),
            )
            respiration_module["summary"]["analysis_channel"] = reference_channel
            respiration_module["summary"]["polarity_note"] = morphology_note
            respiration_module["summary"]["calibrated_final_respiratory_rate_brpm"] = apply_rr_calibration(
                respiration_module["summary"].get("final_respiratory_rate_brpm"),
                calibration,
                run_meta.get("protocol"),
                source_feature="postrun_respiration_final_brpm",
            )
        except Exception as exc:
            respiration_module = error_module(exc)

        validation_snapshots_summary.update(
            add_postrun_respiration_to_validation_snapshots(output_dir, respiration_module)
        )

        try:
            stiffness_module = save_module_result(
                output_dir,
                "stiffness",
                run_stiffness(t_trim, morphology_raw, cfg),
            )
            stiffness_module["summary"]["analysis_channel"] = reference_channel
            stiffness_module["summary"]["polarity_note"] = morphology_note
        except Exception as exc:
            stiffness_module = error_module(exc)

        try:
            tissue_loaded = build_tissue_loaded_alias(df_trim, t_trim, input_path)
            tissue_module = save_module_result(
                output_dir,
                "tissue_oxygenation_trend",
                run_tissue_oxygenation_trend(tissue_loaded, cfg),
            )
            tissue_module["summary"]["input_channel_aliases"] = tissue_loaded.get("channel_aliases", {})
            tissue_module["summary"]["status_note_12ch"] = (
                "Computed on 12ch always-on channels using aliases expected by the diff-channel module; "
                "treat as relative within-run trend, not absolute tissue oxygen saturation."
            )
        except Exception as exc:
            tissue_module = error_module(exc)

        pulseox_spo2 = ground_truth_summary.get("pulseox_spo2_pct_last") or ground_truth_summary.get("pulseox_spo2_pct_median")
        pulseox_hr = ground_truth_summary.get("pulseox_hr_bpm_last") or ground_truth_summary.get("pulseox_hr_bpm_median")

        validation_summary = {
            "pulseox_spo2_pct": pulseox_spo2,
            "spo2_error_vs_pulseox_pct": (
                None if pulseox_spo2 is None or spo2_est is None else float(spo2_est - float(pulseox_spo2))
            ),
            "pulseox_hr_bpm": pulseox_hr,
            "hr_error_vs_pulseox_bpm": (
                None if pulseox_hr is None else float(detected["bpm"] - float(pulseox_hr))
            ),
        }

        master_summary = {
            "input_csv": str(input_path.resolve()),
            "run_dir": str(run_dir.resolve()),
            "data_mode": "red_nir_12ch_always_on",
            "selected_channel": reference_channel,
            "run_metadata": run_meta,
            "subject_metadata": subject_meta,
            "modules": {
                "heart_rate": {
                    "status": "ok",
                    "summary": {
                        "estimated_fs_hz": fs,
                        "reference_channel": reference_channel,
                        "reference_peak_bpm": detected["bpm"],
                        "red_fft_bpm": red_bpm_fft,
                        "ir_fft_bpm": ir_bpm_fft,
                        "num_reference_peaks": int(len(peak_idx)),
                    },
                },
                "spo2_style_estimation": {
                    "status": "ok",
                    "summary_file": str(output_dir / "red_nir_12ch_summary.json"),
                    "summary": summary,
                },
                "respiration": respiration_module,
                "tissue_oxygenation_trend": tissue_module,
                "stiffness": stiffness_module,
                "perfusion_proxy": {
                    "status": "ok",
                    "summary": {
                        "basis": f"{args.red_channel}/{args.ir_channel} beat windows",
                        "median_red_ac": safe_median(red["ac"][quality]),
                        "median_red_dc": safe_median(red["dc"][quality]),
                        "median_red_pi_proxy_pct": 100.0 * safe_median(red["ac"][quality]) / max(safe_median(red["dc"][quality]), 1e-9),
                        "median_ir_ac": safe_median(ir["ac"][quality]),
                        "median_ir_dc": safe_median(ir["dc"][quality]),
                        "median_ir_pi_proxy_pct": 100.0 * safe_median(ir["ac"][quality]) / max(safe_median(ir["dc"][quality]), 1e-9),
                        "median_red_pi_estimated_pct": apply_pi_calibration(
                            100.0 * safe_median(red["ac"][quality]) / max(safe_median(red["dc"][quality]), 1e-9),
                            calibration,
                        ),
                        "median_ir_pi_estimated_pct": apply_pi_calibration(
                            100.0 * safe_median(ir["ac"][quality]) / max(safe_median(ir["dc"][quality]), 1e-9),
                            calibration,
                        ),
                        "pi_calibration_source": calibration.get("perfusion_index", {}).get("source_channel"),
                    },
                },
                "live_metrics": {
                    "status": "ok" if live_summary else "missing",
                    "summary": live_summary,
                },
                "ground_truth": {
                    "status": "ok" if ground_truth_summary else "missing",
                    "summary": ground_truth_summary,
                },
                "events": {
                    "status": "ok" if events_summary else "missing",
                    "summary": events_summary,
                },
                "validation_against_references": {
                    "status": "ok" if pulseox_spo2 is not None or pulseox_hr is not None else "missing_ground_truth",
                    "summary": validation_summary,
                },
                "validation_snapshots": {
                    "status": "ok" if validation_snapshots_summary else "missing_ground_truth_or_live_metrics",
                    "summary": validation_snapshots_summary,
                },
            },
            "future_modules": {
                "hemoglobin_estimation": {
                    "status": "planned_ml",
                    "next_step": "Requires labeled hemoglobin reference values and many runs before regression is meaningful.",
                },
                "skin_tone_compensation": {
                    "status": "planned_ml",
                    "next_step": "Requires subject skin-tone metadata and cross-subject analysis of channel weights/errors.",
                },
            },
        }
        save_json(run_dir / "analysis" / "summary.json", master_summary)

    print(json.dumps(_to_json_safe(summary), indent=2, ensure_ascii=False))
    print(f"\nSaved: {output_dir / 'red_nir_12ch_summary.json'}")
    print(f"Saved: {output_dir / 'red_nir_12ch_beats.csv'}")
    if run_dir is not None:
        print(f"Saved: {run_dir / 'analysis' / 'summary.json'}")


if __name__ == "__main__":
    main()
