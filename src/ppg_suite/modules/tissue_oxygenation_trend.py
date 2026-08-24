from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from ..config import AnalysisConfig
from ..signal import estimate_fs_from_time, moving_average, safe_mean, safe_median


IR_CHANNEL = "NIR_diff"
RED_CANDIDATES = (
    ("F6_diff", 636),
    ("FXL_diff", 596),
    ("FY_diff", 560),
    ("F7_diff", 687),
    ("F8_diff", 748),
)
TREND_SMOOTHING_S = 8.0
PROXY_SMOOTHING_S = 10.0
MIN_TREND_LEVEL_COUNTS = 20.0
MIN_QUALITY_POINTS = 20


def _robust_scale(x: np.ndarray, center: float, floor: float) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float(floor)
    mad = float(np.median(np.abs(x - center)))
    return float(max(1.4826 * mad, floor))


def _robust_inlier_mask(x: np.ndarray, zmax: float, min_scale: float) -> np.ndarray:
    x = np.asarray(x, float)
    finite = np.isfinite(x)
    out = np.zeros(len(x), dtype=bool)
    if not np.any(finite):
        return out
    center = float(np.median(x[finite]))
    scale = _robust_scale(x[finite], center, min_scale)
    out[finite] = np.abs(x[finite] - center) <= (zmax * scale)
    return out


def _pct_change(x: float, ref: float) -> float:
    if not np.isfinite(x) or not np.isfinite(ref) or abs(ref) < 1e-12:
        return np.nan
    return 100.0 * (x - ref) / ref


def _trend_timebase(t_s: np.ndarray, trend_fs: float) -> np.ndarray:
    dt = 1.0 / float(trend_fs)
    return np.arange(float(t_s[0]), float(t_s[-1]) + 0.5 * dt, dt)


def _smooth_interp(t_s: np.ndarray, raw: np.ndarray, t_trend: np.ndarray, trend_fs: float, window_s: float) -> Tuple[np.ndarray, np.ndarray]:
    interp = np.interp(t_trend, t_s, raw)
    window = max(3, int(round(window_s * trend_fs)))
    smooth = moving_average(interp, window)
    return interp, smooth


def _segment_mask(t_s: np.ndarray, cfg: AnalysisConfig, segment_name: str) -> np.ndarray:
    for name, t0, t1 in cfg.segments:
        if name == segment_name:
            return (t_s >= t0) & (t_s < t1)
    return np.zeros(len(t_s), dtype=bool)


def _pair_score(ratio: np.ndarray, quality_mask: np.ndarray, red_level: np.ndarray, ir_level: np.ndarray) -> Tuple[float, float]:
    qratio = np.asarray(ratio[quality_mask], float)
    if len(qratio) == 0:
        return 0.0, np.nan
    center = safe_median(qratio)
    spread = _robust_scale(qratio, center, floor=max(0.01, 0.08 * abs(center) if np.isfinite(center) else 0.01))
    ratio_cv = spread / max(abs(center), 0.05)
    quality_fraction = float(np.sum(quality_mask) / len(ratio))
    signal_floor = min(safe_median(red_level[quality_mask]), safe_median(ir_level[quality_mask]))
    signal_term = np.log10(max(signal_floor, 1.0) + 9.0)
    score = float(np.sqrt(len(qratio)) * max(0.1, quality_fraction) * signal_term / (1.0 + ratio_cv))
    return score, ratio_cv


def run_tissue_oxygenation_trend(loaded: Dict[str, Any], cfg: AnalysisConfig) -> Dict[str, Any]:
    df_trimmed = loaded.get("df_trimmed")
    if df_trimmed is None:
        raise RuntimeError("Tissue oxygenation trend analysis requires df_trimmed in loaded signal data.")

    t_s = np.asarray(loaded["t_s"], dtype=float)
    fs = estimate_fs_from_time(t_s)
    if fs is None:
        raise RuntimeError("Could not estimate sampling rate.")
    if IR_CHANNEL not in df_trimmed.columns:
        raise RuntimeError(f"{IR_CHANNEL} is required for tissue oxygenation trend analysis.")

    available_red_channels = [name for name, _ in RED_CANDIDATES if name in df_trimmed.columns]
    if not available_red_channels:
        raise RuntimeError("No red-like channels available for tissue oxygenation trend analysis.")

    t_trend = _trend_timebase(t_s, cfg.trend_fs)
    if len(t_trend) < 20:
        raise RuntimeError("Too few trend points for tissue oxygenation trend analysis.")

    ir_raw = df_trimmed[IR_CHANNEL].to_numpy(dtype=float)
    ir_interp, ir_smooth = _smooth_interp(t_s, ir_raw, t_trend, cfg.trend_fs, TREND_SMOOTHING_S)

    traces_df = pd.DataFrame({
        "t_trend_s": t_trend,
        f"{IR_CHANNEL}__interp": ir_interp,
        f"{IR_CHANNEL}__smooth": ir_smooth,
    })

    pair_rows = []
    pair_data: Dict[str, Dict[str, np.ndarray]] = {}

    for red_channel, red_nm in RED_CANDIDATES:
        if red_channel not in df_trimmed.columns:
            continue

        red_raw = df_trimmed[red_channel].to_numpy(dtype=float)
        red_interp, red_smooth = _smooth_interp(t_s, red_raw, t_trend, cfg.trend_fs, TREND_SMOOTHING_S)
        ratio = ir_smooth / np.maximum(red_smooth, 1e-9)
        log_ratio = np.log(np.maximum(ir_smooth, 1e-9)) - np.log(np.maximum(red_smooth, 1e-9))

        valid = (
            np.isfinite(red_smooth)
            & np.isfinite(ir_smooth)
            & np.isfinite(ratio)
            & (red_smooth >= MIN_TREND_LEVEL_COUNTS)
            & (ir_smooth >= MIN_TREND_LEVEL_COUNTS)
            & (ratio > 0.0)
        )
        ratio_med = safe_median(ratio[valid])
        ratio_inlier = _robust_inlier_mask(
            ratio,
            zmax=4.0,
            min_scale=max(0.02, 0.10 * abs(ratio_med) if np.isfinite(ratio_med) else 0.02),
        )
        quality_mask = valid & ratio_inlier
        if int(np.sum(quality_mask)) < MIN_QUALITY_POINTS:
            quality_mask = valid
        if int(np.sum(quality_mask)) < 8:
            continue

        score, ratio_cv = _pair_score(ratio, quality_mask, red_smooth, ir_smooth)
        pair_name = f"{red_channel}__vs__{IR_CHANNEL}"

        traces_df[f"{red_channel}__interp"] = red_interp
        traces_df[f"{red_channel}__smooth"] = red_smooth
        traces_df[f"{pair_name}__nir_over_red_ratio"] = ratio
        traces_df[f"{pair_name}__log_balance"] = log_ratio
        traces_df[f"{pair_name}__quality_ok"] = quality_mask

        pair_data[pair_name] = {
            "red_interp": red_interp,
            "red_smooth": red_smooth,
            "ratio": ratio,
            "log_ratio": log_ratio,
            "quality_mask": quality_mask,
        }
        pair_rows.append({
            "pair_name": pair_name,
            "red_channel": red_channel,
            "red_peak_wavelength_nm": red_nm,
            "ir_channel": IR_CHANNEL,
            "ir_peak_wavelength_nm": 855,
            "num_quality_points": int(np.sum(quality_mask)),
            "quality_point_fraction": float(np.sum(quality_mask) / len(t_trend)),
            "median_red_level": safe_median(red_smooth[quality_mask]),
            "median_ir_level": safe_median(ir_smooth[quality_mask]),
            "median_nir_over_red_ratio": safe_median(ratio[quality_mask]),
            "mean_nir_over_red_ratio": safe_mean(ratio[quality_mask]),
            "ratio_dispersion_cv_robust": ratio_cv,
            "pair_quality_score": score,
        })

    if not pair_rows:
        raise RuntimeError("Could not build any valid red/NIR tissue oxygenation trend pairs.")

    pairs_df = pd.DataFrame(pair_rows).sort_values(["pair_quality_score", "num_quality_points"], ascending=[False, False]).reset_index(drop=True)
    best_score = float(pairs_df.iloc[0]["pair_quality_score"])
    selected_row = pairs_df.iloc[0].to_dict()

    for preferred_red, _ in RED_CANDIDATES:
        pref_rows = pairs_df[pairs_df["red_channel"] == preferred_red]
        if pref_rows.empty:
            continue
        pref = pref_rows.iloc[0].to_dict()
        if float(pref["pair_quality_score"]) >= 0.90 * best_score:
            selected_row = pref
            break

    selected_pair = str(selected_row["pair_name"])
    selected = pair_data[selected_pair]
    quality_mask = np.asarray(selected["quality_mask"], dtype=bool)
    ratio = np.asarray(selected["ratio"], dtype=float)
    red_smooth = np.asarray(selected["red_smooth"], dtype=float)

    baseline_mask = _segment_mask(t_trend, cfg, "baseline")
    baseline_quality = quality_mask & baseline_mask
    if int(np.sum(baseline_quality)) >= 5:
        baseline_ref_ratio = safe_median(ratio[baseline_quality])
        baseline_source = "baseline_segment"
    else:
        baseline_ref_ratio = safe_median(ratio[quality_mask])
        baseline_source = "global_quality_points"

    relative_proxy_pct = 100.0 * (ratio / np.maximum(baseline_ref_ratio, 1e-9) - 1.0)
    relative_proxy_smooth = np.full(len(t_trend), np.nan, dtype=float)
    if np.any(quality_mask):
        relative_proxy_smooth[quality_mask] = moving_average(
            relative_proxy_pct[quality_mask],
            max(3, int(round(PROXY_SMOOTHING_S * cfg.trend_fs))),
        )

    pairs_df["selected"] = pairs_df["pair_name"] == selected_pair
    traces_df["selected_red_channel"] = np.where(quality_mask, selected_row["red_channel"], selected_row["red_channel"])
    traces_df["selected_nir_over_red_ratio"] = ratio
    traces_df["selected_relative_oxygenation_proxy_pct"] = relative_proxy_pct
    traces_df["selected_relative_oxygenation_proxy_smooth_pct"] = relative_proxy_smooth
    traces_df["selected_quality_ok"] = quality_mask

    segment_rows = []
    seg_dict: Dict[str, Dict[str, float]] = {}
    for name, t0, t1 in cfg.segments:
        m = (t_trend >= t0) & (t_trend < t1) & quality_mask
        seg = {
            "segment": name,
            "n_points": int(np.sum(m)),
            "median_red_level": safe_median(red_smooth[m]),
            "median_ir_level": safe_median(ir_smooth[m]),
            "median_nir_over_red_ratio": safe_median(ratio[m]),
            "median_relative_oxygenation_proxy_pct": safe_median(relative_proxy_pct[m]),
        }
        seg_dict[name] = seg
        segment_rows.append(seg)
    segment_df = pd.DataFrame(segment_rows)

    q_times = t_trend[quality_mask]
    q_proxy = relative_proxy_pct[quality_mask]
    if len(q_proxy):
        k_min = int(np.argmin(q_proxy))
        k_max = int(np.argmax(q_proxy))
        min_proxy = float(q_proxy[k_min])
        max_proxy = float(q_proxy[k_max])
        time_to_min_s = float(q_times[k_min] - t_trend[0])
        time_to_max_s = float(q_times[k_max] - t_trend[0])
    else:
        min_proxy = np.nan
        max_proxy = np.nan
        time_to_min_s = np.nan
        time_to_max_s = np.nan

    baseline_ratio = seg_dict.get("baseline", {}).get("median_nir_over_red_ratio", np.nan)
    task_ratio = seg_dict.get("task", {}).get("median_nir_over_red_ratio", np.nan)
    recovery_ratio = seg_dict.get("recovery", {}).get("median_nir_over_red_ratio", np.nan)

    summary = {
        "estimated_fs_hz": fs,
        "summary_basis": "quality_resampled_trend_points",
        "status_note": "Relative tissue oxygenation trend proxy from slow red-like vs NIR reflectance balance; useful for within-run changes, not absolute StO2.",
        "ir_channel": IR_CHANNEL,
        "available_red_channels": available_red_channels,
        "selected_pair": selected_pair,
        "selected_red_channel": selected_row["red_channel"],
        "selected_red_peak_wavelength_nm": selected_row["red_peak_wavelength_nm"],
        "selected_ir_peak_wavelength_nm": selected_row["ir_peak_wavelength_nm"],
        "trend_fs_hz": cfg.trend_fs,
        "trend_smoothing_window_s": TREND_SMOOTHING_S,
        "proxy_definition": "100 * ((NIR/red ratio) / baseline_reference_ratio - 1)",
        "num_trend_points": int(len(t_trend)),
        "selected_num_quality_trend_points": selected_row["num_quality_points"],
        "selected_quality_trend_fraction": selected_row["quality_point_fraction"],
        "selected_pair_quality_score": selected_row["pair_quality_score"],
        "selected_median_red_level": selected_row["median_red_level"],
        "selected_median_ir_level": selected_row["median_ir_level"],
        "selected_median_nir_over_red_ratio": selected_row["median_nir_over_red_ratio"],
        "selected_mean_nir_over_red_ratio": selected_row["mean_nir_over_red_ratio"],
        "selected_ratio_dispersion_cv_robust": selected_row["ratio_dispersion_cv_robust"],
        "baseline_reference_source": baseline_source,
        "baseline_reference_nir_over_red_ratio": baseline_ref_ratio,
        "median_relative_oxygenation_proxy_pct": safe_median(q_proxy),
        "mean_relative_oxygenation_proxy_pct": safe_mean(q_proxy),
        "min_relative_oxygenation_proxy_pct": min_proxy,
        "max_relative_oxygenation_proxy_pct": max_proxy,
        "time_to_min_relative_oxygenation_s": time_to_min_s,
        "time_to_max_relative_oxygenation_s": time_to_max_s,
        "baseline_nir_over_red_ratio": baseline_ratio,
        "task_nir_over_red_ratio": task_ratio,
        "recovery_nir_over_red_ratio": recovery_ratio,
        "delta_task_nir_over_red_pct": _pct_change(task_ratio, baseline_ratio),
        "delta_recovery_nir_over_red_pct": _pct_change(recovery_ratio, baseline_ratio),
        "baseline_relative_oxygenation_proxy_pct": seg_dict.get("baseline", {}).get("median_relative_oxygenation_proxy_pct", np.nan),
        "task_relative_oxygenation_proxy_pct": seg_dict.get("task", {}).get("median_relative_oxygenation_proxy_pct", np.nan),
        "recovery_relative_oxygenation_proxy_pct": seg_dict.get("recovery", {}).get("median_relative_oxygenation_proxy_pct", np.nan),
    }

    return {
        "summary": summary,
        "tables": {
            "tissue_oxygenation_trend_pairs": pairs_df,
            "tissue_oxygenation_trend_segments": segment_df,
            "tissue_oxygenation_trend_trace": traces_df,
        },
    }
