from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from ..beats import BeatSeries, detect_beats
from ..config import AnalysisConfig
from ..signal import estimate_fs_from_time, moving_average, safe_mean, safe_median


REFERENCE_CANDIDATES = ("FZ_diff", "NIR_diff", "FXL_diff", "FY_diff")
CHANNEL_WAVELENGTHS_NM = {
    "FZ_diff": 510,
    "FY_diff": 560,
    "FXL_diff": 596,
    "NIR_diff": 855,
    "VIS2_diff": None,
    "FD_diff": None,
}
RESEARCH_CHANNELS = ("FZ_diff", "FY_diff", "FXL_diff", "NIR_diff", "VIS2_diff", "FD_diff")
BALANCE_RED_CANDIDATES = (
    ("FXL_diff", 596),
    ("FY_diff", 560),
    ("FZ_diff", 510),
)
TREND_SMOOTHING_S = 8.0
MIN_LEVEL_COUNTS = 20.0


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


def _pick_reference_channel(df_trimmed: pd.DataFrame, requested_channel: str) -> str:
    candidates = [requested_channel, *REFERENCE_CANDIDATES]
    for name in candidates:
        if name in df_trimmed.columns:
            return name
    diff_cols = [col for col in df_trimmed.columns if col.endswith("_diff")]
    if not diff_cols:
        raise RuntimeError("No diff channels available for hemoglobin research analysis.")
    return diff_cols[0]


def _accepted_trough_indices(beats: BeatSeries) -> np.ndarray:
    trough_idx = []
    troughs = np.asarray(beats.troughs, dtype=int)
    for peak_idx in np.asarray(beats.accepted_peak_idx, dtype=int):
        prev = troughs[troughs < peak_idx]
        trough_idx.append(int(prev[-1]) if len(prev) else -1)
    return np.asarray(trough_idx, dtype=int)


def _channel_beat_landmarks(raw: np.ndarray, peak_idx: np.ndarray, trough_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    amp = np.full(len(peak_idx), np.nan, dtype=float)
    base = np.full(len(peak_idx), np.nan, dtype=float)
    acdc = np.full(len(peak_idx), np.nan, dtype=float)
    valid = np.zeros(len(peak_idx), dtype=bool)

    good = (
        (peak_idx >= 0)
        & (trough_idx >= 0)
        & (peak_idx < len(raw))
        & (trough_idx < len(raw))
        & (peak_idx > trough_idx)
    )
    if not np.any(good):
        return amp, base, acdc, valid

    amp[good] = raw[peak_idx[good]] - raw[trough_idx[good]]
    base[good] = raw[trough_idx[good]]
    acdc[good] = amp[good] / np.maximum(base[good], 1e-6)

    amp_med = safe_median(amp[good])
    base_med = safe_median(base[good])
    acdc_med = safe_median(acdc[good])

    amp_inlier = _robust_inlier_mask(amp, zmax=4.0, min_scale=max(2.0, 0.08 * abs(amp_med) if np.isfinite(amp_med) else 0.0))
    base_inlier = _robust_inlier_mask(base, zmax=5.0, min_scale=max(10.0, 0.03 * abs(base_med) if np.isfinite(base_med) else 0.0))
    acdc_inlier = _robust_inlier_mask(acdc, zmax=4.0, min_scale=max(0.0005, 0.10 * abs(acdc_med) if np.isfinite(acdc_med) else 0.0))

    valid = (
        good
        & np.isfinite(amp)
        & np.isfinite(base)
        & np.isfinite(acdc)
        & (base >= 10.0)
        & (amp > 0.0)
        & (acdc > 0.0)
        & amp_inlier
        & base_inlier
        & acdc_inlier
    )
    return amp, base, acdc, valid


def _trend_timebase(t_s: np.ndarray, trend_fs: float) -> np.ndarray:
    dt = 1.0 / float(trend_fs)
    return np.arange(float(t_s[0]), float(t_s[-1]) + 0.5 * dt, dt)


def _smooth_interp(t_s: np.ndarray, raw: np.ndarray, t_trend: np.ndarray, trend_fs: float, window_s: float) -> Tuple[np.ndarray, np.ndarray]:
    interp = np.interp(t_trend, t_s, raw)
    window = max(3, int(round(window_s * trend_fs)))
    smooth = moving_average(interp, window)
    return interp, smooth


def _segment_pct_change(x: float, ref: float) -> float:
    if not np.isfinite(x) or not np.isfinite(ref) or abs(ref) < 1e-12:
        return np.nan
    return 100.0 * (x - ref) / ref


def run_hemoglobin_estimation(loaded: Dict[str, Any], cfg: AnalysisConfig) -> Dict[str, Any]:
    df_trimmed = loaded.get("df_trimmed")
    if df_trimmed is None:
        raise RuntimeError("Hemoglobin estimation research requires df_trimmed in loaded signal data.")

    t_s = np.asarray(loaded["t_s"], dtype=float)
    fs = estimate_fs_from_time(t_s)
    if fs is None:
        raise RuntimeError("Could not estimate sampling rate.")

    available_channels = [name for name in RESEARCH_CHANNELS if name in df_trimmed.columns]
    if len(available_channels) < 4:
        raise RuntimeError("Too few spectral channels available for hemoglobin research analysis.")

    reference_channel = _pick_reference_channel(df_trimmed, cfg.channel)
    reference_raw = df_trimmed[reference_channel].to_numpy(dtype=float)
    beats = detect_beats(reference_raw, t_s, fs, cfg)
    beat_times = np.asarray(beats.beat_times, dtype=float)
    peak_idx = np.asarray(beats.accepted_peak_idx, dtype=int)
    trough_idx = _accepted_trough_indices(beats)

    if len(beat_times) < 8:
        raise RuntimeError("Too few accepted beats for hemoglobin research analysis.")

    t_trend = _trend_timebase(t_s, cfg.trend_fs)
    if len(t_trend) < 20:
        raise RuntimeError("Too few trend points for hemoglobin research analysis.")

    beat_df = pd.DataFrame({
        "beat_time_s": beat_times,
        "reference_channel": reference_channel,
        "reference_peak_idx": peak_idx,
        "reference_trough_idx": trough_idx,
    })
    trend_df = pd.DataFrame({"t_trend_s": t_trend})

    channel_beats: Dict[str, Dict[str, np.ndarray]] = {}
    channel_trends: Dict[str, np.ndarray] = {}

    for channel_name in available_channels:
        raw = df_trimmed[channel_name].to_numpy(dtype=float)
        amp, base, acdc, valid = _channel_beat_landmarks(raw, peak_idx, trough_idx)
        channel_beats[channel_name] = {
            "amp": amp,
            "base": base,
            "acdc": acdc,
            "quality": valid,
        }
        beat_df[f"{channel_name}__amp"] = amp
        beat_df[f"{channel_name}__base"] = base
        beat_df[f"{channel_name}__acdc"] = acdc
        beat_df[f"{channel_name}__quality_ok"] = valid

        interp, smooth = _smooth_interp(t_s, raw, t_trend, cfg.trend_fs, TREND_SMOOTHING_S)
        channel_trends[channel_name] = smooth
        trend_df[f"{channel_name}__interp"] = interp
        trend_df[f"{channel_name}__smooth"] = smooth

    finite_quality_counts = {name: int(np.sum(info["quality"])) for name, info in channel_beats.items()}
    balance_rows = []

    nir_base = channel_beats["NIR_diff"]["base"] if "NIR_diff" in channel_beats else None
    nir_acdc = channel_beats["NIR_diff"]["acdc"] if "NIR_diff" in channel_beats else None
    nir_quality = channel_beats["NIR_diff"]["quality"] if "NIR_diff" in channel_beats else None

    if nir_base is None:
        raise RuntimeError("NIR_diff is required for hemoglobin research analysis.")

    for red_channel, red_nm in BALANCE_RED_CANDIDATES:
        if red_channel not in channel_beats:
            continue
        red_base = channel_beats[red_channel]["base"]
        red_acdc = channel_beats[red_channel]["acdc"]
        red_quality = channel_beats[red_channel]["quality"]

        base_ratio = red_base / np.maximum(nir_base, 1e-9)
        log_ratio = np.log(np.maximum(red_base, 1e-9)) - np.log(np.maximum(nir_base, 1e-9))
        quality_mask = red_quality & nir_quality & np.isfinite(base_ratio) & (base_ratio > 0.0)
        ratio_med = safe_median(base_ratio[quality_mask])
        ratio_inlier = _robust_inlier_mask(
            base_ratio,
            zmax=4.0,
            min_scale=max(0.03, 0.10 * abs(ratio_med) if np.isfinite(ratio_med) else 0.03),
        )
        quality_mask &= ratio_inlier
        if int(np.sum(quality_mask)) < 8:
            continue

        ratio_cv = _robust_scale(base_ratio[quality_mask], safe_median(base_ratio[quality_mask]), floor=0.01) / max(abs(safe_median(base_ratio[quality_mask])), 0.05)
        quality_fraction = float(np.sum(quality_mask) / len(base_ratio))
        signal_term = np.log10(max(min(safe_median(red_base[quality_mask]), safe_median(nir_base[quality_mask])), 1.0) + 9.0)
        score = float(np.sqrt(np.sum(quality_mask)) * max(0.1, quality_fraction) * signal_term / (1.0 + ratio_cv))

        pair_name = f"{red_channel}__vs__NIR_diff"
        beat_df[f"{pair_name}__baseline_ratio"] = base_ratio
        beat_df[f"{pair_name}__log_ratio"] = log_ratio
        beat_df[f"{pair_name}__quality_ok"] = quality_mask

        balance_rows.append({
            "pair_name": pair_name,
            "red_channel": red_channel,
            "red_peak_wavelength_nm": red_nm,
            "num_quality_beats": int(np.sum(quality_mask)),
            "quality_beat_fraction": quality_fraction,
            "median_red_base": safe_median(red_base[quality_mask]),
            "median_nir_base": safe_median(nir_base[quality_mask]),
            "median_red_acdc": safe_median(red_acdc[quality_mask]),
            "median_nir_acdc": safe_median(nir_acdc[quality_mask]),
            "median_red_over_nir_base_ratio": safe_median(base_ratio[quality_mask]),
            "mean_red_over_nir_base_ratio": safe_mean(base_ratio[quality_mask]),
            "median_log_red_over_nir_base": safe_median(log_ratio[quality_mask]),
            "ratio_dispersion_cv_robust": ratio_cv,
            "pair_quality_score": score,
        })

    if not balance_rows:
        raise RuntimeError("Could not build any valid hemoglobin research balance pairs.")

    balance_df = pd.DataFrame(balance_rows).sort_values(["pair_quality_score", "num_quality_beats"], ascending=[False, False]).reset_index(drop=True)
    selected_balance = balance_df.iloc[0].to_dict()
    balance_df["selected"] = balance_df["pair_name"] == selected_balance["pair_name"]

    smooth_sum = np.zeros(len(t_trend), dtype=float)
    for channel_name in available_channels:
        smooth_sum += np.maximum(channel_trends[channel_name], 0.0)
    smooth_sum = np.maximum(smooth_sum, 1e-9)

    channel_fractions = {}
    for channel_name in available_channels:
        frac = channel_trends[channel_name] / smooth_sum
        channel_fractions[channel_name] = frac
        trend_df[f"{channel_name}__fraction"] = frac

    selected_red_channel = str(selected_balance["red_channel"])
    selected_red_smooth = channel_trends[selected_red_channel]
    nir_smooth = channel_trends["NIR_diff"]
    selected_red_over_nir_trend = selected_red_smooth / np.maximum(nir_smooth, 1e-9)
    selected_hb_proxy = np.log(np.maximum(selected_red_smooth, 1e-9)) - np.log(np.maximum(nir_smooth, 1e-9))
    hb_proxy_quality = (
        np.isfinite(selected_hb_proxy)
        & np.isfinite(selected_red_over_nir_trend)
        & (selected_red_smooth >= MIN_LEVEL_COUNTS)
        & (nir_smooth >= MIN_LEVEL_COUNTS)
    )
    trend_df["selected_red_over_nir_trend_ratio"] = selected_red_over_nir_trend
    trend_df["selected_hb_balance_log_proxy"] = selected_hb_proxy
    trend_df["selected_hb_quality_ok"] = hb_proxy_quality

    segment_rows = []
    seg_dict: Dict[str, Dict[str, float]] = {}
    for name, t0, t1 in cfg.segments:
        m = (t_trend >= t0) & (t_trend < t1) & hb_proxy_quality
        seg = {
            "segment": name,
            "n_points": int(np.sum(m)),
            "median_selected_hb_balance_log_proxy": safe_median(selected_hb_proxy[m]),
            "median_selected_red_over_nir_trend_ratio": safe_median(selected_red_over_nir_trend[m]),
            "median_selected_red_fraction": safe_median(channel_fractions[selected_red_channel][m]),
            "median_nir_fraction": safe_median(channel_fractions["NIR_diff"][m]),
        }
        seg_dict[name] = seg
        segment_rows.append(seg)
    segments_df = pd.DataFrame(segment_rows)

    baseline_proxy = seg_dict.get("baseline", {}).get("median_selected_hb_balance_log_proxy", np.nan)
    task_proxy = seg_dict.get("task", {}).get("median_selected_hb_balance_log_proxy", np.nan)
    recovery_proxy = seg_dict.get("recovery", {}).get("median_selected_hb_balance_log_proxy", np.nan)

    summary = {
        "estimated_fs_hz": fs,
        "summary_basis": "reference_quality_beats_and_slow_trend",
        "status_note": "Exploratory hemoglobin-related spectral feature pack from white-LED reflectance; useful for regression research, not calibrated hemoglobin concentration.",
        "reference_channel": reference_channel,
        "available_channels": available_channels,
        "channel_quality_counts": finite_quality_counts,
        "num_reference_beats": int(len(beat_times)),
        "selected_balance_pair": selected_balance["pair_name"],
        "selected_red_channel": selected_red_channel,
        "selected_red_peak_wavelength_nm": CHANNEL_WAVELENGTHS_NM.get(selected_red_channel),
        "selected_num_quality_beats": selected_balance["num_quality_beats"],
        "selected_quality_beat_fraction": selected_balance["quality_beat_fraction"],
        "selected_pair_quality_score": selected_balance["pair_quality_score"],
        "selected_median_red_base": selected_balance["median_red_base"],
        "selected_median_nir_base": selected_balance["median_nir_base"],
        "selected_median_red_acdc": selected_balance["median_red_acdc"],
        "selected_median_nir_acdc": selected_balance["median_nir_acdc"],
        "selected_median_red_over_nir_base_ratio": selected_balance["median_red_over_nir_base_ratio"],
        "selected_mean_red_over_nir_base_ratio": selected_balance["mean_red_over_nir_base_ratio"],
        "selected_median_log_red_over_nir_base": selected_balance["median_log_red_over_nir_base"],
        "selected_ratio_dispersion_cv_robust": selected_balance["ratio_dispersion_cv_robust"],
        "trend_fs_hz": cfg.trend_fs,
        "trend_smoothing_window_s": TREND_SMOOTHING_S,
        "median_selected_hb_balance_log_proxy": safe_median(selected_hb_proxy[hb_proxy_quality]),
        "mean_selected_hb_balance_log_proxy": safe_mean(selected_hb_proxy[hb_proxy_quality]),
        "baseline_selected_hb_balance_log_proxy": baseline_proxy,
        "task_selected_hb_balance_log_proxy": task_proxy,
        "recovery_selected_hb_balance_log_proxy": recovery_proxy,
        "delta_task_selected_hb_balance_log_proxy_pct": _segment_pct_change(task_proxy, baseline_proxy),
        "delta_recovery_selected_hb_balance_log_proxy_pct": _segment_pct_change(recovery_proxy, baseline_proxy),
        "median_selected_red_fraction": safe_median(channel_fractions[selected_red_channel][hb_proxy_quality]),
        "median_nir_fraction": safe_median(channel_fractions["NIR_diff"][hb_proxy_quality]),
        "median_fy_fraction": safe_median(channel_fractions["FY_diff"][hb_proxy_quality]) if "FY_diff" in channel_fractions else np.nan,
        "median_fxl_fraction": safe_median(channel_fractions["FXL_diff"][hb_proxy_quality]) if "FXL_diff" in channel_fractions else np.nan,
        "median_fz_fraction": safe_median(channel_fractions["FZ_diff"][hb_proxy_quality]) if "FZ_diff" in channel_fractions else np.nan,
        "estimated_hemoglobin_g_dl_available": False,
        "estimated_hemoglobin_g_dl": None,
        "model_ready_note": "Use hemoglobin_research_beats.csv, hemoglobin_research_balance_pairs.csv, and hemoglobin_research_trend.csv as candidate features for supervised regression against lab or reference Hb values.",
    }

    return {
        "summary": summary,
        "tables": {
            "hemoglobin_research_beats": beat_df,
            "hemoglobin_research_balance_pairs": balance_df,
            "hemoglobin_research_segments": segments_df,
            "hemoglobin_research_trend": trend_df,
        },
        "artifacts": {
            "beats": beats,
        },
    }
