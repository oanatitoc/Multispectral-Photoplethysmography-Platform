from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from ..beats import BeatSeries, detect_beats
from ..config import AnalysisConfig
from ..perfusion_support import build_perfusion_beat_series
from ..signal import estimate_fs_from_time, safe_mean, safe_median


IR_CHANNEL = "NIR_diff"
REFERENCE_CANDIDATES = (IR_CHANNEL, "FZ_diff", "FXL_diff", "FY_diff")
RED_CANDIDATES = (
    ("F6_diff", 636),
    ("FXL_diff", 596),
    ("F7_diff", 687),
    ("F8_diff", 748),
    ("FY_diff", 560),
)
MIN_HEURISTIC_NUM_QUALITY_BEATS = 25
MIN_HEURISTIC_QUALITY_FRACTION = 0.60
MIN_HEURISTIC_PAIR_SCORE = 4.0
MIN_HEURISTIC_RED_ACDC = 0.003
MIN_HEURISTIC_IR_ACDC = 0.005
MAX_HEURISTIC_RATIO_CV = 0.15


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
        raise RuntimeError("No diff channels available for SpO2-style analysis.")
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


def _pair_score(ratio: np.ndarray, quality_mask: np.ndarray) -> Tuple[float, float]:
    qratio = np.asarray(ratio[quality_mask], float)
    if len(qratio) == 0:
        return 0.0, np.nan
    center = safe_median(qratio)
    spread = _robust_scale(qratio, center, floor=max(0.01, 0.08 * abs(center) if np.isfinite(center) else 0.01))
    ratio_cv = spread / max(abs(center), 0.05)
    quality_fraction = float(np.sum(quality_mask) / len(ratio))
    score = float(np.sqrt(len(qratio)) * max(0.1, quality_fraction) / (1.0 + ratio_cv))
    return score, ratio_cv


def _heuristic_spo2_from_ratio(ratio: float, cfg: AnalysisConfig) -> Tuple[float | None, str | None]:
    if (
        cfg.spo2_anchor_ratio is None
        or cfg.spo2_anchor_pct is None
        or not np.isfinite(ratio)
    ):
        return None, None

    anchor_ratio = float(cfg.spo2_anchor_ratio)
    anchor_pct = float(cfg.spo2_anchor_pct)
    slope = float(cfg.spo2_linear_slope)
    clip_min = float(cfg.spo2_clip_min_pct)
    clip_max = float(cfg.spo2_clip_max_pct)

    spo2_est = anchor_pct - slope * (ratio - anchor_ratio)
    spo2_est = float(np.clip(spo2_est, clip_min, clip_max))
    intercept = anchor_pct + slope * anchor_ratio
    formula = f"SpO2_est = clip({intercept:.6f} - {slope:.6f} * R, {clip_min:.1f}, {clip_max:.1f})"
    return spo2_est, formula


def _assess_heuristic_gate(selected_row: Dict[str, Any], cfg: AnalysisConfig) -> Tuple[bool, str]:
    reasons: list[str] = []
    num_quality_beats = int(selected_row["num_quality_beats"])
    quality_fraction = float(selected_row["quality_beat_fraction"])
    pair_score = float(selected_row["pair_quality_score"])
    red_acdc = float(selected_row["median_red_acdc"])
    ir_acdc = float(selected_row["median_ir_acdc"])
    ratio = float(selected_row["median_ratio_of_ratios"])
    ratio_cv = float(selected_row["ratio_dispersion_cv_robust"])

    if num_quality_beats < MIN_HEURISTIC_NUM_QUALITY_BEATS:
        reasons.append(f"too few pair-quality beats ({num_quality_beats} < {MIN_HEURISTIC_NUM_QUALITY_BEATS})")
    if quality_fraction < MIN_HEURISTIC_QUALITY_FRACTION:
        reasons.append(f"pair quality fraction too low ({quality_fraction:.2f} < {MIN_HEURISTIC_QUALITY_FRACTION:.2f})")
    if pair_score < MIN_HEURISTIC_PAIR_SCORE:
        reasons.append(f"pair quality score too low ({pair_score:.2f} < {MIN_HEURISTIC_PAIR_SCORE:.2f})")
    if red_acdc < MIN_HEURISTIC_RED_ACDC:
        reasons.append(f"red-like AC/DC too small ({red_acdc:.6f} < {MIN_HEURISTIC_RED_ACDC:.6f})")
    if ir_acdc < MIN_HEURISTIC_IR_ACDC:
        reasons.append(f"IR AC/DC too small ({ir_acdc:.6f} < {MIN_HEURISTIC_IR_ACDC:.6f})")
    if ratio_cv > MAX_HEURISTIC_RATIO_CV:
        reasons.append(f"ratio dispersion too high ({ratio_cv:.2f} > {MAX_HEURISTIC_RATIO_CV:.2f})")

    if cfg.spo2_anchor_ratio is not None and np.isfinite(ratio):
        anchor_ratio = float(cfg.spo2_anchor_ratio)
        ratio_min = max(0.20, 0.50 * anchor_ratio)
        ratio_max = 1.50 * anchor_ratio
        if not (ratio_min <= ratio <= ratio_max):
            reasons.append(f"ratio out of calibrated band ({ratio:.3f} not in [{ratio_min:.3f}, {ratio_max:.3f}])")

    if reasons:
        return False, "; ".join(reasons)
    return True, "selected pair passed heuristic quality gate"


def run_spo2_style_estimation(loaded: Dict[str, Any], cfg: AnalysisConfig) -> Dict[str, Any]:
    df_trimmed = loaded.get("df_trimmed")
    if df_trimmed is None:
        raise RuntimeError("SpO2-style analysis requires df_trimmed in loaded signal data.")

    t_s = np.asarray(loaded["t_s"], dtype=float)
    fs = estimate_fs_from_time(t_s)
    if fs is None:
        raise RuntimeError("Could not estimate sampling rate.")

    if len(df_trimmed) != len(t_s):
        raise RuntimeError("Trimmed dataframe and time vector are misaligned.")

    reference_channel = _pick_reference_channel(df_trimmed, cfg.channel)
    reference_raw = df_trimmed[reference_channel].to_numpy(dtype=float)
    beats = detect_beats(reference_raw, t_s, fs, cfg)
    ref_perf = build_perfusion_beat_series(beats, smooth_window=7)

    peak_idx = np.asarray(beats.accepted_peak_idx, dtype=int)
    trough_idx = _accepted_trough_indices(beats)
    beat_times = np.asarray(ref_perf.beat_times, dtype=float)
    reference_quality = np.asarray(ref_perf.quality_mask, dtype=bool)

    if len(beat_times) < 5:
        raise RuntimeError("Too few accepted beats for SpO2-style analysis.")

    available_red_channels = [name for name, _ in RED_CANDIDATES if name in df_trimmed.columns]
    if IR_CHANNEL not in df_trimmed.columns:
        raise RuntimeError(f"{IR_CHANNEL} is required for SpO2-style analysis.")
    if not available_red_channels:
        raise RuntimeError("No red-like diff channels available for SpO2-style analysis.")

    beat_table = pd.DataFrame({
        "beat_time_s": beat_times,
        "reference_channel": reference_channel,
        "reference_quality_ok": reference_quality,
        "reference_peak_idx": peak_idx,
        "reference_trough_idx": trough_idx,
    })

    channel_landmarks: Dict[str, Dict[str, np.ndarray]] = {}
    tracked_channels = [IR_CHANNEL, *available_red_channels]
    for channel_name in tracked_channels:
        raw = df_trimmed[channel_name].to_numpy(dtype=float)
        amp, base, acdc, valid = _channel_beat_landmarks(raw, peak_idx, trough_idx)
        quality_mask = reference_quality & valid
        channel_landmarks[channel_name] = {
            "amp": amp,
            "base": base,
            "acdc": acdc,
            "quality": quality_mask,
        }
        beat_table[f"{channel_name}__amp"] = amp
        beat_table[f"{channel_name}__base"] = base
        beat_table[f"{channel_name}__acdc"] = acdc
        beat_table[f"{channel_name}__quality_ok"] = quality_mask

    pair_rows = []
    ir_acdc = channel_landmarks[IR_CHANNEL]["acdc"]
    ir_quality = channel_landmarks[IR_CHANNEL]["quality"]

    for red_channel, red_nm in RED_CANDIDATES:
        if red_channel not in channel_landmarks:
            continue

        red_acdc = channel_landmarks[red_channel]["acdc"]
        red_quality = channel_landmarks[red_channel]["quality"]
        ratio = red_acdc / np.maximum(ir_acdc, 1e-9)
        ratio_med = safe_median(ratio)
        ratio_inlier = _robust_inlier_mask(ratio, zmax=4.0, min_scale=max(0.03, 0.10 * abs(ratio_med) if np.isfinite(ratio_med) else 0.03))
        pair_quality = red_quality & ir_quality & ratio_inlier & np.isfinite(ratio) & (ratio > 0.0)
        if int(np.sum(pair_quality)) < 8:
            continue
        score, ratio_cv = _pair_score(ratio, pair_quality)

        pair_name = f"{red_channel}__vs__{IR_CHANNEL}"
        beat_table[f"{pair_name}__ratio"] = ratio
        beat_table[f"{pair_name}__quality_ok"] = pair_quality

        row = {
            "pair_name": pair_name,
            "red_channel": red_channel,
            "red_peak_wavelength_nm": red_nm,
            "ir_channel": IR_CHANNEL,
            "ir_peak_wavelength_nm": 855,
            "num_quality_beats": int(np.sum(pair_quality)),
            "quality_beat_fraction": float(np.sum(pair_quality) / len(beat_times)),
            "median_red_acdc": safe_median(red_acdc[pair_quality]),
            "median_ir_acdc": safe_median(ir_acdc[pair_quality]),
            "median_ratio_of_ratios": safe_median(ratio[pair_quality]),
            "mean_ratio_of_ratios": safe_mean(ratio[pair_quality]),
            "ratio_dispersion_cv_robust": ratio_cv,
            "pair_quality_score": score,
        }
        pair_rows.append(row)

    if not pair_rows:
        raise RuntimeError("Could not build any valid red/NIR SpO2-style pairs.")

    pairs_df = pd.DataFrame(pair_rows).sort_values(["pair_quality_score", "num_quality_beats"], ascending=[False, False]).reset_index(drop=True)
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

    pairs_df["selected"] = pairs_df["pair_name"] == selected_row["pair_name"]
    selected_ratio = float(selected_row["median_ratio_of_ratios"])
    heuristic_gate_ok, heuristic_quality_note = _assess_heuristic_gate(selected_row, cfg)
    heuristic_spo2_pct, heuristic_formula = _heuristic_spo2_from_ratio(selected_ratio, cfg)
    if not heuristic_gate_ok:
        heuristic_spo2_pct = None

    summary = {
        "estimated_fs_hz": fs,
        "summary_basis": "reference_quality_beats",
        "status_note": "Experimental multispectral SpO2-style proxy from white-LED reflectance; ratio-of-ratios are useful for calibration work but are not clinically calibrated SpO2 values.",
        "reference_channel": reference_channel,
        "ir_channel": IR_CHANNEL,
        "available_red_channels": available_red_channels,
        "num_reference_beats": int(len(beat_times)),
        "num_reference_quality_beats": int(np.sum(reference_quality)),
        "selected_pair": selected_row["pair_name"],
        "selected_red_channel": selected_row["red_channel"],
        "selected_red_peak_wavelength_nm": selected_row["red_peak_wavelength_nm"],
        "selected_ir_peak_wavelength_nm": selected_row["ir_peak_wavelength_nm"],
        "selected_num_quality_beats": selected_row["num_quality_beats"],
        "selected_quality_beat_fraction": selected_row["quality_beat_fraction"],
        "selected_ratio_of_ratios": selected_row["median_ratio_of_ratios"],
        "selected_ratio_of_ratios_mean": selected_row["mean_ratio_of_ratios"],
        "selected_ratio_dispersion_cv_robust": selected_row["ratio_dispersion_cv_robust"],
        "selected_pair_quality_score": selected_row["pair_quality_score"],
        "selected_median_red_acdc": selected_row["median_red_acdc"],
        "selected_median_ir_acdc": selected_row["median_ir_acdc"],
        "heuristic_quality_gate_passed": heuristic_gate_ok,
        "heuristic_quality_note": heuristic_quality_note,
        "heuristic_spo2_pct_available": heuristic_spo2_pct is not None,
        "heuristic_spo2_estimated_pct": heuristic_spo2_pct,
        "heuristic_formula": heuristic_formula,
        "heuristic_anchor_ratio": cfg.spo2_anchor_ratio,
        "heuristic_anchor_spo2_pct": cfg.spo2_anchor_pct,
        "heuristic_slope_pct_per_ratio": cfg.spo2_linear_slope,
    }

    return {
        "summary": summary,
        "tables": {
            "spo2_style_beats": beat_table,
            "spo2_style_pairs": pairs_df,
        },
        "artifacts": {
            "beats": beats,
        },
    }
