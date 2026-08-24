from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from .beats import BeatSeries, detect_beats
from .config import AnalysisConfig
from .signal import estimate_fs_from_time, moving_average, safe_median


MIN_BASE_COUNTS = 10.0
MIN_ABS_AMP_COUNTS = 3.0
MIN_PI_PROXY_PCT = 0.02


@dataclass
class PerfusionBeatSeries:
    beat_times: np.ndarray
    pulse_amp: np.ndarray
    pulse_base: np.ndarray
    pulse_peak: np.ndarray
    pi_proxy_pct: np.ndarray
    quality_mask: np.ndarray
    amp_smooth: np.ndarray
    base_smooth: np.ndarray
    pi_smooth: np.ndarray
    quality_summary: Dict[str, Any]


def _robust_scale(x: np.ndarray, center: float, floor: float) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float(floor)
    mad = float(np.median(np.abs(x - center)))
    return float(max(1.4826 * mad, floor))


def _robust_inlier_mask(x: np.ndarray, zmax: float, min_scale: float) -> np.ndarray:
    x = np.asarray(x, float)
    mask = np.isfinite(x)
    out = np.zeros(len(x), dtype=bool)
    if not np.any(mask):
        return out
    center = float(np.median(x[mask]))
    scale = _robust_scale(x[mask], center, min_scale)
    out[mask] = np.abs(x[mask] - center) <= (zmax * scale)
    return out


def build_perfusion_beat_series(beats: BeatSeries, smooth_window: int = 7) -> PerfusionBeatSeries:
    beat_times = np.asarray(beats.beat_times, float)
    pulse_amp = np.asarray(beats.pulse_amp, float)
    pulse_base = np.asarray(beats.pulse_base, float)
    pulse_peak = np.asarray(beats.pulse_peak, float)
    pi_proxy_pct = 100.0 * pulse_amp / np.maximum(pulse_base, 1e-6)

    valid = (
        np.isfinite(beat_times)
        & np.isfinite(pulse_amp)
        & np.isfinite(pulse_base)
        & np.isfinite(pi_proxy_pct)
        & (pulse_base >= MIN_BASE_COUNTS)
        & (pulse_amp > 0.0)
        & (pi_proxy_pct >= MIN_PI_PROXY_PCT)
    )

    amp_med = safe_median(pulse_amp[valid])
    base_med = safe_median(pulse_base[valid])
    pi_med = safe_median(pi_proxy_pct[valid])

    amp_floor = max(MIN_ABS_AMP_COUNTS, 0.25 * amp_med) if np.isfinite(amp_med) else MIN_ABS_AMP_COUNTS
    amp_inlier = _robust_inlier_mask(pulse_amp, zmax=4.0, min_scale=max(2.0, 0.08 * abs(amp_med) if np.isfinite(amp_med) else 0.0))
    base_inlier = _robust_inlier_mask(pulse_base, zmax=5.0, min_scale=max(10.0, 0.03 * abs(base_med) if np.isfinite(base_med) else 0.0))
    pi_inlier = _robust_inlier_mask(pi_proxy_pct, zmax=4.0, min_scale=max(0.08, 0.10 * abs(pi_med) if np.isfinite(pi_med) else 0.0))

    quality_mask = valid & (pulse_amp >= amp_floor) & amp_inlier & base_inlier & pi_inlier

    min_quality_beats = max(8, min(12, len(beat_times) // 3 if len(beat_times) else 0))
    if int(np.sum(quality_mask)) < min_quality_beats:
        quality_mask = valid & (pulse_amp >= amp_floor) & amp_inlier & base_inlier
    if int(np.sum(quality_mask)) < min_quality_beats:
        quality_mask = valid

    amp_smooth = np.full(len(beat_times), np.nan, dtype=float)
    base_smooth = np.full(len(beat_times), np.nan, dtype=float)
    pi_smooth = np.full(len(beat_times), np.nan, dtype=float)

    if np.any(quality_mask):
        q_amp = pulse_amp[quality_mask]
        q_base = pulse_base[quality_mask]
        q_pi = pi_proxy_pct[quality_mask]
        amp_smooth[quality_mask] = moving_average(q_amp, smooth_window)
        base_smooth[quality_mask] = moving_average(q_base, smooth_window)
        pi_smooth[quality_mask] = moving_average(q_pi, smooth_window)

    quality_summary = {
        "num_rr_clean_beats": int(len(beat_times)),
        "num_quality_beats": int(np.sum(quality_mask)),
        "quality_beat_fraction": (
            float(np.sum(quality_mask) / len(beat_times))
            if len(beat_times)
            else np.nan
        ),
        "amp_quality_floor_counts": float(amp_floor),
        "raw_median_pi_proxy_pct": float(pi_med) if np.isfinite(pi_med) else np.nan,
    }

    return PerfusionBeatSeries(
        beat_times=beat_times,
        pulse_amp=pulse_amp,
        pulse_base=pulse_base,
        pulse_peak=pulse_peak,
        pi_proxy_pct=pi_proxy_pct,
        quality_mask=quality_mask,
        amp_smooth=amp_smooth,
        base_smooth=base_smooth,
        pi_smooth=pi_smooth,
        quality_summary=quality_summary,
    )


def score_perfusion_channel(t_s: np.ndarray, raw: np.ndarray, cfg: AnalysisConfig) -> Dict[str, Any]:
    fs = estimate_fs_from_time(t_s)
    if fs is None:
        raise RuntimeError("Could not estimate sampling rate.")

    beats = detect_beats(raw, t_s, fs, cfg)
    perf = build_perfusion_beat_series(beats, smooth_window=7)
    qmask = perf.quality_mask
    if int(np.sum(qmask)) < 5:
        raise RuntimeError("Too few perfusion-quality beats.")

    q_pi = perf.pi_proxy_pct[qmask]
    q_amp = perf.pulse_amp[qmask]
    score = float(np.percentile(q_pi, 25) * np.sqrt(len(q_pi)) * max(0.25, perf.quality_summary["quality_beat_fraction"]))

    return {
        "score": score,
        "estimated_fs_hz": fs,
        "num_candidate_beats": int(len(beats.candidate_times)),
        "num_rr_clean_beats": perf.quality_summary["num_rr_clean_beats"],
        "num_quality_beats": perf.quality_summary["num_quality_beats"],
        "quality_beat_fraction": perf.quality_summary["quality_beat_fraction"],
        "median_pi_proxy_pct": safe_median(q_pi),
        "median_pulse_amplitude": safe_median(q_amp),
    }


def pick_best_perfusion_channel(
    channel_data: Dict[str, Dict[str, np.ndarray]],
    cfg: AnalysisConfig,
    preferred_channel: str | None = None,
) -> Dict[str, Any]:
    details: Dict[str, Any] = {}
    ranked: list[tuple[float, str]] = []

    for channel_name, loaded in channel_data.items():
        try:
            info = score_perfusion_channel(loaded["t_s"], loaded["raw"], cfg)
            details[channel_name] = {"status": "ok", **info}
            ranked.append((float(info["score"]), channel_name))
        except Exception as exc:
            details[channel_name] = {
                "status": "error",
                "error": str(exc),
            }

    if not ranked:
        raise RuntimeError("Could not score any candidate perfusion channels.")

    ranked.sort(reverse=True)
    best_score, best_channel = ranked[0]

    if preferred_channel is not None and preferred_channel in details and details[preferred_channel].get("status") == "ok":
        pref_score = float(details[preferred_channel]["score"])
        if pref_score >= 0.95 * best_score:
            best_channel = preferred_channel
            best_score = pref_score

    return {
        "selected_channel": best_channel,
        "selected_score": best_score,
        "details": details,
    }
