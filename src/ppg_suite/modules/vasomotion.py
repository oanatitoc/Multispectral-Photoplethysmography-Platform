from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd

from ..beats import detect_beats
from ..config import AnalysisConfig
from ..perfusion_support import build_perfusion_beat_series
from ..signal import butter_bandpass, estimate_fs_from_time, spectral_peak_quality, safe_median


def run_vasomotion(t_s: np.ndarray, raw: np.ndarray, cfg: AnalysisConfig) -> Dict[str, Any]:
    fs = estimate_fs_from_time(t_s)
    if fs is None:
        raise RuntimeError("Could not estimate sampling rate.")
    beats = detect_beats(raw, t_s, fs, cfg)
    perf = build_perfusion_beat_series(beats, smooth_window=7)
    qmask = perf.quality_mask

    beat_times = perf.beat_times[qmask]
    pulse_amp = perf.pulse_amp[qmask]
    pulse_base = perf.pulse_base[qmask]

    if len(beat_times) < 20:
        raise RuntimeError("Too few perfusion-quality beats.")

    pi_proxy_pct = perf.pi_proxy_pct[qmask]

    t_trend = np.arange(beat_times[0], beat_times[-1], 1.0 / cfg.trend_fs)
    amp_interp = np.interp(t_trend, beat_times, pulse_amp)
    pi_interp = np.interp(t_trend, beat_times, pi_proxy_pct)
    base_interp = np.interp(t_trend, beat_times, pulse_base)

    amp_vaso = butter_bandpass(amp_interp, cfg.trend_fs, cfg.vaso_band[0], cfg.vaso_band[1], order=2)
    pi_vaso = butter_bandpass(pi_interp, cfg.trend_fs, cfg.vaso_band[0], cfg.vaso_band[1], order=2)
    base_vaso = butter_bandpass(base_interp, cfg.trend_fs, cfg.vaso_band[0], cfg.vaso_band[1], order=2)

    amp_f, amp_q, _, _ = spectral_peak_quality(amp_vaso, cfg.trend_fs, cfg.vaso_band)
    pi_f, pi_q, _, _ = spectral_peak_quality(pi_vaso, cfg.trend_fs, cfg.vaso_band)
    base_f, base_q, _, _ = spectral_peak_quality(base_vaso, cfg.trend_fs, cfg.vaso_band)

    rows = []
    for name, t0, t1 in cfg.segments:
        m = (beat_times >= t0) & (beat_times < t1)
        if np.sum(m) < 5:
            continue
        rows.append({
            "segment": name,
            "n_beats": int(np.sum(m)),
            "median_amp": safe_median(pulse_amp[m]),
            "median_base": safe_median(pulse_base[m]),
            "median_pi_pct": safe_median(pi_proxy_pct[m]),
        })
    segment_df = pd.DataFrame(rows)

    summary = {
        "estimated_fs_hz": fs,
        "summary_basis": "quality_filtered_beats",
        "num_accepted_beats": perf.quality_summary["num_rr_clean_beats"],
        "num_quality_beats": perf.quality_summary["num_quality_beats"],
        "quality_beat_fraction": perf.quality_summary["quality_beat_fraction"],
        "amp_quality_floor_counts": perf.quality_summary["amp_quality_floor_counts"],
        "amp_vasomotion_cpm": None if amp_f is None else amp_f * 60.0,
        "amp_vasomotion_q": amp_q,
        "pi_vasomotion_cpm": None if pi_f is None else pi_f * 60.0,
        "pi_vasomotion_q": pi_q,
        "base_vasomotion_cpm": None if base_f is None else base_f * 60.0,
        "base_vasomotion_q": base_q,
    }

    traces_df = pd.DataFrame({
        "t_trend_s": t_trend,
        "amp_interp": amp_interp,
        "pi_interp": pi_interp,
        "base_interp": base_interp,
        "amp_vaso": amp_vaso if amp_vaso is not None else np.nan,
        "pi_vaso": pi_vaso if pi_vaso is not None else np.nan,
        "base_vaso": base_vaso if base_vaso is not None else np.nan,
    })

    return {
        "summary": summary,
        "tables": {
            "vasomotion_segments": segment_df,
            "vasomotion_summary": pd.DataFrame([summary]),
            "vasomotion_traces": traces_df,
        },
        "artifacts": {
            "beats": beats,
        }
    }
