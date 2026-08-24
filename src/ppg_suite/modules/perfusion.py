from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd

from ..beats import detect_beats
from ..config import AnalysisConfig
from ..perfusion_support import build_perfusion_beat_series
from ..signal import estimate_fs_from_time, safe_median, safe_mean


def run_perfusion(t_s: np.ndarray, raw: np.ndarray, cfg: AnalysisConfig) -> Dict[str, Any]:
    fs = estimate_fs_from_time(t_s)
    if fs is None:
        raise RuntimeError("Could not estimate sampling rate.")
    beats = detect_beats(raw, t_s, fs, cfg)
    perf = build_perfusion_beat_series(beats, smooth_window=7)
    qmask = perf.quality_mask

    beat_times = perf.beat_times[qmask]
    pulse_amp = perf.pulse_amp[qmask]
    pulse_base = perf.pulse_base[qmask]
    pulse_peak = perf.pulse_peak[qmask]
    pi_proxy_pct = perf.pi_proxy_pct[qmask]

    if len(beat_times) < 5:
        raise RuntimeError("Too few perfusion-quality beats.")

    rr_s = np.diff(beat_times)
    rr_times = beat_times[1:]
    hr_bpm = 60.0 / rr_s if len(rr_s) else np.array([])

    segment_rows = []
    for name, t0, t1 in cfg.segments:
        m = (beat_times >= t0) & (beat_times < t1)
        m_hr = (rr_times >= t0) & (rr_times < t1)
        if int(np.sum(m)) < 3:
            continue
        segment_rows.append({
            "segment": name,
            "n_beats": int(np.sum(m)),
            "median_amp": safe_median(pulse_amp[m]),
            "median_base": safe_median(pulse_base[m]),
            "median_pi_pct": safe_median(pi_proxy_pct[m]),
            "mean_pi_pct": safe_mean(pi_proxy_pct[m]),
            "median_hr_bpm": safe_median(hr_bpm[m_hr]) if len(hr_bpm) else np.nan,
        })
    segments_df = pd.DataFrame(segment_rows)

    beats_df = pd.DataFrame({
        "beat_time_s": perf.beat_times,
        "pulse_amp": perf.pulse_amp,
        "pulse_base": perf.pulse_base,
        "pulse_peak": perf.pulse_peak,
        "pi_proxy_pct": perf.pi_proxy_pct,
        "quality_ok": perf.quality_mask,
        "pulse_amp_smooth": perf.amp_smooth,
        "pulse_base_smooth": perf.base_smooth,
        "pi_proxy_smooth": perf.pi_smooth,
    })

    summary = {
        "estimated_fs_hz": fs,
        "summary_basis": "quality_filtered_beats",
        "num_accepted_beats": perf.quality_summary["num_rr_clean_beats"],
        "num_quality_beats": perf.quality_summary["num_quality_beats"],
        "quality_beat_fraction": perf.quality_summary["quality_beat_fraction"],
        "amp_quality_floor_counts": perf.quality_summary["amp_quality_floor_counts"],
        "median_pulse_amplitude": safe_median(pulse_amp),
        "median_pulse_baseline": safe_median(pulse_base),
        "median_pi_proxy_pct": safe_median(pi_proxy_pct),
        "mean_pi_proxy_pct": safe_mean(pi_proxy_pct),
    }

    return {
        "summary": summary,
        "tables": {
            "perfusion_beats": beats_df,
            "perfusion_segments": segments_df,
        },
        "artifacts": {
            "beats": beats,
        }
    }
