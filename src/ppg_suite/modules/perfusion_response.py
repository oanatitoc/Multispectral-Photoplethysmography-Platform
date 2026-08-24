from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd

from ..beats import detect_beats
from ..config import AnalysisConfig
from ..perfusion_support import build_perfusion_beat_series
from ..signal import estimate_fs_from_time, safe_median, safe_mean


def _pct_change(x: float, ref: float) -> float:
    if np.isnan(x) or np.isnan(ref) or abs(ref) < 1e-12:
        return np.nan
    return 100.0 * (x - ref) / ref


def run_perfusion_response(t_s: np.ndarray, raw: np.ndarray, cfg: AnalysisConfig) -> Dict[str, Any]:
    fs = estimate_fs_from_time(t_s)
    if fs is None:
        raise RuntimeError("Could not estimate sampling rate.")
    beats = detect_beats(raw, t_s, fs, cfg)
    perf = build_perfusion_beat_series(beats, smooth_window=9)
    qmask = perf.quality_mask

    beat_times = perf.beat_times[qmask]
    pulse_amp = perf.pulse_amp[qmask]
    pulse_base = perf.pulse_base[qmask]
    pi_proxy_pct = perf.pi_proxy_pct[qmask]
    pi_smooth = perf.pi_smooth[qmask]

    if len(beat_times) < 20:
        raise RuntimeError(f"Too few perfusion-quality beats after cleaning: {len(beat_times)}")

    rr_acc_s = np.diff(beat_times)
    rr_acc_times = beat_times[1:]
    hr_bpm = 60.0 / rr_acc_s

    segment_rows = []
    seg_dict = {}
    for name, t0, t1 in cfg.segments:
        m = (beat_times >= t0) & (beat_times < t1)
        m_hr = (rr_acc_times >= t0) & (rr_acc_times < t1)
        seg = {
            "segment": name,
            "n_beats": int(np.sum(m)),
            "median_amp": safe_median(pulse_amp[m]),
            "median_base": safe_median(pulse_base[m]),
            "median_pi_pct": safe_median(pi_proxy_pct[m]),
            "mean_pi_pct": safe_mean(pi_proxy_pct[m]),
            "median_hr_bpm": safe_median(hr_bpm[m_hr]) if len(hr_bpm) else np.nan,
        }
        seg_dict[name] = seg
        segment_rows.append(seg)
    segment_df = pd.DataFrame(segment_rows)

    baseline_pi = seg_dict.get("baseline", {}).get("median_pi_pct", np.nan)
    task_pi = seg_dict.get("task", {}).get("median_pi_pct", np.nan)
    recovery_pi = seg_dict.get("recovery", {}).get("median_pi_pct", np.nan)

    baseline_amp = seg_dict.get("baseline", {}).get("median_amp", np.nan)
    task_amp = seg_dict.get("task", {}).get("median_amp", np.nan)
    recovery_amp = seg_dict.get("recovery", {}).get("median_amp", np.nan)

    baseline_hr = seg_dict.get("baseline", {}).get("median_hr_bpm", np.nan)
    task_hr = seg_dict.get("task", {}).get("median_hr_bpm", np.nan)
    recovery_hr = seg_dict.get("recovery", {}).get("median_hr_bpm", np.nan)

    recovery_fraction = np.nan
    if not np.isnan(recovery_pi) and not np.isnan(baseline_pi) and baseline_pi > 1e-12:
        recovery_fraction = recovery_pi / baseline_pi

    task_window = next(((t0, t1) for name, t0, t1 in cfg.segments if name == "task"), None)
    recovery_window = next(((t0, t1) for name, t0, t1 in cfg.segments if name == "recovery"), None)

    task_nadir_pi = np.nan
    time_to_nadir_s = np.nan
    time_to_recovery_90_s = np.nan

    if task_window is not None:
        t0, t1 = task_window
        m_task = (beat_times >= t0) & (beat_times < t1)
        if np.sum(m_task) > 0:
            local_times = beat_times[m_task]
            local_pi = pi_smooth[m_task]
            k = int(np.argmin(local_pi))
            task_nadir_pi = float(local_pi[k])
            time_to_nadir_s = float(local_times[k] - t0)

    if recovery_window is not None and not np.isnan(baseline_pi):
        r0, r1 = recovery_window
        target_90 = 0.90 * baseline_pi
        m_rec = (beat_times >= r0) & (beat_times < r1)
        if np.sum(m_rec) > 0:
            rec_times = beat_times[m_rec]
            rec_pi = pi_smooth[m_rec]
            idx = np.where(rec_pi >= target_90)[0]
            if len(idx) > 0:
                time_to_recovery_90_s = float(rec_times[idx[0]] - r0)

    response_metrics = {
        "baseline_pi_pct": baseline_pi,
        "task_pi_pct": task_pi,
        "recovery_pi_pct": recovery_pi,
        "delta_task_pi_pct": _pct_change(task_pi, baseline_pi),
        "delta_recovery_pi_pct": _pct_change(recovery_pi, baseline_pi),
        "baseline_amp": baseline_amp,
        "task_amp": task_amp,
        "recovery_amp": recovery_amp,
        "delta_task_amp_pct": _pct_change(task_amp, baseline_amp),
        "delta_recovery_amp_pct": _pct_change(recovery_amp, baseline_amp),
        "baseline_hr_bpm": baseline_hr,
        "task_hr_bpm": task_hr,
        "recovery_hr_bpm": recovery_hr,
        "delta_task_hr_pct": _pct_change(task_hr, baseline_hr),
        "delta_recovery_hr_pct": _pct_change(recovery_hr, baseline_hr),
        "task_nadir_pi_pct": task_nadir_pi,
        "time_to_nadir_s": time_to_nadir_s,
        "recovery_fraction": recovery_fraction,
        "time_to_recovery_90_s": time_to_recovery_90_s,
    }

    trends_df = pd.DataFrame({
        "beat_time_s": perf.beat_times,
        "pulse_amp": perf.pulse_amp,
        "pulse_base": perf.pulse_base,
        "pi_proxy_pct": perf.pi_proxy_pct,
        "quality_ok": perf.quality_mask,
        "amp_smooth": perf.amp_smooth,
        "base_smooth": perf.base_smooth,
        "pi_smooth": perf.pi_smooth,
    })

    return {
        "summary": {
            "estimated_fs_hz": fs,
            "summary_basis": "quality_filtered_beats",
            "num_accepted_beats": perf.quality_summary["num_rr_clean_beats"],
            "num_quality_beats": perf.quality_summary["num_quality_beats"],
            "quality_beat_fraction": perf.quality_summary["quality_beat_fraction"],
            "amp_quality_floor_counts": perf.quality_summary["amp_quality_floor_counts"],
            **response_metrics,
        },
        "tables": {
            "perfusion_response_segments": segment_df,
            "perfusion_response_metrics": pd.DataFrame([response_metrics]),
            "perfusion_response_trends": trends_df,
        },
        "artifacts": {
            "beats": beats,
        }
    }
