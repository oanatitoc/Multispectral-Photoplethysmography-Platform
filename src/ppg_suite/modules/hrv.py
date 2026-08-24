from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd

from ..beats import detect_beats
from ..config import AnalysisConfig
from ..signal import estimate_fs_from_time


def compute_hrv_metrics(nn_ms: np.ndarray) -> Dict[str, Any]:
    if len(nn_ms) < 3:
        raise RuntimeError("Too few accepted NN intervals after cleaning.")

    dnn = np.diff(nn_ms)
    metrics = {
        "num_intervals": int(len(nn_ms)),
        "mean_nn_ms": float(np.mean(nn_ms)),
        "mean_hr_bpm": float(60000.0 / np.mean(nn_ms)),
        "sdnn_ms": float(np.std(nn_ms, ddof=1)) if len(nn_ms) >= 2 else np.nan,
        "rmssd_ms": float(np.sqrt(np.mean(dnn ** 2))) if len(dnn) >= 1 else np.nan,
        "pnn50_percent": float(100.0 * np.mean(np.abs(dnn) > 50.0)) if len(dnn) >= 1 else np.nan,
    }
    return metrics


def run_hrv(t_s: np.ndarray, raw: np.ndarray, cfg: AnalysisConfig) -> Dict[str, Any]:
    fs = estimate_fs_from_time(t_s)
    if fs is None:
        raise RuntimeError("Could not estimate sampling rate.")

    beats = detect_beats(raw, t_s, fs, cfg)
    rr_ms = np.diff(beats.candidate_times) * 1000.0
    nn_ms = rr_ms[beats.accepted_rr_mask]
    metrics = compute_hrv_metrics(nn_ms)

    peak_times_s = beats.candidate_times
    beats_df = pd.DataFrame({
        "peak_time_s": peak_times_s[1:],
        "rr_ms": rr_ms,
        "accepted": beats.accepted_rr_mask,
    })

    return {
        "summary": {
            "estimated_fs_hz": fs,
            **metrics,
        },
        "tables": {
            "hrv_beats": beats_df,
        },
        "artifacts": {
            "beats": beats,
        }
    }
