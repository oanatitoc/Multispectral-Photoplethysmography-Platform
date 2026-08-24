from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.signal import find_peaks

from .config import AnalysisConfig
from .signal import butter_bandpass


@dataclass
class BeatSeries:
    cardiac: np.ndarray
    peaks: np.ndarray
    troughs: np.ndarray
    candidate_times: np.ndarray
    candidate_amp: np.ndarray
    candidate_base: np.ndarray
    candidate_peak: np.ndarray
    candidate_peak_idx: np.ndarray
    accepted_rr_mask: np.ndarray
    accepted_beats_mask: np.ndarray
    beat_times: np.ndarray
    pulse_amp: np.ndarray
    pulse_base: np.ndarray
    pulse_peak: np.ndarray
    accepted_peak_idx: np.ndarray
    rejected_peak_idx: np.ndarray


def clean_rr_series(rr_ms: np.ndarray, min_rr_ms: float, max_rr_ms: float, tol: float) -> np.ndarray:
    accepted = np.zeros(len(rr_ms), dtype=bool)
    accepted_rr = []
    for i, rr in enumerate(rr_ms):
        if not (min_rr_ms <= rr <= max_rr_ms):
            continue
        if len(accepted_rr) >= 3:
            local_med = np.median(accepted_rr[-5:])
            if abs(rr - local_med) / local_med > tol:
                continue
        accepted[i] = True
        accepted_rr.append(rr)
    return accepted


def detect_beats(raw: np.ndarray, t_s: np.ndarray, fs: float, cfg: AnalysisConfig) -> BeatSeries:
    cardiac = butter_bandpass(raw, fs, cfg.heart_band[0], cfg.heart_band[1], order=2)
    if cardiac is None:
        raise RuntimeError("Cardiac filtering failed.")

    prominence = max(0.5, 0.35 * np.std(cardiac))
    min_peak_distance = max(1, int(0.35 * fs))
    peaks, _ = find_peaks(cardiac, distance=min_peak_distance, prominence=prominence)
    troughs, _ = find_peaks(-cardiac, distance=max(1, int(0.20 * fs)))
    if len(peaks) < 5 or len(troughs) < 5:
        raise RuntimeError("Too few peaks/troughs detected.")

    candidate_times = []
    candidate_amp = []
    candidate_base = []
    candidate_peak = []
    candidate_peak_idx = []

    for p in peaks:
        prev_tr = troughs[troughs < p]
        if len(prev_tr) == 0:
            continue
        tr = prev_tr[-1]
        amp = raw[p] - raw[tr]
        base = raw[tr]
        peakv = raw[p]
        if base <= 1 or amp <= 0:
            continue
        candidate_times.append(t_s[p])
        candidate_amp.append(amp)
        candidate_base.append(base)
        candidate_peak.append(peakv)
        candidate_peak_idx.append(p)

    candidate_times = np.asarray(candidate_times, dtype=float)
    candidate_amp = np.asarray(candidate_amp, dtype=float)
    candidate_base = np.asarray(candidate_base, dtype=float)
    candidate_peak = np.asarray(candidate_peak, dtype=float)
    candidate_peak_idx = np.asarray(candidate_peak_idx, dtype=int)

    if len(candidate_times) < 5:
        raise RuntimeError("Too few candidate beats.")

    rr_ms = np.diff(candidate_times) * 1000.0
    accepted_rr_mask = clean_rr_series(rr_ms, cfg.min_rr_ms, cfg.max_rr_ms, cfg.rr_local_median_tol)

    accepted_beats_mask = np.zeros(len(candidate_times), dtype=bool)
    accepted_beats_mask[0] = False
    accepted_beats_mask[1:] = accepted_rr_mask

    beat_times = candidate_times[accepted_beats_mask]
    pulse_amp = candidate_amp[accepted_beats_mask]
    pulse_base = candidate_base[accepted_beats_mask]
    pulse_peak = candidate_peak[accepted_beats_mask]
    accepted_peak_idx = candidate_peak_idx[accepted_beats_mask]
    rejected_peak_idx = candidate_peak_idx[~accepted_beats_mask]

    if len(beat_times) < 3:
        raise RuntimeError("Too few accepted beats after cleaning.")

    return BeatSeries(
        cardiac=cardiac,
        peaks=peaks,
        troughs=troughs,
        candidate_times=candidate_times,
        candidate_amp=candidate_amp,
        candidate_base=candidate_base,
        candidate_peak=candidate_peak,
        candidate_peak_idx=candidate_peak_idx,
        accepted_rr_mask=accepted_rr_mask,
        accepted_beats_mask=accepted_beats_mask,
        beat_times=beat_times,
        pulse_amp=pulse_amp,
        pulse_base=pulse_base,
        pulse_peak=pulse_peak,
        accepted_peak_idx=accepted_peak_idx,
        rejected_peak_idx=rejected_peak_idx,
    )
