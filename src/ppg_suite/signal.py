from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, welch


def estimate_fs_from_time(t_s: np.ndarray) -> Optional[float]:
    if len(t_s) < 5:
        return None
    dt = np.diff(t_s)
    dt = dt[(dt > 0.005) & (dt < 0.2)]
    if len(dt) < 3:
        return None
    return float(1.0 / np.median(dt))


def butter_bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 2) -> Optional[np.ndarray]:
    if fs is None or len(x) < max(20, int(fs * 2)):
        return None
    nyq = 0.5 * fs
    lo_n = lo / nyq
    hi_n = hi / nyq
    if hi_n >= 1.0:
        hi_n = 0.99
    if lo_n <= 0.0 or lo_n >= hi_n:
        return None
    b, a = butter(order, [lo_n, hi_n], btype="band")
    return filtfilt(b, a, x)


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    x = np.asarray(x, float)
    if len(x) == 0:
        return x
    w = int(min(max(1, w), len(x)))
    return np.convolve(x, np.ones(w) / w, mode="same")


def safe_median(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else np.nan


def safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else np.nan


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    s = np.std(x)
    if s < 1e-12:
        return np.zeros_like(x)
    return (x - np.mean(x)) / s


def _quadratic_peak_interp(x0: float, x1: float, x2: float) -> float:
    denom = (x0 - 2.0 * x1 + x2)
    if abs(denom) < 1e-18:
        return 0.0
    delta = 0.5 * (x0 - x2) / denom
    return float(np.clip(delta, -1.0, 1.0))


def spectral_peak_quality(x: np.ndarray, fs: float, band: Tuple[float, float]) -> Tuple[float | None, float, np.ndarray | None, np.ndarray | None]:
    if x is None or len(x) < int(fs * 20):
        return None, 0.0, None, None
    nperseg = min(len(x), int(fs * 64))
    nfft = max(int(8 * nperseg), nperseg)
    f, pxx = welch(x, fs=fs, nperseg=nperseg, nfft=nfft)
    mask = (f >= band[0]) & (f <= band[1])
    if not np.any(mask):
        return None, 0.0, f, pxx
    fb = f[mask]
    pb = pxx[mask]
    k = int(np.argmax(pb))
    peak_f = float(fb[k])
    if 0 < k < len(fb) - 1:
        delta = _quadratic_peak_interp(float(pb[k - 1]), float(pb[k]), float(pb[k + 1]))
        peak_f = float(fb[k] + delta * (fb[1] - fb[0]))
    peak_p = float(pb[k])
    q = peak_p / (float(np.median(pb)) + 1e-12)
    return peak_f, q, f, pxx
