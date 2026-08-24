from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, Optional

import numpy as np
from scipy.signal import find_peaks

from .beats import clean_rr_series
from .calibration import apply_pi_calibration, apply_rr_calibration
from .signal import butter_bandpass, estimate_fs_from_time, spectral_peak_quality


NON_SIGNAL_COLUMNS = {"pc_time_s", "ms", "us", "astatus"}
DEFAULT_HEART_BAND = (0.7, 4.0)
DEFAULT_RESP_BAND = (0.15, 0.35)
DEFAULT_RESP_FS = 4.0


@dataclass
class LiveMetrics:
    fs_hz: Optional[float] = None
    hr_bpm: Optional[float] = None
    hr_peak_bpm: Optional[float] = None
    hr_fft_bpm: Optional[float] = None
    mean_ibi_ms: Optional[float] = None
    rmssd_ms: Optional[float] = None
    sdnn_ms: Optional[float] = None
    respiratory_rate_brpm: Optional[float] = None
    spo2_estimated_pct: Optional[float] = None
    spo2_ratio: Optional[float] = None
    spo2_status: str = "calibrating"
    perfusion_proxy_pct: Optional[float] = None
    perfusion_index_pct: Optional[float] = None
    pulse_amplitude: Optional[float] = None
    signal_quality: Optional[float] = None
    best_channel: Optional[str] = None
    selected_channel: Optional[str] = None
    red_channel: Optional[str] = None
    ir_channel: Optional[str] = None
    pulse_polarity: Optional[int] = None
    saturation_fraction: Optional[float] = None
    artifact_flag: bool = True

    def to_row(self) -> Dict[str, object]:
        row = asdict(self)
        row["artifact_flag"] = int(self.artifact_flag)
        return row


def signal_columns(header: Iterable[str]) -> list[str]:
    return [col for col in header if col not in NON_SIGNAL_COLUMNS]


def preferred_preview_channel(columns: Iterable[str]) -> str:
    cols = list(columns)
    for preferred in ("F6", "NIR", "FZ_diff", "NIR_diff", "FZ", "VIS2_C2"):
        if preferred in cols:
            return preferred
    return cols[0] if cols else ""


def _finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def _percentile_range(x: np.ndarray, lo: float = 5.0, hi: float = 95.0) -> float:
    x = _finite(x)
    if len(x) == 0:
        return np.nan
    return float(np.nanpercentile(x, hi) - np.nanpercentile(x, lo))


def _fft_bpm_quality(x: np.ndarray, fs: Optional[float], band: tuple[float, float]) -> tuple[Optional[float], float]:
    if fs is None or len(x) < max(40, int(6 * fs)):
        return None, 0.0
    x = np.asarray(x, dtype=float)
    x = x - np.nanmean(x)
    if not np.all(np.isfinite(x)) or np.nanstd(x) < 1e-12:
        return None, 0.0

    win = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * win))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return None, 0.0

    band_freqs = freqs[mask]
    band_spec = spec[mask]
    peak_idx = int(np.argmax(band_spec))
    peak = float(band_spec[peak_idx])
    noise = float(np.median(band_spec) + 1e-9)
    return float(band_freqs[peak_idx] * 60.0), float(peak / noise)


def _detect_regular_peaks(
    t_s: np.ndarray,
    filtered: np.ndarray,
    fs: Optional[float],
    preferred_polarity: Optional[int] = None,
    switch_margin: float = 1.25,
) -> tuple[np.ndarray, int]:
    if fs is None or len(filtered) < max(40, int(6 * fs)):
        return np.array([], dtype=int), preferred_polarity if preferred_polarity in (-1, 1) else 1

    best_peaks = np.array([], dtype=int)
    best_polarity = 1
    best_score = -np.inf
    candidates: list[tuple[float, np.ndarray, int]] = []

    for polarity in (1, -1):
        y = polarity * filtered
        prominence = max(0.25, 0.35 * float(np.nanstd(y)))
        min_distance = max(1, int(0.35 * fs))
        peaks, _ = find_peaks(y, distance=min_distance, prominence=prominence)
        if len(peaks) < 3:
            continue
        ibi = np.diff(t_s[peaks])
        valid = (ibi >= 60.0 / 190.0) & (ibi <= 60.0 / 38.0)
        valid_ibi = ibi[valid]
        if len(valid_ibi) == 0:
            continue
        regularity = 1.0 / (1.0 + (float(np.std(valid_ibi)) / max(float(np.mean(valid_ibi)), 1e-9)))
        score = len(valid_ibi) * regularity
        candidates.append((score, peaks, polarity))
        if score > best_score:
            best_score = score
            best_peaks = peaks
            best_polarity = polarity

    if preferred_polarity in (-1, 1) and candidates:
        preferred = [item for item in candidates if item[2] == preferred_polarity]
        if preferred:
            preferred_score, preferred_peaks, preferred_sign = preferred[0]
            # Sliding windows often make the two polarities score almost equally well.
            # Keep the previous orientation unless the alternate is clearly stronger.
            if preferred_score > 0 and best_polarity != preferred_sign and best_score < preferred_score * switch_margin:
                return preferred_peaks, preferred_sign

    return best_peaks, best_polarity


def _hrv_from_peaks(t_s: np.ndarray, peaks: np.ndarray) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if len(peaks) < 3:
        return None, None, None, None
    ibi_ms = np.diff(t_s[peaks]) * 1000.0
    ibi_ms = ibi_ms[(ibi_ms >= 315.0) & (ibi_ms <= 1580.0)]
    if len(ibi_ms) < 2:
        return None, None, None, None
    mean_ibi = float(np.mean(ibi_ms))
    hr = float(60000.0 / np.median(ibi_ms))
    sdnn = float(np.std(ibi_ms, ddof=1)) if len(ibi_ms) > 1 else None
    rmssd = float(np.sqrt(np.mean(np.diff(ibi_ms) ** 2))) if len(ibi_ms) > 2 else None
    return hr, mean_ibi, rmssd, sdnn


def _resp_rate_from_baseline(
    t_s: np.ndarray,
    raw: np.ndarray,
    fs: Optional[float],
    band: tuple[float, float] = DEFAULT_RESP_BAND,
) -> Optional[float]:
    if fs is None or len(raw) < max(80, int(25 * fs)):
        return None
    resp = butter_bandpass(raw, fs, band[0], band[1], order=2)
    if resp is None:
        return None
    peak_hz, q, _, _ = spectral_peak_quality(resp, fs, band)
    if peak_hz is None or q < 2.0:
        return None
    return float(peak_hz * 60.0)


def _clamp_band(low_hz: float, high_hz: float) -> tuple[float, float]:
    low = max(0.05, float(low_hz))
    high = min(0.95, float(high_hz))
    if high <= low:
        high = min(0.95, low + 0.05)
    return low, high


def _live_respiration_bands(
    protocol: Optional[str],
    resp_target_brpm: Optional[float],
    resp_band: tuple[float, float],
) -> list[tuple[str, tuple[float, float]]]:
    label_bands: list[tuple[str, tuple[float, float]]] = [("configured", resp_band)]
    if resp_target_brpm is not None and np.isfinite(resp_target_brpm) and resp_target_brpm > 0:
        target_hz = float(resp_target_brpm) / 60.0
        label_bands.insert(0, ("target", _clamp_band(target_hz - 4.0 / 60.0, target_hz + 4.0 / 60.0)))

    if (protocol or "").lower() == "post_exercise_recovery":
        label_bands.extend([
            ("post_exercise_wide", (0.10, 0.70)),
            ("post_exercise_fast", (0.30, 0.75)),
        ])

    unique: list[tuple[str, tuple[float, float]]] = []
    seen: set[tuple[float, float]] = set()
    for label, band in label_bands:
        cleaned = _clamp_band(*band)
        key = (round(cleaned[0], 4), round(cleaned[1], 4))
        if key not in seen:
            unique.append((label, cleaned))
            seen.add(key)
    return unique


def _resp_candidate_score(
    protocol: Optional[str],
    resp_target_brpm: Optional[float],
    method: str,
    brpm: float,
    quality: float,
) -> float:
    if not np.isfinite(brpm) or not np.isfinite(quality):
        return 0.0
    score = float(max(quality, 0.0))

    if resp_target_brpm is not None and np.isfinite(resp_target_brpm) and resp_target_brpm > 0:
        target = float(resp_target_brpm)
        tolerance = max(2.0, 0.20 * target)
        return score / (1.0 + (abs(float(brpm) - target) / tolerance) ** 2)

    if (protocol or "").lower() == "post_exercise_recovery" and float(brpm) >= 18.0:
        return score * 1.35

    return score


def _candidate_beat_features(
    t_s: np.ndarray,
    raw: np.ndarray,
    filtered: np.ndarray,
    peaks: np.ndarray,
    polarity: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(peaks) < 5:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)

    orient_raw = np.asarray(raw, dtype=float)
    orient_filtered = np.asarray(filtered, dtype=float)
    if polarity == -1:
        center = float(np.nanmedian(orient_raw))
        orient_raw = 2.0 * center - orient_raw
        orient_filtered = -orient_filtered

    fs = estimate_fs_from_time(t_s)
    trough_distance = max(1, int(0.20 * fs)) if fs is not None else 1
    troughs, _ = find_peaks(-orient_filtered, distance=trough_distance)
    if len(troughs) < 3:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)

    candidate_times: list[float] = []
    candidate_amp: list[float] = []
    candidate_base: list[float] = []
    for p in peaks:
        prev_troughs = troughs[troughs < p]
        if len(prev_troughs) == 0:
            continue
        tr = int(prev_troughs[-1])
        amp = float(orient_raw[int(p)] - orient_raw[tr])
        base = float(orient_raw[tr])
        if not np.isfinite(amp) or not np.isfinite(base) or base <= 1.0 or amp <= 0.0:
            continue
        candidate_times.append(float(t_s[int(p)]))
        candidate_amp.append(amp)
        candidate_base.append(base)

    if len(candidate_times) < 5:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)

    candidate_times_arr = np.asarray(candidate_times, dtype=float)
    rr_ms = np.diff(candidate_times_arr) * 1000.0
    accepted_rr = clean_rr_series(rr_ms, min_rr_ms=315.0, max_rr_ms=1580.0, tol=0.22)
    accepted_beats = np.zeros(len(candidate_times_arr), dtype=bool)
    accepted_beats[0] = False
    accepted_beats[1:] = accepted_rr

    return (
        candidate_times_arr[accepted_beats],
        np.asarray(candidate_amp, dtype=float)[accepted_beats],
        np.asarray(candidate_base, dtype=float)[accepted_beats],
    )


def _resp_rate_from_beats(
    t_s: np.ndarray,
    raw: np.ndarray,
    filtered: Optional[np.ndarray],
    peaks: np.ndarray,
    polarity: int,
    protocol: Optional[str],
    resp_band: tuple[float, float],
    resp_target_brpm: Optional[float],
) -> Optional[float]:
    if filtered is None or len(peaks) < 5:
        return None

    beat_times, pulse_amp, pulse_base = _candidate_beat_features(t_s, raw, filtered, peaks, polarity)
    if len(beat_times) < 5:
        return None

    rr_s = np.diff(beat_times)
    rr_times = beat_times[1:]
    if len(rr_s) < 4:
        rr_interp = None
    else:
        t_resp = np.arange(float(t_s[0]), float(t_s[-1]), 1.0 / DEFAULT_RESP_FS)
        if len(t_resp) < int(DEFAULT_RESP_FS * 20):
            return None
        base_interp = np.interp(t_resp, beat_times, pulse_base)
        amp_interp = np.interp(t_resp, beat_times, pulse_amp)
        rr_interp = np.interp(t_resp, rr_times, rr_s)

        best_bpm = None
        best_score = 0.0
        for _label, band in _live_respiration_bands(protocol, resp_target_brpm, resp_band):
            for method, trace in (
                ("baseline", base_interp),
                ("amplitude", amp_interp),
                ("interval", rr_interp),
            ):
                resp = butter_bandpass(trace, DEFAULT_RESP_FS, band[0], band[1], order=2)
                if resp is None:
                    continue
                peak_hz, quality, _, _ = spectral_peak_quality(resp, DEFAULT_RESP_FS, band)
                if peak_hz is None or quality < 1.8:
                    continue
                bpm = float(peak_hz * 60.0)
                score = _resp_candidate_score(protocol, resp_target_brpm, method, bpm, float(quality))
                if score > best_score:
                    best_score = score
                    best_bpm = bpm

        if best_bpm is not None and best_score >= 2.0:
            return float(best_bpm)
    return None


def _saturation_fraction(y: np.ndarray, low: float = 3.0, high: float = 3580.0) -> float:
    y = _finite(y)
    if len(y) == 0:
        return np.nan
    return float(np.mean((y <= low) | (y >= high)))


def _channel_score(y: np.ndarray, fs: Optional[float]) -> tuple[float, Optional[float], float, float]:
    if fs is None or len(y) < max(50, int(6 * fs)):
        return 0.0, None, 0.0, np.nan
    sat = _saturation_fraction(y)
    if np.isfinite(sat) and sat > 0.30:
        return 0.0, None, 0.0, sat
    filtered = butter_bandpass(y, fs, DEFAULT_HEART_BAND[0], DEFAULT_HEART_BAND[1], order=2)
    if filtered is None:
        return 0.0, None, 0.0, sat
    bpm, q = _fft_bpm_quality(filtered, fs, DEFAULT_HEART_BAND)
    amp = _percentile_range(filtered, 5.0, 95.0)
    sat_penalty = max(0.0, 1.0 - 2.0 * sat) if np.isfinite(sat) else 1.0
    return float(max(q, 0.0) * max(amp, 0.0) * sat_penalty), bpm, q, sat


def _choose_best_channel(t_s: np.ndarray, series: Dict[str, np.ndarray], columns: Iterable[str]) -> tuple[Optional[str], float]:
    fs = estimate_fs_from_time(t_s)
    best_name = None
    best_score = 0.0
    for col in columns:
        y = np.asarray(series.get(col, []), dtype=float)
        if len(y) != len(t_s):
            continue
        score, _, _, _ = _channel_score(y, fs)
        if score > best_score:
            best_name = col
            best_score = score
    return best_name, best_score


def _pick_spo2_channels(columns: Iterable[str]) -> tuple[Optional[str], Optional[str]]:
    cols = set(columns)
    ir = "NIR" if "NIR" in cols else ("NIR_diff" if "NIR_diff" in cols else None)
    for red in ("F6", "F6_diff", "FXL_diff", "FY_diff", "F7", "F8"):
        if red in cols:
            return red, ir
    return None, ir


def _acdc_ratio(y: np.ndarray) -> tuple[Optional[float], Optional[float], Optional[float]]:
    y = _finite(y)
    if len(y) < 20:
        return None, None, None
    dc = float(np.nanmedian(y))
    ac = _percentile_range(y, 5.0, 95.0)
    if not np.isfinite(dc) or not np.isfinite(ac) or dc <= 10.0 or ac <= 0.0:
        return ac if np.isfinite(ac) else None, dc if np.isfinite(dc) else None, None
    return ac, dc, float(ac / dc)


def _robust_scale(x: np.ndarray, center: float, floor: float) -> float:
    x = _finite(x)
    if len(x) == 0:
        return float(floor)
    mad = float(np.median(np.abs(x - center)))
    return float(max(1.4826 * mad, floor))


def _robust_inlier_mask(x: np.ndarray, zmax: float, min_scale: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    out = np.zeros(len(x), dtype=bool)
    if not np.any(finite):
        return out
    center = float(np.median(x[finite]))
    scale = _robust_scale(x[finite], center, min_scale)
    out[finite] = np.abs(x[finite] - center) <= zmax * scale
    return out


def _beatwise_acdc(raw: np.ndarray, peaks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_beats = max(0, len(peaks) - 1)
    ac = np.full(n_beats, np.nan, dtype=float)
    dc = np.full(n_beats, np.nan, dtype=float)
    acdc = np.full(n_beats, np.nan, dtype=float)

    for i in range(n_beats):
        a = int(peaks[i])
        b = int(peaks[i + 1])
        if b <= a + 2:
            continue
        window = _finite(raw[a:b])
        if len(window) < 3:
            continue
        ac[i] = float(np.percentile(window, 95) - np.percentile(window, 5))
        dc[i] = float(np.median(window))
        if dc[i] > 10.0 and ac[i] > 0.0:
            acdc[i] = ac[i] / dc[i]

    return ac, dc, acdc


def _spo2_estimate(
    t_s: np.ndarray,
    series: Dict[str, np.ndarray],
    columns: Iterable[str],
    anchor_ratio: Optional[float],
    anchor_pct: Optional[float],
    slope: float,
    clip_min: float,
    clip_max: float,
) -> tuple[Optional[float], Optional[float], str, Optional[str], Optional[str]]:
    red_col, ir_col = _pick_spo2_channels(columns)
    if red_col is None or ir_col is None:
        return None, None, "red/IR channels unavailable", red_col, ir_col

    fs = estimate_fs_from_time(t_s)
    red = np.asarray(series.get(red_col, []), dtype=float)
    ir = np.asarray(series.get(ir_col, []), dtype=float)
    if fs is None or len(red) < max(80, int(10 * fs)) or len(ir) < max(80, int(10 * fs)):
        return None, None, "calibrating", red_col, ir_col

    ref_col, _ = _choose_best_channel(t_s, series, columns)
    if ref_col is None:
        return None, None, "weak red/IR pulsatile signal", red_col, ir_col
    ref = np.asarray(series[ref_col], dtype=float)
    ref_filtered = butter_bandpass(ref, fs, DEFAULT_HEART_BAND[0], DEFAULT_HEART_BAND[1], order=2)
    if ref_filtered is None:
        return None, None, "calibrating", red_col, ir_col
    peaks, _ = _detect_regular_peaks(t_s, ref_filtered, fs)
    if len(peaks) < 6:
        return None, None, "calibrating", red_col, ir_col

    red_ac, red_dc, red_acdc = _beatwise_acdc(red, peaks)
    ir_ac, ir_dc, ir_acdc = _beatwise_acdc(ir, peaks)
    ratio = red_acdc / np.maximum(ir_acdc, 1e-12)
    valid = (
        np.isfinite(red_acdc)
        & np.isfinite(ir_acdc)
        & np.isfinite(ratio)
        & (red_ac > 0.0)
        & (ir_ac > 0.0)
        & (red_dc >= 20.0)
        & (ir_dc >= 20.0)
        & (ratio > 0.0)
    )
    if int(np.sum(valid)) < 5:
        return None, None, "weak red/IR pulsatile signal", red_col, ir_col

    ratio_med_pre = float(np.median(ratio[valid]))
    ratio_inlier = _robust_inlier_mask(
        ratio,
        zmax=4.0,
        min_scale=max(0.03, 0.10 * abs(ratio_med_pre) if np.isfinite(ratio_med_pre) else 0.03),
    )
    quality = valid & ratio_inlier
    if int(np.sum(quality)) < 5:
        quality = valid

    ratio_value = float(np.median(ratio[quality]))
    if anchor_ratio is None or anchor_pct is None:
        return None, ratio_value, "needs calibration anchor", red_col, ir_col

    estimate = float(anchor_pct - slope * (ratio_value - anchor_ratio))
    estimate = float(np.clip(estimate, clip_min, clip_max))
    return estimate, ratio_value, "estimated", red_col, ir_col


def compute_live_metrics(
    t_s: np.ndarray,
    series: Dict[str, np.ndarray],
    selected_channel: str,
    columns: Iterable[str],
    *,
    spo2_anchor_ratio: Optional[float] = None,
    spo2_anchor_pct: Optional[float] = None,
    spo2_slope: float = 25.0,
    spo2_clip_min: float = 70.0,
    spo2_clip_max: float = 100.0,
    resp_band: tuple[float, float] = DEFAULT_RESP_BAND,
    protocol: Optional[str] = None,
    resp_target_brpm: Optional[float] = None,
    calibration: Optional[dict] = None,
    preferred_pulse_polarity: Optional[int] = None,
) -> tuple[LiveMetrics, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    t_s = np.asarray(t_s, dtype=float)
    columns = list(columns)
    metrics = LiveMetrics(selected_channel=selected_channel)

    if len(t_s) < 10 or selected_channel not in series:
        return metrics, None, None, None

    y = np.asarray(series[selected_channel], dtype=float)
    if len(y) != len(t_s):
        return metrics, None, None, None

    fs = estimate_fs_from_time(t_s)
    metrics.fs_hz = fs
    metrics.saturation_fraction = _saturation_fraction(y)
    metrics.best_channel, best_score = _choose_best_channel(t_s, series, columns)

    filtered = None
    peaks = np.array([], dtype=int)
    polarity = 1
    if fs is not None:
        filtered = butter_bandpass(y, fs, DEFAULT_HEART_BAND[0], DEFAULT_HEART_BAND[1], order=2)

    if filtered is not None:
        hr_fft, q_fft = _fft_bpm_quality(filtered, fs, DEFAULT_HEART_BAND)
        peaks, polarity = _detect_regular_peaks(t_s, filtered, fs, preferred_polarity=preferred_pulse_polarity)
        hr_peak, mean_ibi, rmssd, sdnn = _hrv_from_peaks(t_s, peaks)

        metrics.hr_fft_bpm = hr_fft
        metrics.hr_peak_bpm = hr_peak
        if hr_peak is not None and hr_fft is not None and abs(hr_peak - hr_fft) / max(hr_fft, 1.0) > 0.12:
            metrics.hr_bpm = hr_fft
        else:
            metrics.hr_bpm = hr_peak if hr_peak is not None else hr_fft
        metrics.mean_ibi_ms = mean_ibi
        metrics.rmssd_ms = rmssd
        metrics.sdnn_ms = sdnn
        metrics.signal_quality = float(q_fft)
        metrics.pulse_polarity = int(polarity)

        if polarity == -1:
            filtered = -filtered

    red_col, ir_col = _pick_spo2_channels(columns)
    pi_source = ir_col if ir_col in series else selected_channel
    y_pi = np.asarray(series.get(pi_source, y), dtype=float)

    pulse_amp = _percentile_range(y_pi, 5.0, 95.0)
    dc = float(np.nanmedian(_finite(y_pi))) if len(_finite(y_pi)) else np.nan
    metrics.pulse_amplitude = pulse_amp if np.isfinite(pulse_amp) else None
    if np.isfinite(pulse_amp) and np.isfinite(dc) and abs(dc) > 10.0:
        metrics.perfusion_proxy_pct = float(100.0 * pulse_amp / abs(dc))
        metrics.perfusion_index_pct = apply_pi_calibration(metrics.perfusion_proxy_pct, calibration or {})

    rr_live = _resp_rate_from_beats(
        t_s,
        y,
        filtered,
        peaks,
        polarity,
        protocol,
        resp_band,
        resp_target_brpm,
    )
    if rr_live is None:
        rr_live = _resp_rate_from_baseline(t_s, y, fs, band=resp_band)

    metrics.respiratory_rate_brpm = apply_rr_calibration(
        rr_live,
        calibration or {},
        protocol,
        source_feature="live_respiratory_rate_brpm_median",
    )

    spo2, ratio, spo2_status, red_col, ir_col = _spo2_estimate(
        t_s,
        series,
        columns,
        anchor_ratio=spo2_anchor_ratio,
        anchor_pct=spo2_anchor_pct,
        slope=spo2_slope,
        clip_min=spo2_clip_min,
        clip_max=spo2_clip_max,
    )
    metrics.spo2_estimated_pct = spo2
    metrics.spo2_ratio = ratio
    metrics.spo2_status = spo2_status
    metrics.red_channel = red_col
    metrics.ir_channel = ir_col

    if metrics.signal_quality is None:
        metrics.signal_quality = float(best_score) if best_score > 0 else None

    metrics.artifact_flag = bool(
        metrics.hr_bpm is None
        or metrics.signal_quality is None
        or metrics.signal_quality < 2.0
        or (metrics.saturation_fraction is not None and np.isfinite(metrics.saturation_fraction) and metrics.saturation_fraction > 0.10)
    )

    peak_t = t_s[peaks] if len(peaks) else None
    peak_y = filtered[peaks] if filtered is not None and len(peaks) else None
    return metrics, filtered, peak_t, peak_y
