from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from ..beats import detect_beats
from ..config import AnalysisConfig
from ..signal import butter_bandpass, estimate_fs_from_time, spectral_peak_quality


def _clamp_band(low_hz: float, high_hz: float) -> tuple[float, float]:
    low = max(0.05, float(low_hz))
    high = min(0.95, float(high_hz))
    if high <= low:
        high = min(0.95, low + 0.05)
    return low, high


def _respiration_bands(cfg: AnalysisConfig) -> list[tuple[str, tuple[float, float]]]:
    protocol = (cfg.protocol or "").lower()
    bands: list[tuple[str, tuple[float, float]]] = [("configured", cfg.resp_band)]

    if cfg.resp_target_brpm is not None and np.isfinite(cfg.resp_target_brpm) and cfg.resp_target_brpm > 0:
        target_hz = float(cfg.resp_target_brpm) / 60.0
        # +/- 4 brpm around the metronome target, with enough width for imperfect pacing.
        bands.insert(0, ("target", _clamp_band(target_hz - 4.0 / 60.0, target_hz + 4.0 / 60.0)))

    if protocol == "post_exercise_recovery":
        # Recovery can easily sit above the normal 9-21 brpm band; keep both normal and fast bands.
        bands.extend([
            ("post_exercise_wide", (0.10, 0.70)),
            ("post_exercise_fast", (0.30, 0.75)),
        ])

    unique: list[tuple[str, tuple[float, float]]] = []
    seen: set[tuple[float, float]] = set()
    for label, band in bands:
        cleaned = _clamp_band(*band)
        key = (round(cleaned[0], 4), round(cleaned[1], 4))
        if key not in seen:
            unique.append((label, cleaned))
            seen.add(key)
    return unique


def _candidate_selection_score(candidate: dict[str, Any], cfg: AnalysisConfig) -> float:
    q = float(candidate.get("quality") or 0.0)
    bpm = candidate.get("brpm")
    if bpm is None or not np.isfinite(bpm):
        return 0.0

    if cfg.resp_target_brpm is not None and np.isfinite(cfg.resp_target_brpm) and cfg.resp_target_brpm > 0:
        target = float(cfg.resp_target_brpm)
        tolerance = max(2.0, 0.20 * target)
        return q / (1.0 + (abs(float(bpm) - target) / tolerance) ** 2)

    if (cfg.protocol or "").lower() == "post_exercise_recovery":
        # If a high-rate respiratory modulation is strong enough, prefer it during recovery runs.
        if float(bpm) >= 18.0 and candidate.get("method") in {"amplitude", "interval", "baseline"}:
            return q * 1.35

    return q


def run_respiration(t_s: np.ndarray, raw: np.ndarray, cfg: AnalysisConfig) -> Dict[str, Any]:
    fs = estimate_fs_from_time(t_s)
    if fs is None:
        raise RuntimeError("Could not estimate sampling rate.")
    beats = detect_beats(raw, t_s, fs, cfg)

    beat_times = beats.beat_times
    pulse_amp = beats.pulse_amp
    pulse_base = beats.pulse_base

    if len(beat_times) < 5:
        raise RuntimeError("Too few accepted beats.")

    rr_s = np.diff(beat_times)
    rr_times = beat_times[1:]
    t_resp = np.arange(t_s[0], t_s[-1], 1.0 / cfg.resp_fs)

    base_interp = np.interp(t_resp, beat_times, pulse_base)
    amp_interp = np.interp(t_resp, beat_times, pulse_amp)
    rr_interp = np.interp(t_resp, rr_times, rr_s) if len(rr_s) >= 4 else None

    candidates: list[dict[str, Any]] = []
    traces_by_band: dict[str, dict[str, Any]] = {}
    for band_label, band in _respiration_bands(cfg):
        resp_base = butter_bandpass(base_interp, cfg.resp_fs, band[0], band[1], order=2)
        resp_amp = butter_bandpass(amp_interp, cfg.resp_fs, band[0], band[1], order=2)
        resp_rr = butter_bandpass(rr_interp, cfg.resp_fs, band[0], band[1], order=2) if rr_interp is not None else None

        base_f, q_base, _, _ = spectral_peak_quality(resp_base, cfg.resp_fs, band)
        amp_f, q_amp, _, _ = spectral_peak_quality(resp_amp, cfg.resp_fs, band)
        rr_f, q_rr, _, _ = spectral_peak_quality(resp_rr, cfg.resp_fs, band) if resp_rr is not None else (None, 0.0, None, None)

        traces_by_band[band_label] = {
            "band": band,
            "resp_base": resp_base,
            "resp_amp": resp_amp,
            "resp_rr": resp_rr,
        }

        for method, freq_hz, quality in (
            ("baseline", base_f, q_base),
            ("amplitude", amp_f, q_amp),
            ("interval", rr_f, q_rr),
        ):
            if freq_hz is None:
                continue
            candidate = {
                "method": method,
                "band_label": band_label,
                "band_low_hz": float(band[0]),
                "band_high_hz": float(band[1]),
                "brpm": float(freq_hz * 60.0),
                "quality": float(quality),
            }
            candidate["selection_score"] = _candidate_selection_score(candidate, cfg)
            candidates.append(candidate)

    if not candidates:
        raise RuntimeError("No valid respiratory estimates.")

    best = max(candidates, key=lambda item: float(item["selection_score"]))
    best_method = best["method"]
    bpm_fused = float(best["brpm"])
    best_q = float(best["quality"])
    best_band_label = str(best["band_label"])
    selected_traces = traces_by_band.get(best_band_label, {})
    selected_band = selected_traces.get("band", cfg.resp_band)
    resp_base = selected_traces.get("resp_base")
    resp_amp = selected_traces.get("resp_amp")
    resp_rr = selected_traces.get("resp_rr")

    method_values = {item["method"]: item for item in candidates if item["band_label"] == best_band_label}
    base_item = method_values.get("baseline")
    amp_item = method_values.get("amplitude")
    rr_item = method_values.get("interval")

    fast_candidates = [item for item in candidates if float(item["brpm"]) >= 18.0]
    fast_best = max(fast_candidates, key=lambda item: float(item["quality"])) if fast_candidates else None

    summary = {
        "estimated_fs_hz": fs,
        "protocol": cfg.protocol,
        "resp_target_brpm": cfg.resp_target_brpm,
        "resp_band_low_hz": float(selected_band[0]),
        "resp_band_high_hz": float(selected_band[1]),
        "resp_band_low_brpm": float(selected_band[0] * 60.0),
        "resp_band_high_brpm": float(selected_band[1] * 60.0),
        "resp_baseline_brpm": None if base_item is None else base_item["brpm"],
        "resp_amplitude_brpm": None if amp_item is None else amp_item["brpm"],
        "resp_interval_brpm": None if rr_item is None else rr_item["brpm"],
        "q_baseline": 0.0 if base_item is None else base_item["quality"],
        "q_amplitude": 0.0 if amp_item is None else amp_item["quality"],
        "q_interval": 0.0 if rr_item is None else rr_item["quality"],
        "chosen_method": best_method,
        "chosen_band_label": best_band_label,
        "final_respiratory_rate_brpm": bpm_fused,
        "best_quality": best_q,
        "selection_score": best["selection_score"],
        "fast_respiration_candidate_brpm": None if fast_best is None else fast_best["brpm"],
        "fast_respiration_candidate_method": None if fast_best is None else fast_best["method"],
        "fast_respiration_candidate_quality": None if fast_best is None else fast_best["quality"],
        "target_error_brpm": None if cfg.resp_target_brpm is None else float(bpm_fused - cfg.resp_target_brpm),
        "candidate_estimates": candidates,
    }

    derived_df = pd.DataFrame({
        "t_resp_s": t_resp,
        "baseline_interp": base_interp,
        "amplitude_interp": amp_interp,
        "interval_interp": rr_interp if rr_interp is not None else np.nan,
        "resp_base": resp_base if resp_base is not None else np.nan,
        "resp_amp": resp_amp if resp_amp is not None else np.nan,
        "resp_rr": resp_rr if resp_rr is not None else np.nan,
    })

    return {
        "summary": summary,
        "tables": {
            "respiration_traces": derived_df,
        },
        "artifacts": {
            "beats": beats,
        }
    }
