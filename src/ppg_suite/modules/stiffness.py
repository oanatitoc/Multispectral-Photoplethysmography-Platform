from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

from ..beats import detect_beats
from ..config import AnalysisConfig
from ..signal import estimate_fs_from_time, safe_median, safe_mean


MIN_ABS_AMP = 8.0
MIN_BEAT_DUR_S = 0.45
MAX_BEAT_DUR_S = 1.80
MIN_CREST_RATIO = 0.12
MAX_CREST_RATIO = 0.72
MIN_PW50_S = 0.10
MAX_PW50_S = 0.50
MIN_REFLECTION_S = 0.08
MAX_REFLECTION_S = 0.45
TEMPLATE_N = 101


def _adaptive_window_hint(fs: float, span_s: float, minimum: int = 5, maximum: int = 15) -> int:
    if not np.isfinite(fs) or fs <= 0:
        return minimum
    win = int(round(fs * span_s))
    win = max(minimum, min(maximum, win))
    if win % 2 == 0:
        win += 1
    if win > maximum:
        win = maximum if maximum % 2 == 1 else maximum - 1
    return max(minimum, win)


def _smooth_beat(y: np.ndarray, window_hint: int = 11) -> np.ndarray:
    y = np.asarray(y, float)
    if len(y) < 7:
        return y.copy()
    win = min(len(y) if len(y) % 2 == 1 else len(y) - 1, window_hint)
    win = max(5, win)
    if win % 2 == 0:
        win -= 1
    if win < 5:
        return y.copy()
    poly = 3 if win >= 7 else 2
    return savgol_filter(y, window_length=win, polyorder=poly, mode="interp")

def _refine_systolic_peak(y: np.ndarray, seed_i: int, fs: float) -> int:
    y = np.asarray(y, float)
    if len(y) < 5:
        return int(seed_i)

    w = max(2, int(0.12 * fs))
    lo = max(1, int(seed_i) - w)
    hi = min(len(y) - 2, int(seed_i) + w)
    if hi <= lo:
        return int(seed_i)

    return int(lo + np.argmax(y[lo : hi + 1]))


def _refine_onset(y: np.ndarray, peak_i: int, fs: float) -> int:
    y = np.asarray(y, float)
    if len(y) < 5:
        return 0

    back = max(3, int(0.45 * fs))
    lo = max(0, int(peak_i) - back)
    hi = int(peak_i)
    if hi <= lo:
        return 0

    return int(lo + np.argmin(y[lo : hi + 1]))

def _derivatives(y: np.ndarray, t: np.ndarray, window_hint: int = 11) -> tuple[np.ndarray, np.ndarray]:
    vpg = np.gradient(y, t)
    vpg = _smooth_beat(vpg, window_hint=window_hint)
    apg = np.gradient(vpg, t)
    apg = _smooth_beat(apg, window_hint=window_hint)
    return vpg, apg


def _pulse_width(t: np.ndarray, y: np.ndarray, onset_level: float, peak_level: float, frac: float) -> float:
    amp = peak_level - onset_level
    if amp <= 0:
        return np.nan
    thr = onset_level + frac * amp
    idx = np.flatnonzero(y >= thr)
    if len(idx) < 2:
        return np.nan
    return float(t[idx[-1]] - t[idx[0]])


def _segment_area(t: np.ndarray, y: np.ndarray, baseline: float) -> float:
    return float(np.trapz(np.clip(y - baseline, 0.0, None), t))


def _detect_notch_and_diastolic(t: np.ndarray, y: np.ndarray, peak_i: int) -> tuple[float, float, float, float]:
    n = len(y)
    if peak_i >= n - 5:
        return np.nan, np.nan, np.nan, np.nan

    search = y[peak_i + 1 :]
    if len(search) < 6:
        return np.nan, np.nan, np.nan, np.nan

    mins, _ = find_peaks(
        -search,
        prominence=max(1e-6, 0.03 * np.ptp(y)),
        distance=max(1, len(search) // 10),
    )
    if len(mins) == 0:
        return np.nan, np.nan, np.nan, np.nan

    notch_rel = int(mins[0])
    notch_i = peak_i + 1 + notch_rel
    notch_t = float(t[notch_i])
    notch_y = float(y[notch_i])

    after_notch = y[notch_i + 1 :]
    if len(after_notch) < 4:
        return notch_t, notch_y, np.nan, np.nan

    maxs, _ = find_peaks(
        after_notch,
        prominence=max(1e-6, 0.02 * np.ptp(y)),
        distance=max(1, len(after_notch) // 10),
    )
    if len(maxs) == 0:
        return notch_t, notch_y, np.nan, np.nan

    dias_rel = int(maxs[0])
    dias_i = notch_i + 1 + dias_rel
    dias_t = float(t[dias_i])
    dias_y = float(y[dias_i])
    return notch_t, notch_y, dias_t, dias_y


def _detect_apg_points(t: np.ndarray, apg: np.ndarray) -> dict[str, float]:
    n = len(apg)
    out = {
        "a": np.nan,
        "b": np.nan,
        "c": np.nan,
        "d": np.nan,
        "e": np.nan,
        "a_time_s": np.nan,
        "b_time_s": np.nan,
        "c_time_s": np.nan,
        "d_time_s": np.nan,
        "e_time_s": np.nan,
    }
    if n < 8:
        return out

    maxs, _ = find_peaks(apg, distance=max(1, n // 12))
    mins, _ = find_peaks(-apg, distance=max(1, n // 12))
    if len(maxs) == 0:
        return out

    first35 = max(1, int(0.35 * n))
    a_candidates = maxs[maxs < first35]
    if len(a_candidates) == 0:
        return out

    a_idx = int(a_candidates[np.argmax(apg[a_candidates])])
    out["a"] = float(apg[a_idx])
    out["a_time_s"] = float(t[a_idx])

    def pick_min(start: int, end: int) -> Optional[int]:
        cand = mins[(mins > start) & (mins < end)]
        if len(cand) == 0:
            return None
        return int(cand[np.argmin(apg[cand])])

    def pick_max(start: int, end: int) -> Optional[int]:
        cand = maxs[(maxs > start) & (maxs < end)]
        if len(cand) == 0:
            return None
        return int(cand[np.argmax(apg[cand])])

    b_idx = pick_min(a_idx, int(0.60 * n))
    c_idx = pick_max(b_idx if b_idx is not None else a_idx, int(0.75 * n))
    d_idx = pick_min(c_idx if c_idx is not None else (b_idx if b_idx is not None else a_idx), int(0.90 * n))
    e_idx = pick_max(d_idx if d_idx is not None else (c_idx if c_idx is not None else a_idx), n - 1)

    for name, idx in (("b", b_idx), ("c", c_idx), ("d", d_idx), ("e", e_idx)):
        if idx is not None:
            out[name] = float(apg[idx])
            out[f"{name}_time_s"] = float(t[idx])

    return out


def _ratio(num: float, den: float) -> float:
    if np.isnan(num) or np.isnan(den) or abs(den) < 1e-12:
        return np.nan
    return float(num / den)

def _safe_fraction(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) < 1e-12:
        return np.nan
    return float(num / den)


def _resample_unit_pulse(t: np.ndarray, y: np.ndarray, n: int = TEMPLATE_N) -> tuple[np.ndarray, np.ndarray] | None:
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    if len(t) < 5 or len(y) < 5:
        return None

    dur = float(t[-1] - t[0])
    amp = float(np.max(y) - y[0])
    if dur <= 1e-12 or amp <= 1e-12:
        return None

    u = (t - t[0]) / dur
    y_norm = (y - y[0]) / amp

    # guard against repeated times after normalization
    keep = np.r_[True, np.diff(u) > 1e-12]
    u = u[keep]
    y_norm = y_norm[keep]
    if len(u) < 4:
        return None

    u_grid = np.linspace(0.0, 1.0, n)
    y_grid = np.interp(u_grid, u, y_norm)
    return u_grid, y_grid


def _build_templates(beats_df: pd.DataFrame, waveforms: dict[int, dict[str, np.ndarray]], cfg: AnalysisConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if beats_df.empty:
        return pd.DataFrame()

    segment_defs = [("overall", -np.inf, np.inf), *cfg.segments]

    for seg_name, t0, t1 in segment_defs:
        mats = []
        u_ref = None

        for beat_idx, beat_row in beats_df.iterrows():
            bt = float(beat_row["beat_time_s"])
            if not (t0 <= bt < t1):
                continue

            wf = waveforms.get(int(beat_idx))
            if wf is None:
                continue

            tpl = _resample_unit_pulse(wf["time_s"], wf["ppg"], TEMPLATE_N)
            if tpl is None:
                continue

            u_grid, y_grid = tpl
            u_ref = u_grid
            mats.append(y_grid)

        if len(mats) < 3 or u_ref is None:
            continue

        arr = np.vstack(mats)
        med = np.median(arr, axis=0)
        p25 = np.percentile(arr, 25, axis=0)
        p75 = np.percentile(arr, 75, axis=0)

        for i, u in enumerate(u_ref):
            rows.append({
                "segment": seg_name,
                "n_beats": int(len(mats)),
                "u": float(u),
                "ppg_norm_median": float(med[i]),
                "ppg_norm_p25": float(p25[i]),
                "ppg_norm_p75": float(p75[i]),
            })

    return pd.DataFrame(rows)

def _template_width_u(u: np.ndarray, y: np.ndarray, frac: float) -> float:
    y = np.asarray(y, float)
    u = np.asarray(u, float)
    if len(y) < 5:
        return np.nan

    onset = float(y[0])
    peak = float(np.max(y))
    amp = peak - onset
    if amp <= 1e-12:
        return np.nan

    thr = onset + frac * amp
    idx = np.flatnonzero(y >= thr)
    if len(idx) < 2:
        return np.nan

    return float(u[idx[-1]] - u[idx[0]])


def _template_area_fraction(u: np.ndarray, y: np.ndarray, peak_i: int) -> tuple[float, float]:
    y = np.asarray(y, float)
    u = np.asarray(u, float)
    if len(y) < 5 or peak_i < 1 or peak_i >= len(y) - 1:
        return np.nan, np.nan

    onset = float(y[0])
    total = float(np.trapz(np.clip(y - onset, 0.0, None), u))
    if total <= 1e-12:
        return np.nan, np.nan

    syst = float(np.trapz(np.clip(y[: peak_i + 1] - onset, 0.0, None), u[: peak_i + 1]))
    diast = float(np.trapz(np.clip(y[peak_i:] - onset, 0.0, None), u[peak_i:]))

    return syst / total, diast / total


def _detect_template_notch_and_diastolic(
    u: np.ndarray, y: np.ndarray, peak_i: int
) -> tuple[float, float, float, float, bool]:
    """
    Detect notch and diastolic peak on the normalized median template.
    Returns: notch_u, notch_y, dias_u, dias_y, quality_ok
    """
    if peak_i >= len(y) - 6:
        return np.nan, np.nan, np.nan, np.nan, False

    search = y[peak_i + 1:]
    if len(search) < 6:
        return np.nan, np.nan, np.nan, np.nan, False

    # notch candidate = first meaningful local minimum after systolic peak
    mins, _ = find_peaks(
        -search,
        prominence=max(1e-6, 0.015 * np.ptp(y)),
        distance=max(1, len(search) // 10),
    )
    if len(mins) == 0:
        return np.nan, np.nan, np.nan, np.nan, False

    notch_rel = int(mins[0])
    notch_i = peak_i + 1 + notch_rel

    after_notch = y[notch_i + 1:]
    if len(after_notch) < 4:
        return float(u[notch_i]), float(y[notch_i]), np.nan, np.nan, False

    maxs, _ = find_peaks(
        after_notch,
        prominence=max(1e-6, 0.01 * np.ptp(y)),
        distance=max(1, len(after_notch) // 10),
    )
    if len(maxs) == 0:
        return float(u[notch_i]), float(y[notch_i]), np.nan, np.nan, False

    dias_rel = int(maxs[0])
    dias_i = notch_i + 1 + dias_rel

    notch_u = float(u[notch_i])
    notch_y = float(y[notch_i])
    dias_u = float(u[dias_i])
    dias_y = float(y[dias_i])

    onset_y = float(y[0])
    peak_y = float(y[peak_i])
    peak_u = float(u[peak_i])
    amp = peak_y - onset_y
    if amp <= 1e-12:
        return notch_u, notch_y, dias_u, dias_y, False

    notch_drop = peak_y - notch_y
    dias_rebound = dias_y - notch_y
    refl_u = dias_u - peak_u

    quality_ok = True
    if not (peak_u < notch_u < dias_u < 1.0):
        quality_ok = False
    if not (0.05 <= refl_u <= 0.45):
        quality_ok = False
    if notch_drop < 0.03 * amp:
        quality_ok = False
    if dias_rebound < 0.015 * amp:
        quality_ok = False
    if not (onset_y - 0.05 * amp <= dias_y <= peak_y - 0.01 * amp):
        quality_ok = False

    return notch_u, notch_y, dias_u, dias_y, quality_ok


def _summarize_template_landmarks(templates_df: pd.DataFrame, template_window_hint: int = 11) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if templates_df.empty:
        return pd.DataFrame()

    for seg_name in templates_df["segment"].dropna().unique():
        seg = templates_df[templates_df["segment"] == seg_name].copy()
        seg = seg.sort_values("u")

        u = seg["u"].to_numpy(float)
        y = seg["ppg_norm_median"].to_numpy(float)
        if len(u) < 10:
            continue

        # smooth only lightly; template is already median-based
        y_sm = _smooth_beat(y, window_hint=template_window_hint)
        peak_i = int(np.argmax(y_sm))
        peak_u = float(u[peak_i])
        peak_y = float(y_sm[peak_i])

        pw25_u = _template_width_u(u, y_sm, 0.25)
        pw50_u = _template_width_u(u, y_sm, 0.50)
        pw75_u = _template_width_u(u, y_sm, 0.75)

        syst_frac, diast_frac = _template_area_fraction(u, y_sm, peak_i)
        notch_u, notch_y, dias_u, dias_y, notch_ok = _detect_template_notch_and_diastolic(u, y_sm, peak_i)

        if notch_ok:
            refl_u = dias_u - peak_u
            ai_ratio = float(dias_y / peak_y) if abs(peak_y) > 1e-12 else np.nan
            ai_delta = float((dias_y - peak_y) / peak_y) if abs(peak_y) > 1e-12 else np.nan
        else:
            notch_u = np.nan
            notch_y = np.nan
            dias_u = np.nan
            dias_y = np.nan
            refl_u = np.nan
            ai_ratio = np.nan
            ai_delta = np.nan

        rows.append({
            "segment": seg_name,
            "template_peak_u": peak_u,
            "template_peak_y": peak_y,
            "template_pw25_u": pw25_u,
            "template_pw50_u": pw50_u,
            "template_pw75_u": pw75_u,
            "template_systolic_area_fraction": syst_frac,
            "template_diastolic_area_fraction": diast_frac,
            "template_notch_u": notch_u,
            "template_notch_y": notch_y,
            "template_diastolic_peak_u": dias_u,
            "template_diastolic_peak_y": dias_y,
            "template_reflection_u": refl_u,
            "template_ai_ratio": ai_ratio,
            "template_ai_delta": ai_delta,
            "template_notch_quality_ok": bool(notch_ok),
        })

    return pd.DataFrame(rows)

def _validate_reflection(row: dict[str, Any]) -> dict[str, Any]:
    amp = row["systolic_amplitude"]
    notch_t = row["notch_time_s"]
    dias_t = row["diastolic_peak_time_s"]
    notch_y = row["notch_level"]
    dias_y = row["diastolic_peak_level"]
    peak_y = row["systolic_peak_level"]
    onset_y = row["onset_level"]
    peak_t = row["beat_time_s"]
    end_t = row["beat_end_time_s"]

    ok = True
    if not (np.isfinite(notch_t) and np.isfinite(dias_t) and np.isfinite(notch_y) and np.isfinite(dias_y)):
        ok = False
    elif not (peak_t < notch_t < dias_t < end_t):
        ok = False
    else:
        refl = dias_t - peak_t
        notch_drop = peak_y - notch_y
        dias_rebound = dias_y - notch_y
        if not (MIN_REFLECTION_S <= refl <= MAX_REFLECTION_S):
            ok = False
        if notch_drop < 0.05 * amp:
            ok = False
        if dias_rebound < 0.03 * amp:
            ok = False
        if not (onset_y - 0.10 * amp <= dias_y <= peak_y - 0.02 * amp):
            ok = False

    row["reflection_quality_ok"] = bool(ok)
    if not ok:
        row["notch_time_s"] = np.nan
        row["notch_level"] = np.nan
        row["diastolic_peak_time_s"] = np.nan
        row["diastolic_peak_level"] = np.nan
        row["reflection_time_s"] = np.nan
        row["augmentation_index_ratio"] = np.nan
        row["augmentation_index_delta"] = np.nan
        row["stiffness_index"] = np.nan
    return row


def _validate_sdppg(row: dict[str, Any]) -> dict[str, Any]:
    a = row["a"]
    b = row["b"]
    c = row["c"]
    d = row["d"]
    e = row["e"]
    ta = row["a_time_s"]
    tb = row["b_time_s"]
    tc = row["c_time_s"]
    td = row["d_time_s"]
    te = row["e_time_s"]

    ok = all(np.isfinite(v) for v in (a, b, c, d, e, ta, tb, tc, td, te))
    if ok:
        ok = (a > 0) and (ta < tb < tc < td < te)
        ok = ok and (abs(a) > 1e-6)

    row["sdppg_quality_ok"] = bool(ok)
    if not ok:
        for k in [
            "a", "b", "c", "d", "e",
            "a_over_a", "b_over_a", "c_over_a", "d_over_a", "e_over_a",
            "sdppg_aging_index", "sdppg_substitute_index",
            "a_time_s", "b_time_s", "c_time_s", "d_time_s", "e_time_s",
        ]:
            row[k] = np.nan
    return row


def _stat_if_enough(x: np.ndarray, min_n: int = 10, reducer: str = "median") -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < min_n:
        return np.nan
    if reducer == "mean":
        return float(np.mean(x))
    return float(np.median(x))


def _summarize_segments(df: pd.DataFrame, cfg: AnalysisConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame()

    baseline_ref: dict[str, Any] | None = None

    def pct_delta(curr, base):
        if curr is None or base is None:
            return np.nan
        if not np.isfinite(curr) or not np.isfinite(base) or abs(base) < 1e-12:
            return np.nan
        return 100.0 * (curr - base) / base

    for name, t0, t1 in cfg.segments:
        seg = df[(df["beat_time_s"] >= t0) & (df["beat_time_s"] < t1)]
        if len(seg) < 3:
            continue

        row = {
            "segment": name,
            "n_beats": int(len(seg)),
            "n_reflection_ok": int(seg["reflection_quality_ok"].fillna(False).sum()),
            "n_sdppg_ok": int(seg["sdppg_quality_ok"].fillna(False).sum()),

            "median_crest_time_s": safe_median(seg["crest_time_s"].to_numpy()),
            "median_crest_time_ratio": safe_median(seg["crest_time_ratio"].to_numpy()),
            "median_rise_slope": safe_median(seg["rise_slope"].to_numpy()),

            "median_pulse_width_50_s": safe_median(seg["pulse_width_50_s"].to_numpy()),
            "median_pulse_width_50_ratio": safe_median(seg["pulse_width_50_ratio"].to_numpy()),

            "median_pulse_area_total": safe_median(seg["pulse_area_total"].to_numpy()),
            "median_pulse_area_systolic": safe_median(seg["pulse_area_systolic"].to_numpy()),
            "median_pulse_area_diastolic": safe_median(seg["pulse_area_diastolic"].to_numpy()),

            "median_systolic_area_fraction": safe_median(seg["systolic_area_fraction"].to_numpy()),
            "median_diastolic_area_fraction": safe_median(seg["diastolic_area_fraction"].to_numpy()),

            "median_reflection_time_s": _stat_if_enough(seg["reflection_time_s"].to_numpy(), min_n=5),
            "median_ai_ratio": _stat_if_enough(seg["augmentation_index_ratio"].to_numpy(), min_n=5),
            "median_stiffness_index": _stat_if_enough(seg["stiffness_index"].to_numpy(), min_n=5),

            "median_b_over_a": _stat_if_enough(seg["b_over_a"].to_numpy(), min_n=10),
            "median_c_over_a": _stat_if_enough(seg["c_over_a"].to_numpy(), min_n=10),
            "median_d_over_a": _stat_if_enough(seg["d_over_a"].to_numpy(), min_n=10),
            "median_e_over_a": _stat_if_enough(seg["e_over_a"].to_numpy(), min_n=10),
            "median_sdppg_aging_index": _stat_if_enough(seg["sdppg_aging_index"].to_numpy(), min_n=10),
        }

        if name == "baseline":
            baseline_ref = row.copy()

        if baseline_ref is not None:
            row["delta_vs_baseline_crest_time_pct"] = pct_delta(
                row["median_crest_time_s"], baseline_ref["median_crest_time_s"]
            )
            row["delta_vs_baseline_crest_ratio_pct"] = pct_delta(
                row["median_crest_time_ratio"], baseline_ref["median_crest_time_ratio"]
            )
            row["delta_vs_baseline_rise_slope_pct"] = pct_delta(
                row["median_rise_slope"], baseline_ref["median_rise_slope"]
            )
            row["delta_vs_baseline_pw50_pct"] = pct_delta(
                row["median_pulse_width_50_s"], baseline_ref["median_pulse_width_50_s"]
            )
            row["delta_vs_baseline_pw50_ratio_pct"] = pct_delta(
                row["median_pulse_width_50_ratio"], baseline_ref["median_pulse_width_50_ratio"]
            )
            row["delta_vs_baseline_systolic_area_fraction_pct"] = pct_delta(
                row["median_systolic_area_fraction"], baseline_ref["median_systolic_area_fraction"]
            )
            row["delta_vs_baseline_diastolic_area_fraction_pct"] = pct_delta(
                row["median_diastolic_area_fraction"], baseline_ref["median_diastolic_area_fraction"]
            )
        else:
            row["delta_vs_baseline_crest_time_pct"] = np.nan
            row["delta_vs_baseline_crest_ratio_pct"] = np.nan
            row["delta_vs_baseline_rise_slope_pct"] = np.nan
            row["delta_vs_baseline_pw50_pct"] = np.nan
            row["delta_vs_baseline_pw50_ratio_pct"] = np.nan
            row["delta_vs_baseline_systolic_area_fraction_pct"] = np.nan
            row["delta_vs_baseline_diastolic_area_fraction_pct"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def _pick_representative(df: pd.DataFrame) -> Optional[int]:
    if df.empty:
        return None
    target_amp = safe_median(df["systolic_amplitude"].to_numpy())
    idx = (df["systolic_amplitude"] - target_amp).abs().idxmin()
    return int(idx)


def run_stiffness(t_s: np.ndarray, raw: np.ndarray, cfg: AnalysisConfig) -> Dict[str, Any]:
    fs = estimate_fs_from_time(t_s)
    if fs is None:
        raise RuntimeError("Could not estimate sampling rate.")

    beats = detect_beats(raw, t_s, fs, cfg)
    waveform_window_hint = _adaptive_window_hint(fs, span_s=0.12, minimum=5, maximum=11)
    derivative_window_hint = _adaptive_window_hint(fs, span_s=0.08, minimum=5, maximum=9)
    template_window_hint = _adaptive_window_hint(fs, span_s=0.10, minimum=5, maximum=11)
    rows: list[dict[str, Any]] = []
    waveforms: dict[int, dict[str, np.ndarray]] = {}

    for peak_idx in beats.accepted_peak_idx:
        prev_tr = beats.troughs[beats.troughs < peak_idx]
        next_tr = beats.troughs[beats.troughs > peak_idx]
        if len(prev_tr) == 0 or len(next_tr) == 0:
            continue

        start_i = int(prev_tr[-1])
        end_i = int(next_tr[0])
        if end_i - start_i < max(6, int(0.25 * fs)):
            continue

        t_full = t_s[start_i : end_i + 1]
        y_full = raw[start_i : end_i + 1]
        if len(t_full) < 8:
            continue

        y_full_sm = _smooth_beat(y_full, window_hint=waveform_window_hint)

        rel_peak_seed = int(peak_idx - start_i)
        rel_peak_i_full = _refine_systolic_peak(y_full_sm, rel_peak_seed, fs)
        onset_i_full = _refine_onset(y_full_sm, rel_peak_i_full, fs)

        t_beat = t_full[onset_i_full:]
        y_beat = y_full[onset_i_full:]
        y_sm = y_full_sm[onset_i_full:]
        rel_peak_i = int(rel_peak_i_full - onset_i_full)

        if len(t_beat) < 8 or rel_peak_i < 2 or rel_peak_i >= len(t_beat) - 2:
            continue

        onset_t = float(t_beat[0])
        peak_t = float(t_beat[rel_peak_i])
        end_t = float(t_beat[-1])

        onset_y = float(y_sm[0])
        peak_y = float(y_sm[rel_peak_i])

        beat_dur = end_t - onset_t
        crest_time = peak_t - onset_t
        syst_amp = peak_y - onset_y

        if beat_dur <= 0 or crest_time <= 0 or syst_amp <= 0:
            continue

        pulse_width_25 = _pulse_width(t_beat, y_sm, onset_y, peak_y, 0.25)
        pulse_width_50 = _pulse_width(t_beat, y_sm, onset_y, peak_y, 0.50)
        pulse_width_75 = _pulse_width(t_beat, y_sm, onset_y, peak_y, 0.75)

        area_total = _segment_area(t_beat, y_sm, onset_y)
        area_syst = _segment_area(t_beat[: rel_peak_i + 1], y_sm[: rel_peak_i + 1], onset_y)
        area_diast = _segment_area(t_beat[rel_peak_i:], y_sm[rel_peak_i:], onset_y)

        beat_quality_ok = True
        crest_ratio = crest_time / beat_dur

        if not (MIN_BEAT_DUR_S <= beat_dur <= MAX_BEAT_DUR_S):
            beat_quality_ok = False
        if not (MIN_CREST_RATIO <= crest_ratio <= MAX_CREST_RATIO):
            beat_quality_ok = False
        if not (np.isfinite(pulse_width_50) and MIN_PW50_S <= pulse_width_50 <= MAX_PW50_S):
            beat_quality_ok = False
        if area_total <= 0:
            beat_quality_ok = False

        notch_t, notch_y, dias_t, dias_y = _detect_notch_and_diastolic(t_beat, y_sm, rel_peak_i)
        reflection_time = dias_t - peak_t if np.isfinite(dias_t) else np.nan
        ai_ratio = (dias_y - onset_y) / syst_amp if np.isfinite(dias_y) else np.nan
        ai_delta = (dias_y - peak_y) / syst_amp if np.isfinite(dias_y) else np.nan

        stiffness_index = np.nan
        if cfg.subject_height_m and np.isfinite(reflection_time) and reflection_time > 1e-6:
            stiffness_index = cfg.subject_height_m / reflection_time

        vpg, apg = _derivatives(y_sm, t_beat, window_hint=derivative_window_hint)
        apg_pts = _detect_apg_points(t_beat - onset_t, apg)

        a = apg_pts["a"]
        b = apg_pts["b"]
        c = apg_pts["c"]
        d = apg_pts["d"]
        e = apg_pts["e"]

        row = {
            "beat_time_s": peak_t,
            "onset_time_s": onset_t,
            "beat_end_time_s": end_t,
            "beat_duration_s": beat_dur,
            "onset_level": onset_y,
            "systolic_peak_level": peak_y,
            "systolic_amplitude": syst_amp,
            "crest_time_s": crest_time,
            "crest_time_ratio": crest_ratio,
            "rise_slope": syst_amp / crest_time,
            "decay_time_s": end_t - peak_t,
            "pulse_width_25_s": pulse_width_25,
            "pulse_width_50_s": pulse_width_50,
            "pulse_width_75_s": pulse_width_75,
            "pulse_area_total": area_total,
            "pulse_area_systolic": area_syst,
            "pulse_area_diastolic": area_diast,
            "pulse_width_25_ratio": _safe_fraction(pulse_width_25, beat_dur),
            "pulse_width_50_ratio": _safe_fraction(pulse_width_50, beat_dur),
            "pulse_width_75_ratio": _safe_fraction(pulse_width_75, beat_dur),
            "systolic_area_fraction": _safe_fraction(area_syst, area_total),
            "diastolic_area_fraction": _safe_fraction(area_diast, area_total),
            "notch_time_s": notch_t,
            "notch_level": notch_y,
            "diastolic_peak_time_s": dias_t,
            "diastolic_peak_level": dias_y,
            "reflection_time_s": reflection_time,
            "augmentation_index_ratio": ai_ratio,
            "augmentation_index_delta": ai_delta,
            "stiffness_index": stiffness_index,
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e,
            "a_over_a": _ratio(a, a),
            "b_over_a": _ratio(b, a),
            "c_over_a": _ratio(c, a),
            "d_over_a": _ratio(d, a),
            "e_over_a": _ratio(e, a),
            "sdppg_aging_index": _ratio((b - c - d - e), a),
            "sdppg_substitute_index": _ratio((b - e), a),
            "beat_quality_ok": beat_quality_ok,
            "reflection_quality_ok": False,
            "sdppg_quality_ok": False,
        }

        row.update(apg_pts)
        row = _validate_reflection(row)
        row = _validate_sdppg(row)
        rows.append(row)

        waveforms[len(rows) - 1] = {
            "time_s": t_beat - onset_t,
            "ppg": y_sm,
            "vpg": vpg,
            "apg": apg,
        }

    beats_df = pd.DataFrame(rows)
    if beats_df.empty:
        raise RuntimeError("No usable beats for stiffness/morphology analysis.")

    amp_med = safe_median(beats_df["systolic_amplitude"].to_numpy())
    amp_thr = max(MIN_ABS_AMP, 0.45 * amp_med)

    beats_df["amp_quality_ok"] = beats_df["systolic_amplitude"] >= amp_thr
    beats_df["morphology_quality_ok"] = beats_df["beat_quality_ok"] & beats_df["amp_quality_ok"]
    beats_df = beats_df[beats_df["morphology_quality_ok"]].copy()

    if len(beats_df) < 8:
        raise RuntimeError("Too few morphology-valid beats after quality filtering.")

    rep_idx = _pick_representative(beats_df)
    representative = waveforms.get(rep_idx) if rep_idx is not None else None

    segments_df = _summarize_segments(beats_df, cfg)
    templates_df = _build_templates(beats_df, waveforms, cfg)
    template_landmarks_df = _summarize_template_landmarks(templates_df, template_window_hint=template_window_hint)
    template_overall_available = (not template_landmarks_df.empty) and (template_landmarks_df["segment"] == "overall").any()
    template_notch_ok = (
        bool(template_landmarks_df.loc[template_landmarks_df["segment"] == "overall", "template_notch_quality_ok"].iloc[0])
        if template_overall_available
        else False
    )
    num_reflection_ok = int(beats_df["reflection_quality_ok"].fillna(False).sum())
    num_sdppg_ok = int(beats_df["sdppg_quality_ok"].fillna(False).sum())
    reflection_coverage = num_reflection_ok / len(beats_df)
    sdppg_coverage = num_sdppg_ok / len(beats_df)
    reflection_features_available = bool(template_notch_ok or num_reflection_ok >= 5)
    sdppg_features_available = bool(num_sdppg_ok >= 10)
    fine_morphology_available = bool(reflection_features_available or sdppg_features_available)
    if reflection_features_available and sdppg_features_available:
        analysis_scope = "full_morphology"
        quality_note = None
    elif fine_morphology_available:
        analysis_scope = "partial_fine_morphology"
        if sdppg_features_available and not reflection_features_available:
            quality_note = "SDPPG landmarks passed quality checks, but reflected-wave landmarks did not; use derivative ratios cautiously and treat reflection/stiffness metrics as unavailable."
        else:
            quality_note = "Reflected-wave landmarks were detected, but SDPPG landmarks were insufficient; treat derivative-based metrics as unavailable."
    else:
        analysis_scope = "coarse_morphology_only"
        quality_note = "No robust reflected-wave or SDPPG landmarks were detected; interpret crest time, width, slope, and pulse-area features as coarse trend metrics only."

    summary = {
        "estimated_fs_hz": fs,
        "num_candidate_beats": int(len(rows)),
        "num_accepted_beats": int(len(beats_df)),
        "amp_threshold_counts": float(amp_thr),
        "analysis_scope": analysis_scope,
        "fine_morphology_available": fine_morphology_available,
        "reflection_features_available": reflection_features_available,
        "sdppg_features_available": sdppg_features_available,
        "coarse_morphology_available": True,
        "reflection_landmark_coverage": float(reflection_coverage),
        "sdppg_landmark_coverage": float(sdppg_coverage),
        "quality_note": quality_note,
        "waveform_smoothing_window_samples": int(waveform_window_hint),
        "derivative_smoothing_window_samples": int(derivative_window_hint),
        "template_smoothing_window_samples": int(template_window_hint),
        "num_reflection_ok": num_reflection_ok,
        "num_sdppg_ok": num_sdppg_ok,
        "median_crest_time_s": safe_median(beats_df["crest_time_s"].to_numpy()),
        "median_crest_time_ratio": safe_median(beats_df["crest_time_ratio"].to_numpy()),
        "median_rise_slope": safe_median(beats_df["rise_slope"].to_numpy()),
        "median_pulse_width_50_s": safe_median(beats_df["pulse_width_50_s"].to_numpy()),
        "median_pulse_width_50_ratio": safe_median(beats_df["pulse_width_50_ratio"].to_numpy()),
        "median_systolic_amplitude": safe_median(beats_df["systolic_amplitude"].to_numpy()),
        "median_pulse_area_total": safe_median(beats_df["pulse_area_total"].to_numpy()),
        "median_pulse_area_systolic": safe_median(beats_df["pulse_area_systolic"].to_numpy()),
        "median_pulse_area_diastolic": safe_median(beats_df["pulse_area_diastolic"].to_numpy()),
        "median_systolic_area_fraction": safe_median(beats_df["systolic_area_fraction"].to_numpy()),
        "median_diastolic_area_fraction": safe_median(beats_df["diastolic_area_fraction"].to_numpy()),
        "template_overall_peak_u": (
            float(template_landmarks_df.loc[template_landmarks_df["segment"] == "overall", "template_peak_u"].iloc[0])
            if template_overall_available
            else np.nan
        ),
        "template_overall_pw50_u": (
            float(template_landmarks_df.loc[template_landmarks_df["segment"] == "overall", "template_pw50_u"].iloc[0])
            if template_overall_available
            else np.nan
        ),
        "template_overall_notch_quality_ok": template_notch_ok,
        "template_overall_reflection_u": (
            float(template_landmarks_df.loc[template_landmarks_df["segment"] == "overall", "template_reflection_u"].iloc[0])
            if template_overall_available
            else np.nan
        ),
        "template_overall_ai_ratio": (
            float(template_landmarks_df.loc[template_landmarks_df["segment"] == "overall", "template_ai_ratio"].iloc[0])
            if template_overall_available
            else np.nan
        ),
        "median_reflection_time_s": _stat_if_enough(beats_df["reflection_time_s"].to_numpy(), min_n=5),
        "median_augmentation_index_ratio": _stat_if_enough(beats_df["augmentation_index_ratio"].to_numpy(), min_n=5),
        "median_augmentation_index_delta": _stat_if_enough(beats_df["augmentation_index_delta"].to_numpy(), min_n=5),
        "median_stiffness_index": _stat_if_enough(beats_df["stiffness_index"].to_numpy(), min_n=5),
        "median_b_over_a": _stat_if_enough(beats_df["b_over_a"].to_numpy(), min_n=10),
        "median_c_over_a": _stat_if_enough(beats_df["c_over_a"].to_numpy(), min_n=10),
        "median_d_over_a": _stat_if_enough(beats_df["d_over_a"].to_numpy(), min_n=10),
        "median_e_over_a": _stat_if_enough(beats_df["e_over_a"].to_numpy(), min_n=10),
        "median_sdppg_aging_index": _stat_if_enough(beats_df["sdppg_aging_index"].to_numpy(), min_n=10),
        "mean_sdppg_aging_index": _stat_if_enough(beats_df["sdppg_aging_index"].to_numpy(), min_n=10, reducer="mean"),
        "subject_height_m": cfg.subject_height_m,
    }

    tables: Dict[str, pd.DataFrame] = {
        "stiffness_beats": beats_df,
        "stiffness_segments": segments_df,
        "stiffness_templates": templates_df,
        "stiffness_template_landmarks": template_landmarks_df,
    }
    if representative is not None:
        tables["stiffness_representative_waveform"] = pd.DataFrame(representative)

    return {
        "summary": summary,
        "tables": tables,
        "artifacts": {
            "beats": beats,
        },
    }
