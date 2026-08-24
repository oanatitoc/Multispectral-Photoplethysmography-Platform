from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np


def default_red_nir_12ch_calibration_path(dataset_dir: str | Path) -> Path:
    return Path(dataset_dir) / "calibration" / "red_nir_12ch_calibration.json"


def load_red_nir_12ch_calibration(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def calibration_display_name(calibration: dict[str, Any]) -> str:
    return str(calibration.get("calibration_name") or calibration.get("created_at") or "cohort_calibration")


def resolve_spo2_params(
    calibration: dict[str, Any],
    *,
    anchor_ratio: float | None,
    anchor_pct: float | None,
    slope: float | None,
) -> tuple[float | None, float | None, float, float, float]:
    spo2 = calibration.get("spo2", {}) if isinstance(calibration, dict) else {}
    clip_min = float(spo2.get("clip_min_pct", 70.0))
    clip_max = float(spo2.get("clip_max_pct", 100.0))
    out_anchor_ratio = anchor_ratio
    out_anchor_pct = anchor_pct
    out_slope = float(slope) if slope is not None else 25.0

    if spo2.get("enabled"):
        if out_anchor_ratio is None:
            out_anchor_ratio = _finite_or_none(spo2.get("anchor_ratio"))
        if out_anchor_pct is None:
            out_anchor_pct = _finite_or_none(spo2.get("anchor_pct"))
        if slope is None:
            out_slope = float(spo2.get("slope_pct_per_ratio", 25.0))

    return out_anchor_ratio, out_anchor_pct, out_slope, clip_min, clip_max


def get_pi_model(calibration: dict[str, Any]) -> dict[str, Any]:
    model = calibration.get("perfusion_index", {}) if isinstance(calibration, dict) else {}
    return model if model.get("enabled") else {}


def apply_pi_calibration(proxy_pct: float | None, calibration: dict[str, Any]) -> float | None:
    proxy = _finite_or_none(proxy_pct)
    if proxy is None:
        return None
    model = get_pi_model(calibration)
    if not model:
        return proxy
    intercept = _finite_or_none(model.get("intercept_pct"))
    slope = _finite_or_none(model.get("coef_pct_per_proxy_pct"))
    if intercept is None or slope is None:
        return proxy
    value = float(intercept + slope * proxy)
    clip_min = float(model.get("clip_min_pct", 0.0))
    clip_max = float(model.get("clip_max_pct", 20.0))
    return float(np.clip(value, clip_min, clip_max))


def get_rr_protocol_model(calibration: dict[str, Any], protocol: str | None) -> dict[str, Any]:
    if protocol is None:
        return {}
    rr_models = calibration.get("respiration", {}) if isinstance(calibration, dict) else {}
    model = rr_models.get(protocol, {})
    return model if model.get("enabled") else {}


def apply_rr_calibration(
    value_brpm: float | None,
    calibration: dict[str, Any],
    protocol: str | None,
    *,
    source_feature: str | None = None,
) -> float | None:
    value = _finite_or_none(value_brpm)
    if value is None:
        return None
    model = get_rr_protocol_model(calibration, protocol)
    if not model:
        return value
    model_source = model.get("source_feature")
    if source_feature is not None and model_source not in (None, source_feature):
        return value
    intercept = _finite_or_none(model.get("intercept_brpm"))
    slope = _finite_or_none(model.get("coef_brpm_per_brpm"))
    if intercept is None or slope is None:
        return value
    corrected = float(intercept + slope * value)
    clip_min = float(model.get("clip_min_brpm", 4.0))
    clip_max = float(model.get("clip_max_brpm", 45.0))
    return float(np.clip(corrected, clip_min, clip_max))


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None
