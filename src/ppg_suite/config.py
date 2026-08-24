from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


DEFAULT_SEGMENTS: List[Tuple[str, float, float]] = [
    ("baseline", 0.0, 60.0),
    ("task", 60.0, 120.0),
    ("recovery", 120.0, 180.0),
]


@dataclass
class AnalysisConfig:
    channel: str = "NIR_diff"
    drop_start_sec: float = 3.0
    drop_end_sec: float = 3.0
    heart_band: Tuple[float, float] = (0.7, 4.0)
    resp_band: Tuple[float, float] = (0.15, 0.35)
    resp_fs: float = 4.0
    protocol: str | None = None
    resp_target_brpm: float | None = None
    vaso_band: Tuple[float, float] = (0.009, 0.15)
    trend_fs: float = 2.0
    min_rr_ms: float = 400.0
    max_rr_ms: float = 1500.0
    rr_local_median_tol: float = 0.25
    segments: List[Tuple[str, float, float]] = field(default_factory=lambda: list(DEFAULT_SEGMENTS))
    subject_height_m: float | None = None
    spo2_anchor_ratio: float | None = None
    spo2_anchor_pct: float | None = None
    spo2_linear_slope: float = 25.0
    spo2_clip_min_pct: float = 70.0
    spo2_clip_max_pct: float = 100.0
