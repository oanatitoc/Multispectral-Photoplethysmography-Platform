from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor, LinearRegression, RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from ppg_suite.io_utils import save_json


CORE_PROTOCOLS = (
    "controlled_breathing",
    "post_exercise_recovery",
    "post_cold_recovery",
)

SCSS_HIGH_CONF_ID = "scss_core_high_conf"
SCSS_ALL_PLAUSIBLE_ID = "scss_core_plausible"
RR_HIGH_CONF_ID = "controlled_breathing_high_conf"
RR_ALL_PLAUSIBLE_ID = "controlled_breathing_plausible"
PI_HIGH_CONF_ID = "pi_core_high_conf"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    input_description: str
    comment: str
    feature_cols: tuple[str, ...]
    factory: Callable[[], Any] | None = None
    baseline_col: str | None = None


def ensure_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    work = df.copy()
    for col in columns:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    return work


def load_snapshot_master(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric_cols = [
        "timestamp_rel_s",
        "matched_live_time_min_s",
        "matched_live_time_max_s",
        "matched_live_rows",
        "gt_pulseox_hr_bpm",
        "gt_pulseox_spo2_pct",
        "gt_pulseox_pi_pct",
        "gt_rr_ref_bpm",
        "live_hr_bpm_median",
        "live_hr_fft_bpm_median",
        "live_respiratory_rate_brpm_median",
        "live_spo2_estimated_pct_median",
        "live_spo2_ratio_median",
        "live_perfusion_index_pct_median",
        "live_pulse_amplitude_median",
        "live_signal_quality_median",
        "live_artifact_flag_median",
        "postrun_respiration_final_brpm",
        "postrun_rr_error_vs_reference_brpm",
        "beat_window_count",
        "ratio_of_ratios_median",
        "F6_acdc_median",
        "NIR_acdc_median",
        "F6_ac_median",
        "F6_dc_median",
        "NIR_ac_median",
        "NIR_dc_median",
        "F6_pi_proxy_pct_median",
        "NIR_pi_proxy_pct_median",
    ]
    return ensure_numeric(df, numeric_cols)


def add_snapshot_flags(
    df: pd.DataFrame,
    *,
    stabilization_s: float = 12.0,
    min_beats_high: int = 18,
    min_beats_medium: int = 10,
    min_signal_high: float = 8.5,
    min_signal_medium: float = 6.0,
    min_amp_high: float = 18.0,
    min_amp_medium: float = 12.0,
) -> pd.DataFrame:
    work = df.copy()

    work["plausible_hr_reference"] = work["gt_pulseox_hr_bpm"].isna() | work["gt_pulseox_hr_bpm"].between(40.0, 190.0)
    work["plausible_spo2_reference"] = work["gt_pulseox_spo2_pct"].isna() | work["gt_pulseox_spo2_pct"].between(85.0, 100.0)
    work["plausible_pi_reference"] = work["gt_pulseox_pi_pct"].isna() | work["gt_pulseox_pi_pct"].between(0.0, 25.0)
    work["plausible_rr_reference"] = work["gt_rr_ref_bpm"].isna() | work["gt_rr_ref_bpm"].between(5.0, 40.0)
    work["plausible_reference"] = (
        work["plausible_hr_reference"]
        & work["plausible_spo2_reference"]
        & work["plausible_pi_reference"]
        & work["plausible_rr_reference"]
    )

    work["in_scss_core"] = work["protocol"].isin(CORE_PROTOCOLS)
    work["is_transition_window"] = work["matched_live_time_min_s"].fillna(np.inf) < float(stabilization_s)
    work["has_enough_beats_high"] = work["beat_window_count"].fillna(0.0) >= float(min_beats_high)
    work["has_enough_beats_medium"] = work["beat_window_count"].fillna(0.0) >= float(min_beats_medium)
    work["signal_quality_high"] = work["live_signal_quality_median"].fillna(0.0) >= float(min_signal_high)
    work["signal_quality_medium"] = work["live_signal_quality_median"].fillna(0.0) >= float(min_signal_medium)
    work["pulse_amplitude_high"] = work["live_pulse_amplitude_median"].fillna(0.0) >= float(min_amp_high)
    work["pulse_amplitude_medium"] = work["live_pulse_amplitude_median"].fillna(0.0) >= float(min_amp_medium)
    work["artifact_free"] = work["live_artifact_flag_median"].fillna(0.0) <= 0.5

    high_mask = (
        work["plausible_reference"]
        & work["in_scss_core"]
        & ~work["is_transition_window"]
        & work["has_enough_beats_high"]
        & work["signal_quality_high"]
        & work["pulse_amplitude_high"]
        & work["artifact_free"]
    )
    medium_mask = (
        work["plausible_reference"]
        & work["in_scss_core"]
        & work["has_enough_beats_medium"]
        & work["signal_quality_medium"]
        & work["pulse_amplitude_medium"]
        & work["artifact_free"]
    )

    work["snapshot_confidence"] = np.select([high_mask, medium_mask], ["high", "medium"], default="low")
    work["quality_flag"] = np.select([high_mask, medium_mask], ["good", "warning"], default="bad")
    work["usable_for_scss_figures"] = work["snapshot_confidence"].eq("high")
    return work


def clip_values(values: np.ndarray, limits: tuple[float, float] | None) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    if limits is None:
        return out
    return np.clip(out, limits[0], limits[1])


def loso_predictions(
    df: pd.DataFrame,
    *,
    target_col: str,
    model: ModelSpec,
    clip: tuple[float, float] | None = None,
) -> pd.DataFrame:
    needed = ["subject_id", "run_name", "protocol", "timestamp_rel_s", target_col]
    if model.baseline_col is not None:
        needed.append(model.baseline_col)
    else:
        needed.extend(model.feature_cols)
    work = df[needed].dropna().copy()
    if len(work) == 0:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for subject_id in sorted(work["subject_id"].dropna().unique()):
        train = work[work["subject_id"] != subject_id]
        test = work[work["subject_id"] == subject_id]
        if len(train) == 0 or len(test) == 0:
            continue

        if model.baseline_col is not None:
            pred = test[model.baseline_col].to_numpy(dtype=float)
        else:
            estimator = model.factory()
            estimator.fit(train[list(model.feature_cols)].to_numpy(dtype=float), train[target_col].to_numpy(dtype=float))
            pred = estimator.predict(test[list(model.feature_cols)].to_numpy(dtype=float))

        pred = clip_values(pred, clip)
        truth = test[target_col].to_numpy(dtype=float)
        err = pred - truth

        for idx, (_, row) in enumerate(test.iterrows()):
            rows.append(
                {
                    "subject_id": row["subject_id"],
                    "run_name": row["run_name"],
                    "protocol": row["protocol"],
                    "timestamp_rel_s": float(row["timestamp_rel_s"]),
                    "target": float(truth[idx]),
                    "prediction": float(pred[idx]),
                    "error": float(err[idx]),
                }
            )
    return pd.DataFrame(rows)


def metrics_from_predictions(pred_df: pd.DataFrame) -> dict[str, float]:
    if len(pred_df) == 0:
        return {"mae": np.nan, "rmse": np.nan, "bias": np.nan}
    err = pred_df["error"].to_numpy(dtype=float)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "bias": float(np.mean(err)),
    }


def summarize_models(
    task: str,
    subset_id: str,
    df: pd.DataFrame,
    *,
    target_col: str,
    clip: tuple[float, float] | None,
    specs: list[ModelSpec],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparison_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for spec in specs:
        pred_df = loso_predictions(df, target_col=target_col, model=spec, clip=clip)
        if len(pred_df) == 0:
            continue
        prediction_frames.append(pred_df.assign(task=task, subset_id=subset_id, model_id=spec.model_id))
        stats = metrics_from_predictions(pred_df)
        comparison_rows.append(
            {
                "task": task,
                "subset_id": subset_id,
                "model_id": spec.model_id,
                "display_name": spec.display_name,
                "input_description": spec.input_description,
                "comment": spec.comment,
                "n_rows": int(len(pred_df)),
                "n_subjects": int(pred_df["subject_id"].nunique()),
                "mae": stats["mae"],
                "rmse": stats["rmse"],
                "bias": stats["bias"],
            }
        )

    comparison_df = pd.DataFrame(comparison_rows).sort_values(["task", "subset_id", "mae", "rmse"]).reset_index(drop=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return comparison_df, predictions_df


def spo2_model_specs() -> list[ModelSpec]:
    spectral_cols = (
        "ratio_of_ratios_median",
        "F6_acdc_median",
        "NIR_acdc_median",
        "F6_ac_median",
        "F6_dc_median",
        "NIR_ac_median",
        "NIR_dc_median",
        "F6_pi_proxy_pct_median",
        "NIR_pi_proxy_pct_median",
        "live_signal_quality_median",
        "beat_window_count",
    )
    return [
        ModelSpec(
            model_id="existing_live_estimate",
            display_name="Existing live estimate",
            input_description="Current live SpO2-style output",
            comment="Baseline from the existing pipeline before retraining.",
            feature_cols=(),
            baseline_col="live_spo2_estimated_pct_median",
        ),
        ModelSpec(
            model_id="linear_ratio",
            display_name="Linear Regression",
            input_description="R ratio only",
            comment="Most interpretable cohort calibration.",
            feature_cols=("ratio_of_ratios_median",),
            factory=lambda: LinearRegression(),
        ),
        ModelSpec(
            model_id="huber_ratio",
            display_name="Huber Regression",
            input_description="R ratio only",
            comment="Robust to a small number of outliers.",
            feature_cols=("ratio_of_ratios_median",),
            factory=lambda: make_pipeline(StandardScaler(), HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=500)),
        ),
        ModelSpec(
            model_id="ransac_ratio",
            display_name="RANSAC Regression",
            input_description="R ratio only",
            comment="Fits a line while down-weighting inconsistent points.",
            feature_cols=("ratio_of_ratios_median",),
            factory=lambda: RANSACRegressor(estimator=LinearRegression(), random_state=42),
        ),
        ModelSpec(
            model_id="rf_spectral",
            display_name="Random Forest",
            input_description="Spectral + quality features",
            comment="Flexible nonlinear model; watch for overfitting.",
            feature_cols=spectral_cols,
            factory=lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=4, random_state=42),
        ),
        ModelSpec(
            model_id="gbr_spectral",
            display_name="Gradient Boosting",
            input_description="Spectral + quality features",
            comment="Stronger nonlinear fit, but less interpretable.",
            feature_cols=spectral_cols,
            factory=lambda: GradientBoostingRegressor(
                random_state=42,
                n_estimators=250,
                max_depth=2,
                learning_rate=0.05,
            ),
        ),
    ]


def rr_model_specs() -> list[ModelSpec]:
    multi_cols = (
        "live_respiratory_rate_brpm_median",
        "postrun_respiration_final_brpm",
        "live_hr_bpm_median",
        "live_hr_fft_bpm_median",
        "live_signal_quality_median",
        "beat_window_count",
    )
    return [
        ModelSpec(
            model_id="live_raw",
            display_name="Live raw RR",
            input_description="Live RR estimate",
            comment="No calibration; direct live output.",
            feature_cols=(),
            baseline_col="live_respiratory_rate_brpm_median",
        ),
        ModelSpec(
            model_id="postrun_raw",
            display_name="Post-run raw RR",
            input_description="Post-run RR estimate",
            comment="Best existing estimator before ML correction.",
            feature_cols=(),
            baseline_col="postrun_respiration_final_brpm",
        ),
        ModelSpec(
            model_id="linear_postrun",
            display_name="Linear Regression",
            input_description="Post-run RR only",
            comment="Simple calibration on the strongest single feature.",
            feature_cols=("postrun_respiration_final_brpm",),
            factory=lambda: LinearRegression(),
        ),
        ModelSpec(
            model_id="huber_postrun",
            display_name="Huber Regression",
            input_description="Post-run RR only",
            comment="Robust correction for controlled breathing.",
            feature_cols=("postrun_respiration_final_brpm",),
            factory=lambda: make_pipeline(StandardScaler(), HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=500)),
        ),
        ModelSpec(
            model_id="ransac_postrun",
            display_name="RANSAC Regression",
            input_description="Post-run RR only",
            comment="Robust line fit on the post-run estimator.",
            feature_cols=("postrun_respiration_final_brpm",),
            factory=lambda: RANSACRegressor(estimator=LinearRegression(), random_state=42),
        ),
        ModelSpec(
            model_id="rf_multi",
            display_name="Random Forest",
            input_description="Live + post-run + quality features",
            comment="Flexible nonlinear alternative.",
            feature_cols=multi_cols,
            factory=lambda: RandomForestRegressor(n_estimators=250, min_samples_leaf=4, random_state=42),
        ),
        ModelSpec(
            model_id="gbr_multi",
            display_name="Gradient Boosting",
            input_description="Live + post-run + quality features",
            comment="Boosted tree model for comparison.",
            feature_cols=multi_cols,
            factory=lambda: GradientBoostingRegressor(
                random_state=42,
                n_estimators=250,
                max_depth=2,
                learning_rate=0.05,
            ),
        ),
    ]


def pi_model_specs() -> list[ModelSpec]:
    multi_cols = (
        "F6_pi_proxy_pct_median",
        "NIR_pi_proxy_pct_median",
        "live_perfusion_index_pct_median",
        "live_signal_quality_median",
        "beat_window_count",
    )
    return [
        ModelSpec(
            model_id="live_pi_raw",
            display_name="Existing live PI",
            input_description="Current live PI proxy",
            comment="Absolute scale before additional training.",
            feature_cols=(),
            baseline_col="live_perfusion_index_pct_median",
        ),
        ModelSpec(
            model_id="linear_f6",
            display_name="Linear Regression",
            input_description="F6 PI proxy only",
            comment="Simple absolute PI calibration candidate.",
            feature_cols=("F6_pi_proxy_pct_median",),
            factory=lambda: LinearRegression(),
        ),
        ModelSpec(
            model_id="huber_f6",
            display_name="Huber Regression",
            input_description="F6 PI proxy only",
            comment="Robust PI regression candidate.",
            feature_cols=("F6_pi_proxy_pct_median",),
            factory=lambda: make_pipeline(StandardScaler(), HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=500)),
        ),
        ModelSpec(
            model_id="rf_pi",
            display_name="Random Forest",
            input_description="PI + quality features",
            comment="Nonlinear PI comparison model.",
            feature_cols=multi_cols,
            factory=lambda: RandomForestRegressor(n_estimators=250, min_samples_leaf=4, random_state=42),
        ),
        ModelSpec(
            model_id="gbr_pi",
            display_name="Gradient Boosting",
            input_description="PI + quality features",
            comment="Boosted PI comparison model.",
            feature_cols=multi_cols,
            factory=lambda: GradientBoostingRegressor(
                random_state=42,
                n_estimators=250,
                max_depth=2,
                learning_rate=0.05,
            ),
        ),
    ]


def filter_spo2(df: pd.DataFrame, *, high_conf: bool, core_only: bool) -> pd.DataFrame:
    work = df.copy()
    mask = work["plausible_spo2_reference"] & work["ratio_of_ratios_median"].gt(0.0)
    if core_only:
        mask &= work["in_scss_core"]
    if high_conf:
        mask &= work["snapshot_confidence"].eq("high")
    return work.loc[mask].copy()


def filter_rr_controlled(df: pd.DataFrame, *, high_conf: bool) -> pd.DataFrame:
    work = df.copy()
    mask = work["protocol"].eq("controlled_breathing") & work["plausible_rr_reference"]
    mask &= work["postrun_respiration_final_brpm"].gt(0.0)
    if high_conf:
        mask &= work["snapshot_confidence"].eq("high")
    return work.loc[mask].copy()


def filter_pi(df: pd.DataFrame, *, high_conf: bool) -> pd.DataFrame:
    work = df.copy()
    mask = work["in_scss_core"] & work["plausible_pi_reference"]
    mask &= work["F6_pi_proxy_pct_median"].gt(0.0)
    mask &= work["live_perfusion_index_pct_median"].gt(0.0)
    if high_conf:
        mask &= work["snapshot_confidence"].eq("high")
    return work.loc[mask].copy()


def best_row(df: pd.DataFrame, task: str, subset_id: str) -> dict[str, Any]:
    sub = df[(df["task"] == task) & (df["subset_id"] == subset_id)].copy()
    if len(sub) == 0:
        return {}
    row = sub.sort_values(["mae", "rmse", "bias"], key=lambda s: np.abs(s) if s.name == "bias" else s).iloc[0]
    return row.to_dict()


def snapshot_summary(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "total_rows": int(len(df)),
        "core_protocol_rows": int(df["in_scss_core"].sum()),
        "high_conf_rows": int(df["snapshot_confidence"].eq("high").sum()),
        "high_conf_core_rows": int((df["in_scss_core"] & df["snapshot_confidence"].eq("high")).sum()),
        "confidence_counts": df["snapshot_confidence"].value_counts(dropna=False).to_dict(),
    }


def filtering_summary(
    comparison_df: pd.DataFrame,
    *,
    task: str,
    before_subset: str,
    after_subset: str,
    preferred_model_id: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {"task": task, "model_id": preferred_model_id}
    before = comparison_df[
        (comparison_df["task"] == task)
        & (comparison_df["subset_id"] == before_subset)
        & (comparison_df["model_id"] == preferred_model_id)
    ]
    after = comparison_df[
        (comparison_df["task"] == task)
        & (comparison_df["subset_id"] == after_subset)
        & (comparison_df["model_id"] == preferred_model_id)
    ]
    if len(before):
        out["before"] = before.iloc[0][["n_rows", "n_subjects", "mae", "rmse", "bias"]].to_dict()
    if len(after):
        out["after"] = after.iloc[0][["n_rows", "n_subjects", "mae", "rmse", "bias"]].to_dict()
    if "before" in out and "after" in out:
        out["mae_improvement"] = float(out["before"]["mae"] - out["after"]["mae"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare cohort-level ML/regression models for SCSS results.")
    parser.add_argument("--snapshot-master", default=str(ROOT / "dataset" / "calibration" / "red_nir_12ch_snapshot_master.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "dataset" / "reports" / "ml_results"))
    args = parser.parse_args()

    snapshot_master_path = Path(args.snapshot_master)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_snapshot_master(snapshot_master_path)
    scored_df = add_snapshot_flags(raw_df)
    scored_path = output_dir / "red_nir_12ch_snapshot_master_scored.csv"
    scored_df.to_csv(scored_path, index=False)

    frames: list[pd.DataFrame] = []
    pred_frames: list[pd.DataFrame] = []

    spo2_all = filter_spo2(scored_df, high_conf=False, core_only=True)
    spo2_high = filter_spo2(scored_df, high_conf=True, core_only=True)
    rr_all = filter_rr_controlled(scored_df, high_conf=False)
    rr_high = filter_rr_controlled(scored_df, high_conf=True)
    pi_high = filter_pi(scored_df, high_conf=True)

    comp_df, pred_df = summarize_models(
        "spo2",
        SCSS_ALL_PLAUSIBLE_ID,
        spo2_all,
        target_col="gt_pulseox_spo2_pct",
        clip=(70.0, 100.0),
        specs=spo2_model_specs(),
    )
    frames.append(comp_df)
    pred_frames.append(pred_df)

    comp_df, pred_df = summarize_models(
        "spo2",
        SCSS_HIGH_CONF_ID,
        spo2_high,
        target_col="gt_pulseox_spo2_pct",
        clip=(70.0, 100.0),
        specs=spo2_model_specs(),
    )
    frames.append(comp_df)
    pred_frames.append(pred_df)

    comp_df, pred_df = summarize_models(
        "rr_controlled_breathing",
        RR_ALL_PLAUSIBLE_ID,
        rr_all,
        target_col="gt_rr_ref_bpm",
        clip=(4.0, 45.0),
        specs=rr_model_specs(),
    )
    frames.append(comp_df)
    pred_frames.append(pred_df)

    comp_df, pred_df = summarize_models(
        "rr_controlled_breathing",
        RR_HIGH_CONF_ID,
        rr_high,
        target_col="gt_rr_ref_bpm",
        clip=(4.0, 45.0),
        specs=rr_model_specs(),
    )
    frames.append(comp_df)
    pred_frames.append(pred_df)

    comp_df, pred_df = summarize_models(
        "pi_absolute",
        PI_HIGH_CONF_ID,
        pi_high,
        target_col="gt_pulseox_pi_pct",
        clip=(0.0, 25.0),
        specs=pi_model_specs(),
    )
    frames.append(comp_df)
    pred_frames.append(pred_df)

    comparison_df = pd.concat(frames, ignore_index=True)
    predictions_df = pd.concat(pred_frames, ignore_index=True)

    comparison_path = output_dir / "model_comparison.csv"
    predictions_path = output_dir / "model_predictions.csv"
    comparison_df.to_csv(comparison_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "snapshot_master": str(snapshot_master_path),
        "scored_snapshot_master": str(scored_path),
        "comparison_csv": str(comparison_path),
        "predictions_csv": str(predictions_path),
        "cohort": {
            "subjects_with_snapshots": int(raw_df["subject_id"].nunique()),
            "runs_with_snapshots": int(raw_df["run_name"].nunique()),
            "snapshot_rows": int(len(raw_df)),
            "core_protocols": list(CORE_PROTOCOLS),
            "protocol_counts": raw_df["protocol"].value_counts(dropna=False).to_dict(),
        },
        "snapshot_flags": snapshot_summary(scored_df),
        "best_models": {
            "spo2_high_conf": best_row(comparison_df, "spo2", SCSS_HIGH_CONF_ID),
            "rr_controlled_high_conf": best_row(comparison_df, "rr_controlled_breathing", RR_HIGH_CONF_ID),
            "pi_absolute_high_conf": best_row(comparison_df, "pi_absolute", PI_HIGH_CONF_ID),
        },
        "filtering_effect": {
            "spo2": filtering_summary(
                comparison_df,
                task="spo2",
                before_subset=SCSS_ALL_PLAUSIBLE_ID,
                after_subset=SCSS_HIGH_CONF_ID,
                preferred_model_id="linear_ratio",
            ),
            "rr_controlled_breathing": filtering_summary(
                comparison_df,
                task="rr_controlled_breathing",
                before_subset=RR_ALL_PLAUSIBLE_ID,
                after_subset=RR_HIGH_CONF_ID,
                preferred_model_id="huber_postrun",
            ),
        },
        "notes": [
            "Physiological plausibility flags mark suspicious references instead of deleting raw rows.",
            "High-confidence windows require non-transition timing, enough beats, adequate pulse amplitude, and good signal quality.",
            "SCSS plots focus on controlled breathing, post-exercise recovery, and post-cold recovery.",
        ],
    }
    summary_path = output_dir / "ml_summary.json"
    save_json(summary_path, summary)

    print(f"Saved: {scored_path}")
    print(f"Saved: {comparison_path}")
    print(f"Saved: {predictions_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
