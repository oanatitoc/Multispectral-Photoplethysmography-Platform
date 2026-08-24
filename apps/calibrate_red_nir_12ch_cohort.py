from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from ppg_suite.calibration import default_red_nir_12ch_calibration_path
from ppg_suite.io_utils import save_json


@dataclass
class LinearFit:
    slope: float
    intercept: float
    inlier_mask: np.ndarray
    train_mae: float
    train_rmse: float
    train_bias: float
    train_corr: float | None
    inlier_count: int
    total_count: int


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def robust_scale(values: np.ndarray, floor: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float(floor)
    mad = float(np.median(np.abs(values - np.median(values))))
    return float(max(1.4826 * mad, floor))


def robust_linear_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    min_points: int = 12,
    max_iter: int = 6,
    zmax: float = 3.5,
    scale_floor: float = 0.25,
) -> LinearFit | None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(valid)) < min_points:
        return None

    inliers = valid.copy()
    for _ in range(max_iter):
        if int(np.sum(inliers)) < min_points:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.exceptions.RankWarning)
            slope, intercept = np.polyfit(x[inliers], y[inliers], 1)
        residual = y - (slope * x + intercept)
        scale = robust_scale(residual[inliers], floor=scale_floor)
        new_inliers = valid & (np.abs(residual - np.median(residual[inliers])) <= zmax * scale)
        if int(np.sum(new_inliers)) == int(np.sum(inliers)):
            inliers = new_inliers
            break
        inliers = new_inliers

    if int(np.sum(inliers)) < min_points:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.RankWarning)
        slope, intercept = np.polyfit(x[inliers], y[inliers], 1)
    pred = slope * x[inliers] + intercept
    err = pred - y[inliers]
    corr = None
    if int(np.sum(inliers)) >= 3 and np.std(x[inliers]) > 1e-9 and np.std(y[inliers]) > 1e-9:
        corr = float(np.corrcoef(x[inliers], y[inliers])[0, 1])
    return LinearFit(
        slope=float(slope),
        intercept=float(intercept),
        inlier_mask=inliers,
        train_mae=float(np.mean(np.abs(err))),
        train_rmse=float(np.sqrt(np.mean(err**2))),
        train_bias=float(np.mean(err)),
        train_corr=corr,
        inlier_count=int(np.sum(inliers)),
        total_count=int(np.sum(valid)),
    )


def leave_one_subject_out_metrics(df: pd.DataFrame, x_col: str, y_col: str, subject_col: str) -> dict[str, Any]:
    rows = []
    for subject_id in sorted(df[subject_col].dropna().unique()):
        train = df[df[subject_col] != subject_id]
        test = df[df[subject_col] == subject_id]
        fit = robust_linear_fit(train[x_col].to_numpy(), train[y_col].to_numpy())
        if fit is None:
            continue
        x_test = test[x_col].to_numpy(dtype=float)
        y_test = test[y_col].to_numpy(dtype=float)
        valid = np.isfinite(x_test) & np.isfinite(y_test)
        if int(np.sum(valid)) == 0:
            continue
        pred = fit.slope * x_test[valid] + fit.intercept
        err = pred - y_test[valid]
        rows.extend(
            {
                "subject_id": subject_id,
                "pred": float(p),
                "gt": float(g),
                "err": float(e),
            }
            for p, g, e in zip(pred, y_test[valid], err)
        )
    if not rows:
        return {}
    eval_df = pd.DataFrame(rows)
    return {
        "num_predictions": int(len(eval_df)),
        "mae": float(np.mean(np.abs(eval_df["err"]))),
        "rmse": float(np.sqrt(np.mean(np.square(eval_df["err"])))),
        "bias": float(np.mean(eval_df["err"])),
    }


def normalize_category(value: Any, mapping: dict[str, str]) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return mapping.get(text.lower(), text)


def summarize_identity_error(df: pd.DataFrame, x_col: str, y_col: str) -> dict[str, Any]:
    sub = df[[x_col, y_col]].dropna()
    if len(sub) == 0:
        return {}
    err = sub[x_col] - sub[y_col]
    return {
        "num_points": int(len(sub)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "bias": float(np.mean(err)),
    }


def extract_beat_window_features(beats_df: pd.DataFrame, t_min: float, t_max: float) -> dict[str, Any]:
    window = beats_df[(beats_df["beat_time_s"] >= t_min) & (beats_df["beat_time_s"] <= t_max)].copy()
    if len(window) == 0:
        return {}
    if "quality_ok" in window.columns:
        quality = window["quality_ok"].astype(bool)
        qwin = window[quality].copy()
        if len(qwin) >= 3:
            window = qwin
    out: dict[str, Any] = {
        "beat_window_count": int(len(window)),
        "beat_window_time_min_s": float(window["beat_time_s"].min()),
        "beat_window_time_max_s": float(window["beat_time_s"].max()),
    }
    for col in ("ratio_of_ratios", "F6_acdc", "NIR_acdc", "F6_ac", "F6_dc", "NIR_ac", "NIR_dc"):
        if col in window.columns:
            values = numeric(window[col]).dropna()
            if len(values):
                out[f"{col}_median"] = float(values.median())
    if "F6_acdc_median" in out:
        out["F6_pi_proxy_pct_median"] = float(100.0 * out["F6_acdc_median"])
    if "NIR_acdc_median" in out:
        out["NIR_pi_proxy_pct_median"] = float(100.0 * out["NIR_acdc_median"])
    return out


def collect_snapshot_master(dataset_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subject_dir in sorted(dataset_dir.glob("subject_*")):
        subject_meta_path = subject_dir / "subject_metadata.json"
        if not subject_meta_path.exists():
            continue
        subject_meta = json.loads(subject_meta_path.read_text(encoding="utf-8-sig"))
        for run_dir in sorted([p for p in subject_dir.glob("run_*") if p.is_dir()]):
            snapshot_path = run_dir / "analysis" / "red_nir_12ch" / "validation_snapshots.csv"
            beats_path = run_dir / "analysis" / "red_nir_12ch" / "red_nir_12ch_beats.csv"
            run_meta_path = run_dir / "run_metadata.json"
            if not snapshot_path.exists() or not run_meta_path.exists():
                continue

            run_meta = json.loads(run_meta_path.read_text(encoding="utf-8-sig"))
            snapshots = pd.read_csv(snapshot_path)
            beats_df = pd.read_csv(beats_path) if beats_path.exists() else pd.DataFrame()
            if "quality_ok" in beats_df.columns:
                beats_df["quality_ok"] = beats_df["quality_ok"].astype(bool)
            for _, snap in snapshots.iterrows():
                row = {
                    "subject_id": subject_dir.name,
                    "run_name": run_dir.name,
                    "protocol": run_meta.get("protocol"),
                    "duration_s": run_meta.get("duration_s"),
                    "selected_channel": run_meta.get("selected_channel"),
                    "sex_raw": subject_meta.get("sex"),
                    "dominant_hand_raw": subject_meta.get("dominant_hand"),
                    "skin_tone_raw": subject_meta.get("skin_tone"),
                    "sex_normalized": normalize_category(subject_meta.get("sex"), {"f": "F", "female": "F", "m": "M", "male": "M"}),
                    "dominant_hand_normalized": normalize_category(subject_meta.get("dominant_hand"), {"l": "left", "left": "left", "r": "right", "right": "right"}),
                    "snapshot_path": str(snapshot_path),
                }
                row.update(snap.to_dict())
                t_min = row.get("matched_live_time_min_s")
                t_max = row.get("matched_live_time_max_s")
                try:
                    t_min = float(t_min)
                    t_max = float(t_max)
                except (TypeError, ValueError):
                    t_min = t_max = None
                if t_min is not None and t_max is not None and len(beats_df):
                    row.update(extract_beat_window_features(beats_df, t_min, t_max))
                rows.append(row)
    if not rows:
        return pd.DataFrame()
    master = pd.DataFrame(rows)
    for col in master.columns:
        if col.startswith(("gt_", "live_", "matched_", "timestamp_rel_s", "duration_s")) or col.endswith(
            ("_median", "_count", "_s", "_pct")
        ):
            try:
                master[col] = pd.to_numeric(master[col])
            except Exception:
                pass
    return master


def collect_run_master(dataset_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subject_dir in sorted(dataset_dir.glob("subject_*")):
        for run_dir in sorted([p for p in subject_dir.glob("run_*") if p.is_dir()]):
            summary_path = run_dir / "analysis" / "summary.json"
            meta_path = run_dir / "run_metadata.json"
            if not summary_path.exists() or not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
            summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
            modules = summary.get("modules", {})
            spo2 = modules.get("spo2_style_estimation", {}).get("summary", {})
            live = modules.get("live_metrics", {}).get("summary", {})
            rows.append(
                {
                    "subject_id": subject_dir.name,
                    "run_name": run_dir.name,
                    "protocol": meta.get("protocol"),
                    "duration_s": meta.get("duration_s"),
                    "gt_count": modules.get("ground_truth", {}).get("summary", {}).get("num_snapshots"),
                    "selected_channel": summary.get("selected_channel"),
                    "spo2_estimated_pct": spo2.get("spo2_estimated_pct"),
                    "spo2_ratio": spo2.get("selected_ratio_of_ratios"),
                    "spo2_quality_fraction": spo2.get("quality_beat_fraction"),
                    "hr_bpm_median": live.get("hr_bpm_median"),
                    "spo2_estimated_pct_median": live.get("spo2_estimated_pct_median"),
                    "perfusion_index_pct_median": live.get("perfusion_index_pct_median"),
                    "respiratory_rate_brpm_median": live.get("respiratory_rate_brpm_median"),
                    "artifact_fraction": live.get("artifact_fraction"),
                }
            )
    return pd.DataFrame(rows)


def build_spo2_calibration(master: pd.DataFrame) -> dict[str, Any]:
    df = master[["subject_id", "gt_pulseox_spo2_pct", "ratio_of_ratios_median", "live_signal_quality_median"]].copy()
    df["gt_pulseox_spo2_pct"] = numeric(df["gt_pulseox_spo2_pct"])
    df["ratio_of_ratios_median"] = numeric(df["ratio_of_ratios_median"])
    df["live_signal_quality_median"] = numeric(df["live_signal_quality_median"])
    df = df[
        np.isfinite(df["gt_pulseox_spo2_pct"])
        & np.isfinite(df["ratio_of_ratios_median"])
        & (df["ratio_of_ratios_median"] > 0.0)
    ].copy()
    fit = robust_linear_fit(df["ratio_of_ratios_median"].to_numpy(), df["gt_pulseox_spo2_pct"].to_numpy(), scale_floor=0.35)
    if fit is None:
        return {"enabled": False, "reason": "Not enough valid SpO2 calibration points."}

    df = df.loc[fit.inlier_mask].copy()
    cv = leave_one_subject_out_metrics(df, "ratio_of_ratios_median", "gt_pulseox_spo2_pct", "subject_id")

    anchor_ratio = float(df["ratio_of_ratios_median"].median())
    slope = float(max(0.1, -fit.slope))
    anchor_pct = float(fit.intercept - slope * anchor_ratio)
    return {
        "enabled": True,
        "source_feature": "ratio_of_ratios_median",
        "model": "linear_ratio",
        "intercept_pct": float(fit.intercept),
        "coef_pct_per_ratio": float(fit.slope),
        "anchor_ratio": anchor_ratio,
        "anchor_pct": anchor_pct,
        "slope_pct_per_ratio": slope,
        "clip_min_pct": 70.0,
        "clip_max_pct": 100.0,
        "train_stats": {
            "num_points": fit.total_count,
            "num_inliers": fit.inlier_count,
            "mae": fit.train_mae,
            "rmse": fit.train_rmse,
            "bias": fit.train_bias,
            "corr": fit.train_corr,
        },
        "leave_one_subject_out": cv,
        "formula": f"SpO2_est = clip({fit.intercept:.6f} + ({fit.slope:.6f}) * R, 70.0, 100.0)",
    }


def build_pi_calibration(master: pd.DataFrame) -> dict[str, Any]:
    candidates = []
    for feature in ("NIR_pi_proxy_pct_median", "F6_pi_proxy_pct_median"):
        if feature not in master.columns:
            continue
        df = master[["subject_id", "gt_pulseox_pi_pct", feature]].copy()
        df["gt_pulseox_pi_pct"] = numeric(df["gt_pulseox_pi_pct"])
        df[feature] = numeric(df[feature])
        df = df[
            np.isfinite(df["gt_pulseox_pi_pct"])
            & np.isfinite(df[feature])
            & (df[feature] > 0.0)
        ].copy()
        identity = summarize_identity_error(df, feature, "gt_pulseox_pi_pct")
        fit = robust_linear_fit(df[feature].to_numpy(), df["gt_pulseox_pi_pct"].to_numpy(), scale_floor=0.75)
        if fit is None:
            continue
        inlier_df = df.loc[fit.inlier_mask].copy()
        cv = leave_one_subject_out_metrics(inlier_df, feature, "gt_pulseox_pi_pct", "subject_id")
        candidates.append((feature, fit, cv, identity))

    if not candidates:
        return {"enabled": False, "reason": "Not enough valid PI calibration points."}

    def score(item: tuple[str, LinearFit, dict[str, Any], dict[str, Any]]) -> tuple[float, float]:
        _, fit, cv, _ = item
        cv_mae = cv.get("mae", np.inf)
        return (cv_mae if np.isfinite(cv_mae) else np.inf, fit.train_mae)

    feature, fit, cv, identity = sorted(candidates, key=score)[0]
    source_channel = "NIR" if feature.startswith("NIR") else "F6"
    enabled = bool(
        fit.slope > 0.0
        and cv.get("mae") is not None
        and identity.get("mae") is not None
        and np.isfinite(cv.get("mae"))
        and np.isfinite(identity.get("mae"))
        and cv["mae"] <= 0.95 * identity["mae"]
        and fit.train_corr is not None
        and fit.train_corr >= 0.25
    )
    return {
        "enabled": enabled,
        "source_channel": source_channel,
        "source_feature": feature,
        "model": "linear_proxy",
        "intercept_pct": float(fit.intercept),
        "coef_pct_per_proxy_pct": float(fit.slope),
        "clip_min_pct": 0.0,
        "clip_max_pct": 20.0,
        "train_stats": {
            "num_points": fit.total_count,
            "num_inliers": fit.inlier_count,
            "mae": fit.train_mae,
            "rmse": fit.train_rmse,
            "bias": fit.train_bias,
            "corr": fit.train_corr,
        },
        "leave_one_subject_out": cv,
        "identity_baseline": identity,
        "formula": f"PI_est = clip({fit.intercept:.6f} + ({fit.slope:.6f}) * PI_proxy, 0.0, 20.0)",
        "apply_note": (
            "Enabled only when the cohort relation is positive and held-out performance improves over the raw proxy."
            if enabled
            else "Kept as evaluation-only. Future runs still benefit from using a consistent IR-based PI proxy, but absolute PI calibration is not stable enough yet."
        ),
    }


def build_rr_calibration(master: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for protocol in ("controlled_breathing", "resting", "post_exercise_recovery", "post_cold_recovery"):
        sub = master[master["protocol"] == protocol].copy()
        if len(sub) == 0:
            continue
        candidates = []
        for feature in ("live_respiratory_rate_brpm_median", "postrun_respiration_final_brpm"):
            if feature not in sub.columns:
                continue
            work = sub[["subject_id", "gt_rr_ref_bpm", feature]].copy()
            work["gt_rr_ref_bpm"] = numeric(work["gt_rr_ref_bpm"])
            work[feature] = numeric(work[feature])
            work = work[np.isfinite(work["gt_rr_ref_bpm"]) & np.isfinite(work[feature]) & (work[feature] > 0.0)].copy()
            if len(work) < 12:
                continue
            fit = robust_linear_fit(work[feature].to_numpy(), work["gt_rr_ref_bpm"].to_numpy(), scale_floor=1.0)
            if fit is None:
                continue
            identity = summarize_identity_error(work, feature, "gt_rr_ref_bpm")
            inlier_df = work.loc[fit.inlier_mask].copy()
            cv = leave_one_subject_out_metrics(inlier_df, feature, "gt_rr_ref_bpm", "subject_id")
            candidates.append((feature, fit, cv, identity))

        if not candidates:
            out[protocol] = {"enabled": False, "reason": "Not enough RR reference points."}
            continue

        feature, fit, cv, identity = sorted(
            candidates,
            key=lambda item: (
                item[2].get("mae", np.inf) if item[2] else np.inf,
                item[1].train_mae,
            ),
        )[0]
        cv_mae = cv.get("mae")
        identity_mae = identity.get("mae")
        enable = bool(
            cv_mae is not None
            and identity_mae is not None
            and np.isfinite(cv_mae)
            and np.isfinite(identity_mae)
            and cv_mae <= 0.95 * identity_mae
            and fit.train_corr is not None
            and fit.train_corr >= 0.35
        )
        out[protocol] = {
            "enabled": enable,
            "source_feature": feature,
            "model": "linear_brpm",
            "intercept_brpm": float(fit.intercept),
            "coef_brpm_per_brpm": float(fit.slope),
            "clip_min_brpm": 4.0,
            "clip_max_brpm": 45.0,
            "train_stats": {
                "num_points": fit.total_count,
                "num_inliers": fit.inlier_count,
                "mae": fit.train_mae,
                "rmse": fit.train_rmse,
                "bias": fit.train_bias,
                "corr": fit.train_corr,
            },
            "leave_one_subject_out": cv,
            "identity_baseline": identity,
            "apply_note": (
                "Enabled only when the subject-held-out MAE improves over the identity mapping and the cohort relation is stable."
                if enable
                else "Kept as evaluation-only; current cohort relation is too weak or too protocol-dependent for automatic correction."
            ),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit cohort calibration for red/NIR 12ch runs and save a reusable calibration JSON.")
    parser.add_argument("--dataset-dir", default=str(ROOT / "dataset"))
    parser.add_argument("--output-dir", default=None, help="Defaults to dataset/calibration")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir) if args.output_dir else default_red_nir_12ch_calibration_path(dataset_dir).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_master = collect_snapshot_master(dataset_dir)
    if len(snapshot_master) == 0:
        raise SystemExit("No red_nir_12ch validation snapshots found in dataset.")
    run_master = collect_run_master(dataset_dir)

    snapshot_master_path = output_dir / "red_nir_12ch_snapshot_master.csv"
    run_master_path = output_dir / "red_nir_12ch_run_master.csv"
    snapshot_master.to_csv(snapshot_master_path, index=False)
    run_master.to_csv(run_master_path, index=False)

    calibration = {
        "schema_version": 1,
        "calibration_name": "red_nir_12ch_cohort_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": str(dataset_dir),
        "source_files": {
            "snapshot_master_csv": str(snapshot_master_path),
            "run_master_csv": str(run_master_path),
        },
        "cohort_stats": {
            "num_subjects": int(snapshot_master["subject_id"].nunique()),
            "num_runs": int(snapshot_master["run_name"].nunique()),
            "num_snapshot_rows": int(len(snapshot_master)),
            "protocol_counts": snapshot_master["protocol"].value_counts(dropna=False).to_dict(),
        },
        "spo2": build_spo2_calibration(snapshot_master),
        "perfusion_index": build_pi_calibration(snapshot_master),
        "respiration": build_rr_calibration(snapshot_master),
        "notes": [
            "SpO2 is calibrated on beat-window ratio-of-ratios vs pulseox SpO2 snapshots.",
            "PI is calibrated on a consistent beat-window proxy channel, not on the GUI-selected display channel.",
            "RR correction is only enabled when held-out cohort performance improves over the identity mapping.",
        ],
    }

    out_path = output_dir / "red_nir_12ch_calibration.json"
    save_json(out_path, calibration)
    print(f"Saved: {snapshot_master_path}")
    print(f"Saved: {run_master_path}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
