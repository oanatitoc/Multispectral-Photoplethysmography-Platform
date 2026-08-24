from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def rmse(values: pd.Series) -> float:
    x = values.dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return np.nan
    return float(np.sqrt(np.mean(x ** 2)))


def metric_stats(series: pd.Series, absolute: bool = False) -> dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return {
            "n": 0,
            "mae": np.nan,
            "median_abs": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
        }
    work = s.abs() if absolute else s
    return {
        "n": int(len(s)),
        "mae": float(s.abs().mean()),
        "median_abs": float(s.abs().median()),
        "rmse": rmse(s),
        "bias": float(s.mean()),
    }


def protocol_report_rows(snapshot_df: pd.DataFrame, run_df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    protocols = ["overall"] + sorted(str(p) for p in snapshot_df["protocol"].dropna().unique())
    for protocol in protocols:
        if protocol == "overall":
            snap = snapshot_df.copy()
            runs = run_df.copy()
        else:
            snap = snapshot_df[snapshot_df["protocol"] == protocol].copy()
            runs = run_df[run_df["protocol"] == protocol].copy()

        hr = metric_stats(snap["hr_error_vs_pulseox_bpm"])
        spo2 = metric_stats(snap["spo2_error_vs_pulseox_pct"])
        pi = metric_stats(snap["pi_proxy_minus_pulseox_pct_points"])
        rr_live = metric_stats(snap["rr_error_vs_reference_brpm"])
        rr_post = metric_stats(snap["postrun_rr_error_vs_reference_brpm"])

        rows.append(
            {
                "protocol": protocol,
                "num_subjects": int(snap["subject_id"].nunique()) if len(snap) else 0,
                "num_runs": int(runs["run_key"].nunique()) if len(runs) else 0,
                "num_snapshots": int(len(snap)),
                "mean_duration_s": float(runs["duration_s"].mean()) if len(runs) else np.nan,
                "median_duration_s": float(runs["duration_s"].median()) if len(runs) else np.nan,
                "artifact_fraction_mean": float(runs["artifact_fraction"].mean()) if len(runs) else np.nan,
                "signal_quality_mean": float(snap["live_signal_quality_median"].mean()) if len(snap) else np.nan,
                "hr_mae_bpm": hr["mae"],
                "hr_rmse_bpm": hr["rmse"],
                "hr_bias_bpm": hr["bias"],
                "hr_median_abs_bpm": hr["median_abs"],
                "hr_n": hr["n"],
                "spo2_mae_pct": spo2["mae"],
                "spo2_rmse_pct": spo2["rmse"],
                "spo2_bias_pct": spo2["bias"],
                "spo2_median_abs_pct": spo2["median_abs"],
                "spo2_n": spo2["n"],
                "pi_proxy_mae_pct_points": pi["mae"],
                "pi_proxy_rmse_pct_points": pi["rmse"],
                "pi_proxy_bias_pct_points": pi["bias"],
                "pi_proxy_median_abs_pct_points": pi["median_abs"],
                "pi_proxy_n": pi["n"],
                "rr_live_mae_brpm": rr_live["mae"],
                "rr_live_rmse_brpm": rr_live["rmse"],
                "rr_live_bias_brpm": rr_live["bias"],
                "rr_live_median_abs_brpm": rr_live["median_abs"],
                "rr_live_n": rr_live["n"],
                "rr_postrun_mae_brpm": rr_post["mae"],
                "rr_postrun_rmse_brpm": rr_post["rmse"],
                "rr_postrun_bias_brpm": rr_post["bias"],
                "rr_postrun_median_abs_brpm": rr_post["median_abs"],
                "rr_postrun_n": rr_post["n"],
            }
        )
    return rows


def subject_report_rows(snapshot_df: pd.DataFrame, run_df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for subject_id in sorted(str(s) for s in snapshot_df["subject_id"].dropna().unique()):
        snap = snapshot_df[snapshot_df["subject_id"] == subject_id].copy()
        runs = run_df[run_df["subject_id"] == subject_id].copy()
        hr = metric_stats(snap["hr_error_vs_pulseox_bpm"])
        spo2 = metric_stats(snap["spo2_error_vs_pulseox_pct"])
        pi = metric_stats(snap["pi_proxy_minus_pulseox_pct_points"])
        rr_live = metric_stats(snap["rr_error_vs_reference_brpm"])
        rr_post = metric_stats(snap["postrun_rr_error_vs_reference_brpm"])
        rows.append(
            {
                "subject_id": subject_id,
                "num_protocols": int(snap["protocol"].nunique()),
                "num_runs": int(runs["run_key"].nunique()),
                "num_snapshots": int(len(snap)),
                "mean_duration_s": float(runs["duration_s"].mean()) if len(runs) else np.nan,
                "artifact_fraction_mean": float(runs["artifact_fraction"].mean()) if len(runs) else np.nan,
                "hr_mae_bpm": hr["mae"],
                "spo2_mae_pct": spo2["mae"],
                "pi_proxy_mae_pct_points": pi["mae"],
                "rr_live_mae_brpm": rr_live["mae"],
                "rr_postrun_mae_brpm": rr_post["mae"],
            }
        )
    return rows


def readiness_lines(protocol_report: pd.DataFrame, calibration: dict) -> list[str]:
    overall = protocol_report[protocol_report["protocol"] == "overall"]
    controlled = protocol_report[protocol_report["protocol"] == "controlled_breathing"]
    resting = protocol_report[protocol_report["protocol"] == "resting"]
    exercise = protocol_report[protocol_report["protocol"] == "post_exercise_recovery"]
    cold = protocol_report[protocol_report["protocol"] == "post_cold_recovery"]

    lines: list[str] = []
    if not overall.empty:
        row = overall.iloc[0]
        lines.append(
            f"- HR is usable now: overall snapshot MAE is {row['hr_mae_bpm']:.2f} bpm across {int(row['hr_n'])} matched snapshots."
        )
        lines.append(
            f"- SpO2-style is the strongest calibrated metric: overall MAE is {row['spo2_mae_pct']:.2f} percentage points across {int(row['spo2_n'])} snapshots."
        )
        lines.append(
            f"- PI is still trend-oriented rather than absolutely calibrated: overall absolute difference vs pulseox PI is {row['pi_proxy_mae_pct_points']:.2f} points."
        )

    if not controlled.empty:
        row = controlled.iloc[0]
        lines.append(
            f"- Controlled-breathing RR is the best respiration protocol: live RR MAE is {row['rr_live_mae_brpm']:.2f} brpm and post-run RR MAE is {row['rr_postrun_mae_brpm']:.2f} brpm."
        )
    if not resting.empty:
        row = resting.iloc[0]
        lines.append(
            f"- Resting RR remains weak for calibration: live RR MAE is {row['rr_live_mae_brpm']:.2f} brpm and post-run RR MAE is {row['rr_postrun_mae_brpm']:.2f} brpm."
        )
    if not exercise.empty:
        row = exercise.iloc[0]
        lines.append(
            f"- Post-exercise RR tracks recovery directionally, but snapshot error is still high: live RR MAE is {row['rr_live_mae_brpm']:.2f} brpm."
        )
    if not cold.empty:
        row = cold.iloc[0]
        lines.append(
            f"- Cold-exposure RR is not a strong validation protocol: live RR MAE is {row['rr_live_mae_brpm']:.2f} brpm and should be treated as exploratory."
        )

    spo2_cal = calibration.get("spo2", {})
    if spo2_cal.get("enabled"):
        loso = spo2_cal.get("leave_one_subject_out", {})
        lines.append(
            f"- Cohort SpO2 calibration is enabled. Leave-one-subject-out MAE is {float(loso.get('mae', np.nan)):.2f}% with formula `{spo2_cal.get('formula')}`."
        )
    rr_cal = calibration.get("respiration", {}).get("controlled_breathing", {})
    if rr_cal.get("enabled"):
        loso = rr_cal.get("leave_one_subject_out", {})
        lines.append(
            f"- Controlled-breathing RR calibration is enabled. Leave-one-subject-out MAE is {float(loso.get('mae', np.nan)):.2f} brpm."
        )
    pi_cal = calibration.get("perfusion_index", {})
    if not pi_cal.get("enabled", False):
        lines.append("- PI absolute calibration is still disabled; the current PI output should be interpreted as a consistent proxy/trend, not a calibrated absolute PI.")
    return lines


def next_steps_lines(protocol_report: pd.DataFrame) -> list[str]:
    lines = [
        "- Keep collecting `controlled_breathing` runs, because this is the cleanest protocol for RR calibration and algorithm validation.",
        "- Keep collecting `resting` runs for HR, SpO2-style stability, PI trend, tissue trend, and morphology/stiffness exploration.",
        "- Keep collecting `post_exercise_recovery` runs for dynamic HR recovery, RR recovery, PI trend, and tissue oxygenation trend.",
        "- Keep collecting `post_cold_recovery` runs mainly for perfusion and tissue/stiffness response, not as the main RR protocol.",
        "- Standardize the ground-truth workflow as much as possible: same snapshot timings, same finger placement, same delay after exercise/cold stimulus.",
    ]

    overall = protocol_report[protocol_report["protocol"] == "overall"]
    if not overall.empty:
        num_subjects = int(overall.iloc[0]["num_subjects"])
        if num_subjects < 20:
            lines.append(
                f"- The next meaningful cohort target is about 20 subjects. You already have {num_subjects}; moving to 20 will make cross-subject calibration much more credible."
            )
            lines.append(
                "- After ~20 subjects, rerun calibration and re-check whether PI or resting RR become stable enough for automatic correction."
            )
        else:
            lines.append("- With 20+ subjects, the next step becomes protocol balancing and feature-specific validation, not just raw cohort growth.")
    lines.append(
        "- Do not start hemoglobin ML yet unless you can attach real laboratory Hb labels; otherwise it remains future work."
    )
    lines.append(
        "- Continue saving skin tone on the Monk scale so later you can test whether error differs by skin tone group."
    )
    return lines


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._"
    cols = list(df.columns)
    widths = []
    for col in cols:
        cell_values = ["" if pd.isna(v) else str(v) for v in df[col].tolist()]
        widths.append(max(len(str(col)), *(len(v) for v in cell_values)))

    def fmt_row(values: list[str]) -> str:
        return "| " + " | ".join(v.ljust(w) for v, w in zip(values, widths)) + " |"

    header = fmt_row([str(c) for c in cols])
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    body = [
        fmt_row(["" if pd.isna(v) else str(v) for v in row])
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep, *body])


def build_markdown(
    out_path: Path,
    calibration: dict,
    protocol_report: pd.DataFrame,
    subject_report: pd.DataFrame,
) -> None:
    cohort = calibration.get("cohort_stats", {})
    lines = [
        "# Multispectral PPG Cohort Report",
        "",
        "## Cohort",
        f"- Subjects used in calibration: {cohort.get('num_subjects')}",
        f"- Runs used in calibration: {cohort.get('num_runs')}",
        f"- Ground-truth snapshots used in calibration: {cohort.get('num_snapshot_rows')}",
        f"- Protocol counts: {json.dumps(cohort.get('protocol_counts', {}), ensure_ascii=False)}",
        "",
        "## Key Findings",
    ]
    lines.extend(readiness_lines(protocol_report, calibration))
    lines.extend([
        "",
        "## Protocol Table",
        "",
        markdown_table(protocol_report),
        "",
        "## Subject Table",
        "",
        markdown_table(subject_report),
        "",
        "## Recommended Next Steps",
    ])
    lines.extend(next_steps_lines(protocol_report))
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cohort-level validation and calibration report for red/NIR 12ch data.")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    calibration_dir = dataset_dir / "calibration"
    out_dir = Path(args.out_dir) if args.out_dir else (dataset_dir / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = calibration_dir / "red_nir_12ch_snapshot_master.csv"
    run_path = calibration_dir / "red_nir_12ch_run_master.csv"
    calibration_path = calibration_dir / "red_nir_12ch_calibration.json"

    snapshot_df = pd.read_csv(snapshot_path)
    run_df = pd.read_csv(run_path)
    calibration = load_json(calibration_path)

    snapshot_df = numeric(
        snapshot_df,
        [
            "duration_s",
            "gt_pulseox_hr_bpm",
            "gt_pulseox_spo2_pct",
            "gt_pulseox_pi_pct",
            "gt_rr_ref_bpm",
            "live_hr_bpm_median",
            "live_hr_fft_bpm_median",
            "live_respiratory_rate_brpm_median",
            "live_spo2_estimated_pct_median",
            "live_perfusion_index_pct_median",
            "live_signal_quality_median",
            "live_artifact_flag_median",
            "hr_error_vs_pulseox_bpm",
            "spo2_error_vs_pulseox_pct",
            "pi_proxy_minus_pulseox_pct_points",
            "rr_error_vs_reference_brpm",
            "postrun_respiration_final_brpm",
            "postrun_rr_error_vs_reference_brpm",
        ],
    )
    run_df = numeric(
        run_df,
        [
            "duration_s",
            "gt_count",
            "spo2_estimated_pct",
            "spo2_ratio",
            "spo2_quality_fraction",
            "hr_bpm_median",
            "spo2_estimated_pct_median",
            "perfusion_index_pct_median",
            "respiratory_rate_brpm_median",
            "artifact_fraction",
        ],
    )
    snapshot_df["run_key"] = snapshot_df["subject_id"].astype(str) + "/" + snapshot_df["run_name"].astype(str)
    run_df["run_key"] = run_df["subject_id"].astype(str) + "/" + run_df["run_name"].astype(str)

    protocol_report = pd.DataFrame(protocol_report_rows(snapshot_df, run_df))
    subject_report = pd.DataFrame(subject_report_rows(snapshot_df, run_df))

    protocol_csv = out_dir / "red_nir_12ch_protocol_report.csv"
    subject_csv = out_dir / "red_nir_12ch_subject_report.csv"
    snapshot_csv = out_dir / "red_nir_12ch_snapshot_report.csv"
    run_csv = out_dir / "red_nir_12ch_run_report.csv"
    md_path = out_dir / "red_nir_12ch_cohort_report.md"
    txt_path = out_dir / "red_nir_12ch_cohort_report.txt"

    protocol_report.to_csv(protocol_csv, index=False)
    subject_report.to_csv(subject_csv, index=False)
    snapshot_df.to_csv(snapshot_csv, index=False)
    run_df.to_csv(run_csv, index=False)
    build_markdown(md_path, calibration, protocol_report, subject_report)
    txt_path.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"Saved: {protocol_csv}")
    print(f"Saved: {subject_csv}")
    print(f"Saved: {snapshot_csv}")
    print(f"Saved: {run_csv}")
    print(f"Saved: {md_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
