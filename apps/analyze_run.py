from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppg_suite.config import AnalysisConfig
from ppg_suite.io_utils import (
    find_subject_metadata,
    read_signal_csv,
    resolve_run_dir_from_input,
    save_json,
    write_df,
)
from ppg_suite.perfusion_support import pick_best_perfusion_channel
from ppg_suite.modules import (
    run_hemoglobin_estimation,
    placeholder_modules,
    run_hrv,
    run_perfusion,
    run_perfusion_response,
    run_respiration,
    run_spo2_style_estimation,
    run_stiffness,
    run_tissue_oxygenation_trend,
    run_vasomotion,
)


AUTO_FZ_NIR = "auto_fz_nir"
AUTO_CHANNEL_CANDIDATES = ("FZ_diff", "NIR_diff")


def parse_segments(text: str):
    parts = []
    for item in text.split(","):
        name, t0, t1 = item.split(":")
        parts.append((name, float(t0), float(t1)))
    return parts


def load_analysis_input(input_path: str, requested_channel: str, cfg: AnalysisConfig):
    if requested_channel != AUTO_FZ_NIR:
        loaded = read_signal_csv(input_path, requested_channel, cfg.drop_start_sec, cfg.drop_end_sec)
        return loaded, requested_channel, {
            "mode": "manual",
            "selected_channel": requested_channel,
        }

    candidate_loads = {}
    candidate_errors = {}
    for channel_name in AUTO_CHANNEL_CANDIDATES:
        try:
            candidate_loads[channel_name] = read_signal_csv(input_path, channel_name, cfg.drop_start_sec, cfg.drop_end_sec)
        except Exception as exc:
            candidate_errors[channel_name] = str(exc)

    if not candidate_loads:
        raise RuntimeError(f"No candidate channels available for {AUTO_FZ_NIR}: {candidate_errors}")

    selection = pick_best_perfusion_channel(candidate_loads, cfg, preferred_channel="NIR_diff")
    selection["mode"] = AUTO_FZ_NIR
    selection["candidate_errors"] = candidate_errors
    selected_channel = selection["selected_channel"]
    return candidate_loads[selected_channel], selected_channel, selection


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze one PPG CSV run and save structured outputs.")
    parser.add_argument("--input", required=True, help="Path to raw CSV.")
    parser.add_argument("--output-dir", default=None, help="Optional output folder.")
    parser.add_argument("--channel", default=AUTO_FZ_NIR)
    parser.add_argument("--drop-start-sec", type=float, default=3.0)
    parser.add_argument("--drop-end-sec", type=float, default=3.0)
    parser.add_argument("--segments", default="baseline:0:60,task:60:120,recovery:120:180")
    parser.add_argument("--height-cm", type=float, default=None, help="Optional subject height for stiffness index.")
    parser.add_argument("--spo2-anchor-ratio", type=float, default=None, help="Optional custom anchor ratio R for heuristic SpO2 calibration.")
    parser.add_argument("--spo2-anchor-pct", type=float, default=None, help="Optional custom anchor SpO2 percent for heuristic calibration.")
    parser.add_argument("--spo2-slope", type=float, default=25.0, help="Heuristic linear slope in percent per unit ratio.")
    parser.add_argument("--spo2-clip-min", type=float, default=70.0, help="Lower clip bound for heuristic SpO2 percent.")
    parser.add_argument("--spo2-clip-max", type=float, default=100.0, help="Upper clip bound for heuristic SpO2 percent.")
    args = parser.parse_args()

    requested_channel = args.channel
    cfg = AnalysisConfig(
        channel=requested_channel,
        drop_start_sec=args.drop_start_sec,
        drop_end_sec=args.drop_end_sec,
        segments=parse_segments(args.segments),
        spo2_anchor_ratio=args.spo2_anchor_ratio,
        spo2_anchor_pct=args.spo2_anchor_pct,
        spo2_linear_slope=args.spo2_slope,
        spo2_clip_min_pct=args.spo2_clip_min,
        spo2_clip_max_pct=args.spo2_clip_max,
    )

    loaded, selected_channel, channel_selection = load_analysis_input(args.input, requested_channel, cfg)
    cfg.channel = selected_channel
    run_dir, analysis_dir = resolve_run_dir_from_input(args.input, args.output_dir, requested_channel)

    subject_meta = find_subject_metadata(run_dir)
    height_cm = args.height_cm if args.height_cm is not None else subject_meta.get("height_cm")
    if height_cm is not None:
        cfg.subject_height_m = float(height_cm) / 100.0

    modules = {
        "hrv": run_hrv,
        "respiration": run_respiration,
        "perfusion": run_perfusion,
        "perfusion_response": run_perfusion_response,
        "vasomotion": run_vasomotion,
        "stiffness": run_stiffness,
        "tissue_oxygenation_trend": None,
        "hemoglobin_estimation": None,
        "spo2_style_estimation": None,
    }

    master_summary = {
        "input_csv": str(Path(args.input).resolve()),
        "channel": requested_channel,
        "selected_channel": selected_channel,
        "channel_selection": channel_selection,
        "run_dir": str(run_dir.resolve()),
        "subject_metadata": subject_meta,
        "modules": {},
        "future_modules": placeholder_modules(),
    }

    for name, fn in modules.items():
        try:
            if name == "spo2_style_estimation":
                result = run_spo2_style_estimation(loaded, cfg)
            elif name == "tissue_oxygenation_trend":
                result = run_tissue_oxygenation_trend(loaded, cfg)
            elif name == "hemoglobin_estimation":
                result = run_hemoglobin_estimation(loaded, cfg)
            else:
                result = fn(loaded["t_s"], loaded["raw"], cfg)
            summary = result["summary"]
            save_json(analysis_dir / f"{name}_summary.json", summary)
            for table_name, df in result.get("tables", {}).items():
                write_df(analysis_dir / f"{table_name}.csv", df)
            master_summary["modules"][name] = {
                "status": "ok",
                "summary_file": f"{name}_summary.json",
                "summary": summary,
            }
            print(f"[OK] {name}")
        except Exception as exc:
            master_summary["modules"][name] = {
                "status": "error",
                "error": str(exc),
            }
            print(f"[ERROR] {name}: {exc}")

    save_json(analysis_dir / "summary.json", master_summary)
    print(f"\nSaved analysis to: {analysis_dir}")


if __name__ == "__main__":
    main()
