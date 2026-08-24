from __future__ import annotations

import argparse
import csv
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import serial
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import messagebox, simpledialog, ttk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppg_suite.calibration import (
    calibration_display_name,
    default_red_nir_12ch_calibration_path,
    load_red_nir_12ch_calibration,
    resolve_spo2_params,
)
from ppg_suite.io_utils import create_run_dir, create_subject, load_json, save_json
from ppg_suite.live_metrics import (
    LiveMetrics,
    compute_live_metrics,
    preferred_preview_channel,
    signal_columns,
)


APP_VERSION = "0.1.0"
METRIC_FIELDS = list(LiveMetrics().to_row().keys())
SEX_OPTIONS = ["F", "M"]
DOMINANT_HAND_OPTIONS = ["left", "right"]
SKIN_TONE_OPTIONS = [
    "Monk 01",
    "Monk 02",
    "Monk 03",
    "Monk 04",
    "Monk 05",
    "Monk 06",
    "Monk 07",
    "Monk 08",
    "Monk 09",
    "Monk 10",
]


def open_serial_reset(port: str, baud: int) -> serial.Serial:
    ser = serial.Serial(port, baud, timeout=1)
    ser.setDTR(False)
    time.sleep(0.1)
    ser.setDTR(True)
    time.sleep(1.0)
    return ser


def read_header(ser: serial.Serial, timeout_s: float = 8.0) -> list[str]:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        line = ser.readline().decode(errors="ignore").strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("ms,"):
            return [part.strip() for part in line.split(",")]
    raise RuntimeError("No CSV header received from firmware.")


def fmt(value: object, digits: int = 1, suffix: str = "") -> str:
    if value is None:
        return "--"
    try:
        v = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(v):
        return "--"
    return f"{v:.{digits}f}{suffix}"


def next_subject_id(dataset_dir: Path) -> str:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    max_idx = 0
    for path in dataset_dir.glob("subject_*"):
        if not path.is_dir():
            continue
        try:
            max_idx = max(max_idx, int(path.name.split("_")[-1]))
        except ValueError:
            continue
    return f"subject_{max_idx + 1:04d}"


class SubjectDialog(tk.Toplevel):
    def __init__(self, parent: tk.Tk, dataset_dir: Path) -> None:
        super().__init__(parent)
        self.title("Select / Add Subject")
        self.configure(bg="#F8FAFC")
        self.resizable(False, False)
        self.dataset_dir = dataset_dir
        self.result: Optional[tuple[str, dict]] = None

        self.vars = {
            "subject_id": tk.StringVar(value=next_subject_id(dataset_dir)),
            "name": tk.StringVar(),
            "age": tk.StringVar(),
            "sex": tk.StringVar(),
            "height_cm": tk.StringVar(),
            "weight_kg": tk.StringVar(),
            "skin_tone": tk.StringVar(),
            "dominant_hand": tk.StringVar(),
            "measurement_site": tk.StringVar(value="finger"),
            "notes": tk.StringVar(),
        }

        existing = sorted(path.name for path in dataset_dir.glob("subject_*") if path.is_dir())
        top = ttk.Frame(self, padding=16)
        top.grid(row=0, column=0, sticky="nsew")
        self.field_widgets: dict[str, ttk.Widget] = {}

        ttk.Label(top, text="Existing subject").grid(row=0, column=0, sticky="w")
        self.existing = ttk.Combobox(top, values=existing, width=24, state="readonly")
        self.existing.grid(row=0, column=1, sticky="ew", padx=(8, 6))
        ttk.Button(top, text="Load", command=self._load_selected).grid(row=0, column=2, sticky="ew")

        fields = [
            ("Subject ID", "subject_id"),
            ("Name / code", "name"),
            ("Age", "age"),
            ("Sex", "sex"),
            ("Height cm", "height_cm"),
            ("Weight kg", "weight_kg"),
            ("Skin tone", "skin_tone"),
            ("Dominant hand", "dominant_hand"),
            ("Default site", "measurement_site"),
            ("Notes", "notes"),
        ]
        for row, (label, key) in enumerate(fields, start=1):
            ttk.Label(top, text=label).grid(row=row, column=0, sticky="w", pady=5)
            widget = self._build_field_widget(top, key)
            widget.grid(row=row, column=1, columnspan=2, sticky="ew", pady=5)
            self.field_widgets[key] = widget

        ttk.Label(
            top,
            text="Skin tone uses the Monk Skin Tone scale for new subjects.",
        ).grid(row=len(fields) + 1, column=0, columnspan=3, sticky="w", pady=(2, 0))

        buttons = ttk.Frame(top)
        buttons.grid(row=len(fields) + 2, column=0, columnspan=3, sticky="ew", pady=(14, 0))
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side="right")
        ttk.Button(buttons, text="Save / Select", command=self._save).pack(side="right", padx=(0, 8))

        self.transient(parent)
        self.grab_set()
        self.wait_visibility()
        self.focus()

    def _build_field_widget(self, parent: ttk.Frame, key: str) -> ttk.Widget:
        if key == "sex":
            return ttk.Combobox(parent, textvariable=self.vars[key], values=SEX_OPTIONS, width=32, state="readonly")
        if key == "skin_tone":
            return ttk.Combobox(parent, textvariable=self.vars[key], values=SKIN_TONE_OPTIONS, width=32, state="readonly")
        if key == "dominant_hand":
            return ttk.Combobox(parent, textvariable=self.vars[key], values=DOMINANT_HAND_OPTIONS, width=32, state="readonly")
        return ttk.Entry(parent, textvariable=self.vars[key], width=34)

    def _load_combo_value(self, key: str, base_options: list[str], value: object) -> None:
        widget = self.field_widgets.get(key)
        text = "" if value is None else str(value).strip()
        if not isinstance(widget, ttk.Combobox):
            self.vars[key].set(text)
            return
        values = list(base_options)
        if text and text not in values:
            values = [text] + values
        widget.configure(values=values)
        self.vars[key].set(text)

    def _load_selected(self) -> None:
        subject_id = self.existing.get().strip()
        if not subject_id:
            return
        meta_path = self.dataset_dir / subject_id / "subject_metadata.json"
        metadata = load_json(meta_path) if meta_path.exists() else {"subject_id": subject_id}
        for key, var in self.vars.items():
            value = metadata.get(key, "")
            if key == "sex":
                self._load_combo_value(key, SEX_OPTIONS, value)
            elif key == "skin_tone":
                self._load_combo_value(key, SKIN_TONE_OPTIONS, value)
            elif key == "dominant_hand":
                self._load_combo_value(key, DOMINANT_HAND_OPTIONS, value)
            else:
                var.set("" if value is None else str(value))
        self.vars["subject_id"].set(subject_id)

    def _parse_float(self, key: str) -> Optional[float]:
        text = self.vars[key].get().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"{key} must be numeric.") from exc

    def _save(self) -> None:
        try:
            subject_id = self.vars["subject_id"].get().strip()
            if not subject_id:
                raise ValueError("Subject ID is required.")

            height_cm = self._parse_float("height_cm")
            weight_kg = self._parse_float("weight_kg")
            age = self._parse_float("age")
            bmi = None
            if height_cm and weight_kg and height_cm > 0:
                bmi = weight_kg / ((height_cm / 100.0) ** 2)

            metadata = {
                "subject_id": subject_id,
                "name": self.vars["name"].get().strip(),
                "age": age,
                "sex": self.vars["sex"].get().strip(),
                "height_cm": height_cm,
                "weight_kg": weight_kg,
                "bmi": bmi,
                "skin_tone": self.vars["skin_tone"].get().strip(),
                "dominant_hand": self.vars["dominant_hand"].get().strip(),
                "default_measurement_site": self.vars["measurement_site"].get().strip(),
                "notes": self.vars["notes"].get().strip(),
            }
            metadata = {k: v for k, v in metadata.items() if v not in ("", None)}
            create_subject(self.dataset_dir, subject_id, metadata)
            self.result = (subject_id, metadata)
            self.destroy()
        except Exception as exc:
            messagebox.showerror("Subject metadata", str(exc), parent=self)


class PPGDashboard:
    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.args = args
        self.dataset_dir = Path(args.dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_path = Path(args.calibration_file) if args.calibration_file else default_red_nir_12ch_calibration_path(self.dataset_dir)
        self.calibration = load_red_nir_12ch_calibration(self.calibration_path)
        (
            self.default_spo2_anchor_ratio,
            self.default_spo2_anchor_pct,
            self.default_spo2_slope,
            self.default_spo2_clip_min,
            self.default_spo2_clip_max,
        ) = resolve_spo2_params(
            self.calibration,
            anchor_ratio=args.spo2_anchor_ratio,
            anchor_pct=args.spo2_anchor_pct,
            slope=args.spo2_slope,
        )

        self.ser: Optional[serial.Serial] = None
        self.reader_thread: Optional[threading.Thread] = None
        self.reader_running = False
        self.sample_queue: queue.Queue[tuple[float, list[str]]] = queue.Queue(maxsize=5000)

        self.header: list[str] = []
        self.columns: list[str] = []
        self.column_indices: dict[str, int] = {}
        self.us_idx: Optional[int] = None
        self.ms_idx: Optional[int] = None
        self.t0_sensor: Optional[float] = None
        self.tbuf: deque[float] = deque(maxlen=args.max_samples)
        self.series: dict[str, deque[float]] = {}

        self.subject_id: Optional[str] = None
        self.subject_meta: dict = {}
        self.run_dir: Optional[Path] = None
        self.recording = False
        self.recording_started_pc: Optional[float] = None
        self.rows_written = 0
        self.last_metrics_write_pc = 0.0
        self.latest_metrics = LiveMetrics()
        self.stable_pulse_polarity: dict[str, int] = {}
        self.run_meta: dict = {}

        self.raw_file = None
        self.raw_writer = None
        self.metrics_file = None
        self.metrics_writer = None
        self.ground_truth_file = None
        self.ground_truth_writer = None
        self.events_file = None
        self.events_writer = None

        self.port_var = tk.StringVar(value=args.port)
        self.baud_var = tk.StringVar(value=str(args.baud))
        self.state_var = tk.StringVar(value="Idle")
        self.sensor_var = tk.StringVar(value="Sensor disconnected")
        self.elapsed_var = tk.StringVar(value="00:00")
        self.subject_var = tk.StringVar(value="No subject selected")
        self.channel_var = tk.StringVar(value=args.preview_channel)
        self.protocol_var = tk.StringVar(value=args.protocol)
        self.operator_var = tk.StringVar(value=args.operator)
        self.resp_target_brpm_var = tk.StringVar(value="" if args.resp_target_brpm is None else str(args.resp_target_brpm))
        self.overlay_filtered_var = tk.BooleanVar(value=True)
        self.overlay_peaks_var = tk.BooleanVar(value=True)
        self.orient_raw_var = tk.BooleanVar(value=True)
        self.auto_best_channel_var = tk.BooleanVar(value=False)
        self.auto_analyze_var = tk.BooleanVar(value=False)
        self.analysis_status_var = tk.StringVar(value="Analysis: not run")
        self.calibration_status_var = tk.StringVar(
            value=(
                f"Calibration: {calibration_display_name(self.calibration)}"
                if self.calibration
                else "Calibration: none"
            )
        )
        self.spo2_anchor_ratio_var = tk.StringVar(
            value="" if self.default_spo2_anchor_ratio is None else str(self.default_spo2_anchor_ratio)
        )
        self.spo2_anchor_pct_var = tk.StringVar(
            value="" if self.default_spo2_anchor_pct is None else str(self.default_spo2_anchor_pct)
        )
        self.window_sec_var = tk.StringVar(value=str(args.window_sec))

        self.metric_vars = {
            "HR": tk.StringVar(value="--"),
            "SpO2": tk.StringVar(value="--"),
            "PI": tk.StringVar(value="--"),
            "RR": tk.StringVar(value="--"),
            "RMSSD": tk.StringVar(value="--"),
            "SDNN": tk.StringVar(value="--"),
            "Quality": tk.StringVar(value="--"),
            "Best Channel": tk.StringVar(value="--"),
            "Tissue O2": tk.StringVar(value="post-run"),
            "Hb": tk.StringVar(value="ML post-run"),
            "Stiffness": tk.StringVar(value="post-run"),
        }
        self.ground_truth_vars = {
            "pulseox_model": tk.StringVar(value=""),
            "pulseox_hr_bpm": tk.StringVar(value=""),
            "pulseox_spo2_pct": tk.StringVar(value=""),
            "pulseox_pi_pct": tk.StringVar(value=""),
            "rr_ref_bpm": tk.StringVar(value=""),
            "ecg_hr_bpm": tk.StringVar(value=""),
            "ecg_hrv_metric": tk.StringVar(value=""),
            "operator_comment": tk.StringVar(value=""),
        }

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(60, self._process_queue)
        self.root.after(250, self._refresh_plot_and_metrics)

    def _build_ui(self) -> None:
        self.root.title("Multispectral PPG Acquisition")
        self.root.geometry("1450x860")
        self.root.minsize(1100, 700)
        self.root.configure(bg="#E2E8F0")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#F8FAFC")
        style.configure("Card.TFrame", background="#FFFFFF", relief="flat")
        style.configure("TLabel", background="#F8FAFC", foreground="#0F172A")
        style.configure("Muted.TLabel", foreground="#64748B")
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))

        header = tk.Frame(self.root, bg="#0F172A", height=86)
        header.pack(fill="x")
        header.pack_propagate(False)

        title_area = tk.Frame(header, bg="#0F172A")
        title_area.pack(side="left", fill="y", padx=22)
        tk.Label(
            title_area,
            text="Multispectral PPG Acquisition",
            bg="#0F172A",
            fg="#FFFFFF",
            font=("Segoe UI", 20, "bold"),
        ).pack(anchor="w", pady=(14, 0))
        tk.Label(
            title_area,
            text="Live waveform, real-time metrics, dataset logging, ground-truth snapshots",
            bg="#0F172A",
            fg="#CBD5E1",
            font=("Segoe UI", 10),
        ).pack(anchor="w")

        status_area = tk.Frame(header, bg="#0F172A")
        status_area.pack(side="right", fill="y", padx=18)
        for var in (self.sensor_var, self.state_var, self.elapsed_var):
            tk.Label(
                status_area,
                textvariable=var,
                bg="#1E293B",
                fg="#E2E8F0",
                font=("Segoe UI", 10, "bold"),
                padx=12,
                pady=7,
            ).pack(side="left", padx=5, pady=25)

        controls = ttk.Frame(self.root, padding=(16, 10))
        controls.pack(fill="x")
        ttk.Label(controls, text="Port").pack(side="left")
        ttk.Entry(controls, textvariable=self.port_var, width=8).pack(side="left", padx=(5, 10))
        ttk.Label(controls, text="Baud").pack(side="left")
        ttk.Entry(controls, textvariable=self.baud_var, width=8).pack(side="left", padx=(5, 10))
        self.connect_btn = ttk.Button(controls, text="Connect Sensor", command=self.connect_sensor)
        self.connect_btn.pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Select / Add Subject", command=self.select_subject).pack(side="left", padx=(0, 8))
        self.start_btn = ttk.Button(
            controls,
            text="Start Recording",
            command=self.start_recording,
            state="disabled",
            style="Accent.TButton",
        )
        self.start_btn.pack(side="left", padx=(0, 8))
        self.stop_btn = ttk.Button(controls, text="Stop Recording", command=self.stop_recording, state="disabled")
        self.stop_btn.pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Mark Event", command=self.mark_event).pack(side="left", padx=(0, 8))
        self.analysis_btn = ttk.Button(controls, text="Run Analysis", command=self.run_final_analysis, state="disabled")
        self.analysis_btn.pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Open Subject Folder", command=self.open_subject_folder).pack(side="left")

        body = tk.Frame(self.root, bg="#E2E8F0")
        body.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        left = ttk.Frame(body)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        left.configure(width=290)
        left.grid_propagate(False)
        left_content = self._make_scrollable_panel(left)
        self._build_left_panel(left_content)

        center = ttk.Frame(body, padding=12)
        center.grid(row=0, column=1, sticky="nsew")
        center.grid_rowconfigure(1, weight=1)
        center.grid_columnconfigure(0, weight=1)
        self._build_waveform_panel(center)

        right = ttk.Frame(body)
        right.grid(row=0, column=2, sticky="nse", padx=(12, 0))
        right.configure(width=360)
        right.grid_propagate(False)
        right_content = self._make_scrollable_panel(right)
        self._build_right_panel(right_content)

    def _make_scrollable_panel(self, parent: ttk.Frame) -> ttk.Frame:
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        canvas = tk.Canvas(parent, bg="#F8FAFC", highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        content = ttk.Frame(canvas, padding=12)
        window_id = canvas.create_window((0, 0), window=content, anchor="nw")

        def on_content_configure(_event: tk.Event) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def on_canvas_configure(event: tk.Event) -> None:
            canvas.itemconfigure(window_id, width=event.width)

        def on_mousewheel(event: tk.Event) -> None:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def bind_mousewheel(_event: tk.Event) -> None:
            canvas.bind_all("<MouseWheel>", on_mousewheel)

        def unbind_mousewheel(_event: tk.Event) -> None:
            canvas.unbind_all("<MouseWheel>")

        content.bind("<Configure>", on_content_configure)
        canvas.bind("<Configure>", on_canvas_configure)
        canvas.bind("<Enter>", bind_mousewheel)
        canvas.bind("<Leave>", unbind_mousewheel)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        return content

    def _section_label(self, parent: tk.Widget, text: str) -> None:
        ttk.Label(parent, text=text, style="Muted.TLabel", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(8, 6))

    def _card(self, parent: tk.Widget, padding: int = 12) -> ttk.Frame:
        frame = ttk.Frame(parent, style="Card.TFrame", padding=padding)
        frame.pack(fill="x", pady=(0, 12))
        return frame

    def _build_left_panel(self, parent: ttk.Frame) -> None:
        self._section_label(parent, "Current Subject")
        card = self._card(parent)
        ttk.Label(card, textvariable=self.subject_var, background="#FFFFFF", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        self.subject_detail = tk.Text(card, height=8, width=30, bg="#FFFFFF", relief="flat", fg="#334155", font=("Consolas", 9))
        self.subject_detail.pack(fill="x", pady=(8, 0))
        self.subject_detail.insert("1.0", "Select or create a subject to enable recording.")
        self.subject_detail.configure(state="disabled")

        self._section_label(parent, "Run Setup")
        setup = self._card(parent)
        ttk.Label(setup, text="Protocol", background="#FFFFFF").pack(anchor="w")
        ttk.Combobox(
            setup,
            textvariable=self.protocol_var,
            values=["resting", "controlled_breathing", "post_exercise_recovery", "post_cold_recovery", "custom"],
        ).pack(fill="x", pady=(2, 8))
        ttk.Label(setup, text="Controlled RR target (brpm)", background="#FFFFFF").pack(anchor="w")
        ttk.Entry(setup, textvariable=self.resp_target_brpm_var).pack(fill="x", pady=(2, 8))
        ttk.Label(setup, text="SpO2 anchor ratio", background="#FFFFFF").pack(anchor="w")
        ttk.Entry(setup, textvariable=self.spo2_anchor_ratio_var).pack(fill="x", pady=(2, 8))
        ttk.Label(setup, text="SpO2 anchor percent", background="#FFFFFF").pack(anchor="w")
        ttk.Entry(setup, textvariable=self.spo2_anchor_pct_var).pack(fill="x", pady=(2, 8))
        ttk.Checkbutton(setup, text="Auto-run final analysis on Stop", variable=self.auto_analyze_var).pack(anchor="w")
        ttk.Label(setup, textvariable=self.calibration_status_var, background="#FFFFFF", foreground="#64748B", wraplength=250, justify="left").pack(anchor="w", pady=(6, 0))
        ttk.Label(setup, textvariable=self.analysis_status_var, background="#FFFFFF", foreground="#64748B").pack(anchor="w", pady=(6, 0))

        self._section_label(parent, "Signal Controls")
        signal = self._card(parent)
        ttk.Label(signal, text="Displayed channel", background="#FFFFFF").pack(anchor="w")
        self.channel_combo = ttk.Combobox(signal, textvariable=self.channel_var, values=[], state="readonly")
        self.channel_combo.pack(fill="x", pady=(2, 8))
        ttk.Checkbutton(signal, text="Auto-follow best channel", variable=self.auto_best_channel_var).pack(anchor="w")
        ttk.Checkbutton(signal, text="Show raw in physiological orientation", variable=self.orient_raw_var).pack(anchor="w")
        ttk.Checkbutton(signal, text="Overlay filtered signal", variable=self.overlay_filtered_var).pack(anchor="w")
        ttk.Checkbutton(signal, text="Overlay detected peaks", variable=self.overlay_peaks_var).pack(anchor="w")
        ttk.Label(signal, text="Plot window seconds", background="#FFFFFF").pack(anchor="w", pady=(8, 0))
        ttk.Entry(signal, textvariable=self.window_sec_var).pack(fill="x", pady=(2, 0))

    def _build_waveform_panel(self, parent: ttk.Frame) -> None:
        top = ttk.Frame(parent)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(top, text="Live Waveform", font=("Segoe UI", 14, "bold")).pack(side="left")
        self.waveform_status = tk.StringVar(value="Connect sensor to start preview.")
        ttk.Label(top, textvariable=self.waveform_status, style="Muted.TLabel").pack(side="right")

        fig = Figure(figsize=(8.0, 5.4), dpi=100, facecolor="#FFFFFF")
        self.ax_raw = fig.add_subplot(211)
        self.ax_filt = fig.add_subplot(212, sharex=self.ax_raw)
        self.raw_line, = self.ax_raw.plot([], [], color="#0EA5E9", linewidth=1.7, label="raw")
        self.filtered_line, = self.ax_filt.plot([], [], color="#F97316", linewidth=1.3, label="heart-band filtered")
        self.peaks_line, = self.ax_filt.plot([], [], "o", color="#22C55E", markersize=4, label="detected peaks")
        self.ax_raw.set_ylabel("ADC counts")
        self.ax_filt.set_ylabel("filtered")
        self.ax_filt.set_xlabel("time (s)")
        self.ax_raw.grid(True, color="#E2E8F0", linewidth=0.8)
        self.ax_filt.grid(True, color="#E2E8F0", linewidth=0.8)
        self.ax_raw.legend(loc="upper right")
        self.ax_filt.legend(loc="upper right")
        fig.tight_layout()
        self.figure = fig

        canvas_frame = ttk.Frame(parent)
        canvas_frame.grid(row=1, column=0, sticky="nsew")
        self.canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        self._section_label(parent, "Live Metrics")
        metrics_grid = ttk.Frame(parent, style="Card.TFrame", padding=10)
        metrics_grid.pack(fill="x", pady=(0, 12))
        metric_specs = [
            ("HR", "bpm"),
            ("SpO2", "% est."),
            ("PI", "% proxy"),
            ("RR", "brpm"),
            ("Quality", "q"),
            ("Best Channel", ""),
        ]
        for idx, (name, unit) in enumerate(metric_specs):
            card = tk.Frame(metrics_grid, bg="#FFFFFF", highlightbackground="#E2E8F0", highlightthickness=1)
            card.grid(row=idx // 2, column=idx % 2, sticky="nsew", padx=4, pady=4)
            metrics_grid.grid_columnconfigure(idx % 2, weight=1)
            tk.Label(card, text=name, bg="#FFFFFF", fg="#64748B", font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=10, pady=(7, 0))
            tk.Label(card, textvariable=self.metric_vars[name], bg="#FFFFFF", fg="#0F172A", font=("Segoe UI", 15, "bold")).pack(anchor="w", padx=10)
            tk.Label(card, text=unit, bg="#FFFFFF", fg="#94A3B8", font=("Segoe UI", 8)).pack(anchor="w", padx=10, pady=(0, 7))

        self._section_label(parent, "Ground Truth Snapshot")
        gt = self._card(parent, padding=10)
        ttk.Label(
            gt,
            background="#FFFFFF",
            foreground="#64748B",
            wraplength=300,
        ).pack(anchor="w", pady=(0, 8))
        fields = [
            ("Pulseox model", "pulseox_model"),
            ("Pulseox HR", "pulseox_hr_bpm"),
            ("Pulseox SpO2", "pulseox_spo2_pct"),
            ("Pulseox PI", "pulseox_pi_pct"),
            ("RR reference", "rr_ref_bpm"),
        ]
        for label, key in fields:
            row = ttk.Frame(gt, style="Card.TFrame")
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label, width=16, background="#FFFFFF").pack(side="left")
            ttk.Entry(row, textvariable=self.ground_truth_vars[key], width=20).pack(side="left", fill="x", expand=True)
        ttk.Button(gt, text="Capture Ground Truth Snapshot", command=self.capture_ground_truth).pack(fill="x", pady=(10, 0))

        self._section_label(parent, "Rolling / Post-run Research")
        research_grid = ttk.Frame(parent, style="Card.TFrame", padding=10)
        research_grid.pack(fill="x", pady=(0, 12))
        research_specs = [
            ("RMSSD", "ms"),
            ("SDNN", "ms"),
            ("Tissue O2", "post-run"),
            ("Hb", "ML"),
            ("Stiffness", "post-run"),
        ]
        for idx, (name, unit) in enumerate(research_specs):
            card = tk.Frame(research_grid, bg="#FFFFFF", highlightbackground="#E2E8F0", highlightthickness=1)
            card.grid(row=idx // 2, column=idx % 2, sticky="nsew", padx=4, pady=4)
            research_grid.grid_columnconfigure(idx % 2, weight=1)
            tk.Label(card, text=name, bg="#FFFFFF", fg="#64748B", font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=10, pady=(7, 0))
            tk.Label(card, textvariable=self.metric_vars[name], bg="#FFFFFF", fg="#0F172A", font=("Segoe UI", 14, "bold")).pack(anchor="w", padx=10)
            tk.Label(card, text=unit, bg="#FFFFFF", fg="#94A3B8", font=("Segoe UI", 8)).pack(anchor="w", padx=10, pady=(0, 7))

    def connect_sensor(self) -> None:
        if self.ser is not None:
            self.disconnect_sensor()
            return
        try:
            port = self.port_var.get().strip()
            baud = int(self.baud_var.get().strip())
            self.sensor_var.set("Connecting...")
            self.root.update_idletasks()
            ser = open_serial_reset(port, baud)
            header = read_header(ser)

            self.ser = ser
            self.header = header
            self.columns = signal_columns(header)
            self.column_indices = {name: idx for idx, name in enumerate(header)}
            self.us_idx = self.column_indices.get("us")
            self.ms_idx = self.column_indices.get("ms")
            self.t0_sensor = None
            self.tbuf = deque(maxlen=self.args.max_samples)
            self.series = {col: deque(maxlen=self.args.max_samples) for col in self.columns}
            self.stable_pulse_polarity = {}

            chosen = self.args.preview_channel if self.args.preview_channel in self.columns else preferred_preview_channel(self.columns)
            self.channel_var.set(chosen)
            self.channel_combo.configure(values=self.columns)

            self.reader_running = True
            self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.reader_thread.start()
            self.sensor_var.set(f"Connected {port}")
            self.state_var.set("Preview")
            self.waveform_status.set(f"{len(self.columns)} signal columns detected")
            self.connect_btn.configure(text="Disconnect")
            self._update_button_states()
        except Exception as exc:
            self.sensor_var.set("Sensor disconnected")
            if self.ser is not None:
                self.ser.close()
                self.ser = None
            messagebox.showerror("Sensor connection", str(exc), parent=self.root)

    def disconnect_sensor(self) -> None:
        if self.recording:
            messagebox.showwarning("Recording active", "Stop recording before disconnecting the sensor.", parent=self.root)
            return
        self.reader_running = False
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
        self.ser = None
        self.sensor_var.set("Sensor disconnected")
        self.state_var.set("Idle")
        self.waveform_status.set("Connect sensor to start preview.")
        self.connect_btn.configure(text="Connect Sensor")
        self._update_button_states()

    def _reader_loop(self) -> None:
        assert self.ser is not None
        while self.reader_running:
            try:
                raw = self.ser.readline().decode(errors="ignore").strip()
            except Exception:
                break
            if not raw or raw.startswith("#") or not raw[0].isdigit():
                continue
            parts = raw.split(",")
            if len(parts) != len(self.header):
                continue
            try:
                self.sample_queue.put_nowait((time.time(), parts))
            except queue.Full:
                pass

    def _sample_time_s(self, parts: list[str], pc_time_s: float) -> Optional[float]:
        try:
            if self.us_idx is not None:
                value = float(parts[self.us_idx])
                if self.t0_sensor is None:
                    self.t0_sensor = value
                return (value - self.t0_sensor) / 1_000_000.0
            if self.ms_idx is not None:
                value = float(parts[self.ms_idx])
                if self.t0_sensor is None:
                    self.t0_sensor = value
                return (value - self.t0_sensor) / 1000.0
        except Exception:
            return None

        if self.t0_sensor is None:
            self.t0_sensor = pc_time_s
        return pc_time_s - self.t0_sensor

    def _process_queue(self) -> None:
        processed = 0
        while processed < 500:
            try:
                pc_time_s, parts = self.sample_queue.get_nowait()
            except queue.Empty:
                break
            processed += 1
            t_s = self._sample_time_s(parts, pc_time_s)
            if t_s is None:
                continue
            self.tbuf.append(t_s)
            for col in self.columns:
                idx = self.column_indices[col]
                try:
                    value = float(parts[idx])
                except Exception:
                    value = np.nan
                self.series[col].append(value)

            if self.recording and self.raw_writer is not None:
                self.raw_writer.writerow([pc_time_s, *parts])
                self.rows_written += 1
                if self.rows_written % 50 == 0 and self.raw_file is not None:
                    self.raw_file.flush()

        self.root.after(60, self._process_queue)

    def _window_seconds(self) -> float:
        try:
            return max(8.0, float(self.window_sec_var.get().strip()))
        except ValueError:
            return 25.0

    def _float_or_none(self, var: tk.StringVar) -> Optional[float]:
        text = var.get().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _current_resp_band(self) -> tuple[float, float]:
        protocol = self.protocol_var.get().strip()
        target = self._float_or_none(self.resp_target_brpm_var)
        if protocol == "controlled_breathing" and target is not None and target > 0:
            center = target / 60.0
            return (max(0.05, center - 4.0 / 60.0), min(0.95, center + 4.0 / 60.0))
        if protocol == "post_exercise_recovery":
            return (0.10, 0.70)
        return (0.15, 0.35)

    def _refresh_plot_and_metrics(self) -> None:
        try:
            self._update_elapsed()
            if self.columns and len(self.tbuf) > 10:
                self._draw_current_window()
        finally:
            self.root.after(250, self._refresh_plot_and_metrics)

    def _draw_current_window(self) -> None:
        selected = self.channel_var.get()
        if selected not in self.columns:
            return

        t_all = np.asarray(self.tbuf, dtype=float)
        if len(t_all) < 10:
            return
        keep = t_all >= (t_all[-1] - self._window_seconds())
        t_win = t_all[keep]
        if len(t_win) < 10:
            return

        series_win = {}
        for col in self.columns:
            values = np.asarray(self.series[col], dtype=float)
            if len(values) == len(t_all):
                series_win[col] = values[keep]

        if self.auto_best_channel_var.get() and self.latest_metrics.best_channel in self.columns:
            selected = self.latest_metrics.best_channel
            self.channel_var.set(selected)

        preferred_polarity = self.stable_pulse_polarity.get(selected)
        metrics, filtered, peak_t, peak_y = compute_live_metrics(
            t_win,
            series_win,
            selected,
            self.columns,
            spo2_anchor_ratio=self._float_or_none(self.spo2_anchor_ratio_var),
            spo2_anchor_pct=self._float_or_none(self.spo2_anchor_pct_var),
            spo2_slope=self.default_spo2_slope,
            spo2_clip_min=self.default_spo2_clip_min,
            spo2_clip_max=self.default_spo2_clip_max,
            resp_band=self._current_resp_band(),
            protocol=self.protocol_var.get().strip(),
            resp_target_brpm=self._float_or_none(self.resp_target_brpm_var),
            calibration=self.calibration,
            preferred_pulse_polarity=preferred_polarity,
        )
        if metrics.pulse_polarity in (-1, 1):
            self.stable_pulse_polarity[selected] = int(metrics.pulse_polarity)
        self.latest_metrics = metrics

        y = series_win.get(selected)
        if y is None:
            return

        raw_display = y
        if self.orient_raw_var.get() and metrics.pulse_polarity == -1:
            finite_y = y[np.isfinite(y)]
            if len(finite_y):
                raw_display = 2.0 * float(np.nanmedian(finite_y)) - y

        self.raw_line.set_data(t_win, raw_display)
        if self.overlay_filtered_var.get() and filtered is not None:
            self.filtered_line.set_data(t_win, filtered)
        else:
            self.filtered_line.set_data([], [])

        if self.overlay_peaks_var.get() and peak_t is not None and peak_y is not None:
            self.peaks_line.set_data(peak_t, peak_y)
        else:
            self.peaks_line.set_data([], [])

        self.ax_raw.relim()
        self.ax_raw.autoscale_view()
        self.ax_filt.relim()
        self.ax_filt.autoscale_view()
        self.canvas.draw_idle()

        self._update_metric_cards(metrics)
        fs_text = fmt(metrics.fs_hz, 2, " Hz")
        q_text = fmt(metrics.signal_quality, 1)
        sat_text = fmt(None if metrics.saturation_fraction is None else 100.0 * metrics.saturation_fraction, 1, "%")
        self.waveform_status.set(f"channel={selected} | fs={fs_text} | q={q_text} | saturation={sat_text}")

        now_pc = time.time()
        if self.recording and self.metrics_writer is not None and now_pc - self.last_metrics_write_pc >= 1.0:
            self._write_metrics_row(now_pc, metrics)
            self.last_metrics_write_pc = now_pc

    def _update_metric_cards(self, metrics: LiveMetrics) -> None:
        self.metric_vars["HR"].set(fmt(metrics.hr_bpm, 1))
        self.metric_vars["SpO2"].set(fmt(metrics.spo2_estimated_pct, 1) if metrics.spo2_estimated_pct is not None else metrics.spo2_status)
        self.metric_vars["PI"].set(fmt(metrics.perfusion_index_pct, 2))
        self.metric_vars["RR"].set(fmt(metrics.respiratory_rate_brpm, 1))
        self.metric_vars["RMSSD"].set(fmt(metrics.rmssd_ms, 1))
        self.metric_vars["SDNN"].set(fmt(metrics.sdnn_ms, 1))
        self.metric_vars["Quality"].set(fmt(metrics.signal_quality, 1))
        self.metric_vars["Best Channel"].set(metrics.best_channel or "--")

    def select_subject(self) -> None:
        dialog = SubjectDialog(self.root, self.dataset_dir)
        self.root.wait_window(dialog)
        if dialog.result is None:
            return
        self.subject_id, self.subject_meta = dialog.result
        self._refresh_subject_panel()
        self._update_button_states()

    def _refresh_subject_panel(self) -> None:
        if self.subject_id is None:
            self.subject_var.set("No subject selected")
            return
        name = self.subject_meta.get("name", "")
        self.subject_var.set(f"{self.subject_id}" + (f" | {name}" if name else ""))
        lines = [
            f"age: {self.subject_meta.get('age', '--')}",
            f"sex: {self.subject_meta.get('sex', '--')}",
            f"height_cm: {self.subject_meta.get('height_cm', '--')}",
            f"weight_kg: {self.subject_meta.get('weight_kg', '--')}",
            f"bmi: {fmt(self.subject_meta.get('bmi'), 1)}",
            f"skin_tone: {self.subject_meta.get('skin_tone', '--')}",
            f"site: {self.subject_meta.get('default_measurement_site', '--')}",
        ]
        self.subject_detail.configure(state="normal")
        self.subject_detail.delete("1.0", "end")
        self.subject_detail.insert("1.0", "\n".join(lines))
        self.subject_detail.configure(state="disabled")

    def _update_button_states(self) -> None:
        can_start = self.ser is not None and self.subject_id is not None and not self.recording
        self.start_btn.configure(state="normal" if can_start else "disabled")
        self.stop_btn.configure(state="normal" if self.recording else "disabled")
        can_analyze = self.run_dir is not None and not self.recording
        self.analysis_btn.configure(state="normal" if can_analyze else "disabled")

    def _run_elapsed_s(self, now_pc: Optional[float] = None) -> float:
        if self.recording_started_pc is None:
            return 0.0
        return (now_pc or time.time()) - self.recording_started_pc

    def _update_elapsed(self) -> None:
        elapsed = self._run_elapsed_s() if self.recording else 0.0
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        self.elapsed_var.set(f"{minutes:02d}:{seconds:02d}")

    def start_recording(self) -> None:
        if self.subject_id is None or self.ser is None:
            return
        try:
            timestamp = datetime.now()
            self.run_dir = create_run_dir(self.dataset_dir, self.subject_id, timestamp)
            raw_path = self.run_dir / "raw" / "tcs3448_raw.csv"
            metrics_path = self.run_dir / "metrics_live.csv"
            gt_path = self.run_dir / "ground_truth.csv"
            events_path = self.run_dir / "events.csv"
            notes_path = self.run_dir / "notes.txt"

            self.raw_file = open(raw_path, "w", newline="", encoding="utf-8")
            self.raw_writer = csv.writer(self.raw_file)
            self.raw_writer.writerow(["pc_time_s", *self.header])

            self.metrics_file = open(metrics_path, "w", newline="", encoding="utf-8")
            self.metrics_writer = csv.DictWriter(self.metrics_file, fieldnames=["timestamp_iso", "timestamp_rel_s", *METRIC_FIELDS])
            self.metrics_writer.writeheader()

            self.ground_truth_file = open(gt_path, "w", newline="", encoding="utf-8")
            self.ground_truth_writer = csv.DictWriter(
                self.ground_truth_file,
                fieldnames=["timestamp_iso", "timestamp_rel_s", *self.ground_truth_vars.keys()],
            )
            self.ground_truth_writer.writeheader()

            self.events_file = open(events_path, "w", newline="", encoding="utf-8")
            self.events_writer = csv.DictWriter(self.events_file, fieldnames=["timestamp_iso", "timestamp_rel_s", "event_type", "label"])
            self.events_writer.writeheader()
            notes_path.write_text("", encoding="utf-8")

            self.rows_written = 0
            self.last_metrics_write_pc = 0.0
            self.recording_started_pc = time.time()
            self.recording = True
            self.state_var.set("Recording")

            self.run_meta = {
                "run_id": self.run_dir.name,
                "subject_id": self.subject_id,
                "start_time": timestamp.isoformat(timespec="seconds"),
                "operator": self.operator_var.get().strip(),
                "protocol": self.protocol_var.get().strip(),
                "respiration_target_brpm": self._float_or_none(self.resp_target_brpm_var),
                "port": self.port_var.get().strip(),
                "baud": int(self.baud_var.get().strip()),
                "app_version": APP_VERSION,
                "columns": self.header,
                "selected_channel": self.channel_var.get(),
                "spo2_anchor_ratio": self._float_or_none(self.spo2_anchor_ratio_var),
                "spo2_anchor_pct": self._float_or_none(self.spo2_anchor_pct_var),
                "spo2_slope": self.default_spo2_slope,
                "calibration_file": str(self.calibration_path) if self.calibration else None,
                "calibration_name": calibration_display_name(self.calibration) if self.calibration else None,
                "subject_metadata_snapshot": self.subject_meta,
            }
            save_json(self.run_dir / "meta" / "run_metadata.json", self.run_meta)
            save_json(self.run_dir / "run_metadata.json", self.run_meta)
            self._write_event("recording_start", self.protocol_var.get().strip() or "run")
            self._update_button_states()
        except Exception as exc:
            messagebox.showerror("Start recording", str(exc), parent=self.root)
            self._close_recording_files()
            self.recording = False
            self.state_var.set("Preview")
            self._update_button_states()

    def stop_recording(self) -> None:
        if not self.recording:
            return
        now = datetime.now()
        try:
            self._write_event("recording_stop", self.protocol_var.get().strip() or "run")
            self.recording = False
            self.state_var.set("Preview")
            self.run_meta.update(
                {
                    "stop_time": now.isoformat(timespec="seconds"),
                    "duration_s": self._run_elapsed_s(),
                    "rows_written": self.rows_written,
                    "final_selected_channel": self.channel_var.get(),
                    "final_live_metrics": self.latest_metrics.to_row(),
                }
            )
            if self.run_dir is not None:
                save_json(self.run_dir / "meta" / "run_metadata.json", self.run_meta)
                save_json(self.run_dir / "run_metadata.json", self.run_meta)
        finally:
            self._close_recording_files()
            self._update_button_states()
            if self.auto_analyze_var.get() and self.run_dir is not None:
                self.run_final_analysis()

    def _close_recording_files(self) -> None:
        for attr in ("raw_file", "metrics_file", "ground_truth_file", "events_file"):
            f = getattr(self, attr)
            if f is not None:
                try:
                    f.flush()
                    f.close()
                except Exception:
                    pass
                setattr(self, attr, None)
        self.raw_writer = None
        self.metrics_writer = None
        self.ground_truth_writer = None
        self.events_writer = None

    def _write_metrics_row(self, now_pc: float, metrics: LiveMetrics) -> None:
        if self.metrics_writer is None:
            return
        row = {
            "timestamp_iso": datetime.now().isoformat(timespec="milliseconds"),
            "timestamp_rel_s": self._run_elapsed_s(now_pc),
            **metrics.to_row(),
        }
        self.metrics_writer.writerow(row)
        if self.metrics_file is not None:
            self.metrics_file.flush()

    def _write_event(self, event_type: str, label: str) -> None:
        if self.events_writer is None:
            return
        self.events_writer.writerow(
            {
                "timestamp_iso": datetime.now().isoformat(timespec="milliseconds"),
                "timestamp_rel_s": self._run_elapsed_s(),
                "event_type": event_type,
                "label": label,
            }
        )
        if self.events_file is not None:
            self.events_file.flush()

    def mark_event(self) -> None:
        if not self.recording:
            messagebox.showinfo("Mark event", "Start recording before marking events.", parent=self.root)
            return
        label = simpledialog.askstring("Mark Event", "Event label:", parent=self.root)
        if label:
            self._write_event("manual", label.strip())

    def capture_ground_truth(self) -> None:
        if not self.recording or self.ground_truth_writer is None:
            messagebox.showinfo("Ground truth", "Start recording before capturing a ground-truth snapshot.", parent=self.root)
            return
        self._sync_controlled_rr_reference()
        row = {
            "timestamp_iso": datetime.now().isoformat(timespec="milliseconds"),
            "timestamp_rel_s": self._run_elapsed_s(),
        }
        row.update({key: var.get().strip() for key, var in self.ground_truth_vars.items()})
        self.ground_truth_writer.writerow(row)
        if self.ground_truth_file is not None:
            self.ground_truth_file.flush()

    def _sync_controlled_rr_reference(self) -> None:
        if self.protocol_var.get().strip() != "controlled_breathing":
            return
        target = self.resp_target_brpm_var.get().strip()
        if target and not self.ground_truth_vars["rr_ref_bpm"].get().strip():
            self.ground_truth_vars["rr_ref_bpm"].set(target)

    def run_final_analysis(self) -> None:
        if self.run_dir is None:
            messagebox.showinfo("Run analysis", "No run has been recorded in this GUI session yet.", parent=self.root)
            return
        run_dir = self.run_dir
        self.analysis_status_var.set("Analysis: running...")
        self.analysis_btn.configure(state="disabled")
        threading.Thread(target=self._run_final_analysis_worker, args=(run_dir,), daemon=True).start()

    def _analysis_command_for_run(self, run_dir: Path) -> list[str]:
        raw_path = run_dir / "raw" / "tcs3448_raw.csv"
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw CSV not found: {raw_path}")

        with open(raw_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")

        columns = set(header)
        anchor_ratio = self._float_or_none(self.spo2_anchor_ratio_var)
        anchor_pct = self._float_or_none(self.spo2_anchor_pct_var)

        if {"F6", "NIR"}.issubset(columns):
            cmd = [
                sys.executable,
                str(ROOT / "lab" / "tcs3448_red_nir_12ch_lab" / "tools" / "analyze_red_nir_12ch.py"),
                "--input",
                str(raw_path),
                "--output-dir",
                str(run_dir / "analysis" / "red_nir_12ch"),
            ]
            if anchor_ratio is not None:
                cmd.extend(["--spo2-anchor-ratio", str(anchor_ratio)])
            if anchor_pct is not None:
                cmd.extend(["--spo2-anchor-pct", str(anchor_pct)])
            cmd.extend(["--spo2-slope", str(self.default_spo2_slope)])
            if self.calibration_path.exists():
                cmd.extend(["--calibration-file", str(self.calibration_path)])
            return cmd

        if {"FZ_diff", "NIR_diff"}.issubset(columns):
            cmd = [
                sys.executable,
                str(ROOT / "apps" / "analyze_run.py"),
                "--input",
                str(raw_path),
                "--channel",
                "auto_fz_nir",
            ]
            height_cm = self.subject_meta.get("height_cm")
            if height_cm is not None:
                cmd.extend(["--height-cm", str(height_cm)])
            if anchor_ratio is not None:
                cmd.extend(["--spo2-anchor-ratio", str(anchor_ratio)])
            if anchor_pct is not None:
                cmd.extend(["--spo2-anchor-pct", str(anchor_pct)])
            cmd.extend(["--spo2-slope", str(self.default_spo2_slope)])
            return cmd

        selected = self.channel_var.get()
        if selected in columns:
            return [
                sys.executable,
                str(ROOT / "apps" / "analyze_run.py"),
                "--input",
                str(raw_path),
                "--channel",
                selected,
            ]

        raise RuntimeError(f"Could not infer analysis mode from CSV columns: {header}")

    def _run_final_analysis_worker(self, run_dir: Path) -> None:
        try:
            cmd = self._analysis_command_for_run(run_dir)
            log_path = run_dir / "analysis" / "analysis_log.txt"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            completed = subprocess.run(
                cmd,
                cwd=str(ROOT),
                text=True,
                capture_output=True,
                check=False,
            )
            log_path.write_text(
                "COMMAND:\n"
                + " ".join(cmd)
                + "\n\nSTDOUT:\n"
                + completed.stdout
                + "\n\nSTDERR:\n"
                + completed.stderr,
                encoding="utf-8",
            )
            if completed.returncode != 0:
                raise RuntimeError(f"analysis failed; see {log_path}")
            self.root.after(0, lambda: self._analysis_finished(run_dir, None))
        except Exception as exc:
            self.root.after(0, lambda exc=exc: self._analysis_finished(run_dir, exc))

    def _analysis_finished(self, run_dir: Path, error: Exception | None) -> None:
        if error is None:
            self.analysis_status_var.set(f"Analysis: saved in {run_dir / 'analysis'}")
        else:
            self.analysis_status_var.set("Analysis: failed")
            messagebox.showerror("Run analysis", str(error), parent=self.root)
        self._update_button_states()

    def open_subject_folder(self) -> None:
        if self.subject_id is None:
            messagebox.showinfo("Subject folder", "Select a subject first.", parent=self.root)
            return
        path = self.dataset_dir / self.subject_id
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(path)

    def close(self) -> None:
        if self.recording:
            if not messagebox.askyesno("Recording active", "Stop recording and close the app?", parent=self.root):
                return
            self.stop_recording()
        self.reader_running = False
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
        self._close_recording_files()
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Desktop GUI for TCS3448 PPG acquisition and live metrics.")
    parser.add_argument("--dataset-dir", default=str(ROOT / "dataset"))
    parser.add_argument("--port", default="COM5")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--preview-channel", default="NIR_diff")
    parser.add_argument("--protocol", default="resting")
    parser.add_argument("--operator", default="")
    parser.add_argument("--resp-target-brpm", type=float, default=None)
    parser.add_argument("--window-sec", type=float, default=25.0)
    parser.add_argument("--max-samples", type=int, default=6000)
    parser.add_argument("--spo2-anchor-ratio", type=float, default=None)
    parser.add_argument("--spo2-anchor-pct", type=float, default=None)
    parser.add_argument("--spo2-slope", type=float, default=None)
    parser.add_argument("--calibration-file", default=None)
    args = parser.parse_args()

    root = tk.Tk()
    PPGDashboard(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
