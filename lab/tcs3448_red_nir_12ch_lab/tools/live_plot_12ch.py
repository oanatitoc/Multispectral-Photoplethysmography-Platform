from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import serial
from scipy.signal import butter, filtfilt


HEART_BAND = (0.7, 4.0)
NON_SIGNAL_COLUMNS = {"ms", "us", "astatus"}


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
        if not line:
            continue
        if line.startswith("#"):
            print(line)
            continue
        if line.startswith("ms,"):
            return [x.strip() for x in line.split(",")]
    raise RuntimeError("No CSV header received from 12-channel lab firmware.")


def estimate_fs_us(us_values: np.ndarray) -> Optional[float]:
    if len(us_values) < 5:
        return None
    dt = np.diff(us_values) / 1_000_000.0
    dt = dt[(dt > 0.005) & (dt < 1.0)]
    if len(dt) < 3:
        return None
    return float(1.0 / np.median(dt))


def butter_bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 2) -> Optional[np.ndarray]:
    if fs is None or len(x) < max(20, int(fs * 2)):
        return None
    nyq = 0.5 * fs
    lo_n = lo / nyq
    hi_n = hi / nyq
    if hi_n >= 1.0:
        hi_n = 0.99
    if lo_n <= 0.0 or lo_n >= hi_n:
        return None
    b, a = butter(order, [lo_n, hi_n], btype="band")
    return filtfilt(b, a, x)


def fft_quality(x: np.ndarray, fs: float, band: tuple[float, float]) -> tuple[float, float]:
    if fs is None or len(x) < max(30, int(fs * 6)):
        return 0.0, np.nan
    x = np.asarray(x, float)
    x = x - np.mean(x)
    win = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * win))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return 0.0, np.nan
    fb = freqs[mask]
    sb = spec[mask]
    k = int(np.argmax(sb))
    q = float(sb[k]) / (float(np.median(sb)) + 1e-9)
    bpm = float(fb[k] * 60.0)
    return q, bpm


def saturation_fraction(y: np.ndarray, threshold: float) -> float:
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return np.nan
    return float(np.mean(y >= threshold))


def score_column(y: np.ndarray, fs: float, saturation_threshold: float) -> tuple[float, Optional[np.ndarray], float, float, float]:
    sat_frac = saturation_fraction(y, saturation_threshold)
    raw_range = float(np.nanpercentile(y, 95) - np.nanpercentile(y, 5))
    if np.isfinite(sat_frac) and sat_frac > 0.30:
        return 0.0, None, np.nan, sat_frac, raw_range
    yf = butter_bandpass(y, fs, HEART_BAND[0], HEART_BAND[1], order=2)
    if yf is None:
        return 0.0, None, np.nan, sat_frac, raw_range
    q_fft, bpm = fft_quality(yf, fs, HEART_BAND)
    amp = float(np.percentile(yf, 95) - np.percentile(yf, 5))
    sat_penalty = max(0.0, 1.0 - 2.0 * sat_frac) if np.isfinite(sat_frac) else 1.0
    score = float(max(0.0, q_fft) * max(amp, 0.0) * sat_penalty)
    return score, yf, bpm, sat_frac, raw_range


def signal_columns(header: list[str]) -> list[str]:
    return [col for col in header if col not in NON_SIGNAL_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Live monitor for TCS3448 12-channel red/NIR always-on lab firmware.")
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--preview-column", default=None, help="Exact CSV column name to plot, e.g. F6 or NIR.")
    parser.add_argument("--window-sec", type=float, default=25.0)
    parser.add_argument("--saturation-threshold", type=float, default=3580.0)
    parser.add_argument("--csv-out", default=None, help="Optional CSV file to record the serial stream.")
    args = parser.parse_args()

    ser = open_serial_reset(args.port, args.baud)
    header = read_header(ser)

    data_columns = signal_columns(header)
    if args.preview_column is not None and args.preview_column not in data_columns:
        raise ValueError(f"Preview column {args.preview_column} not found. Available: {data_columns}")

    print("Available signal columns:", ", ".join(data_columns))
    print("Tip: test fixed channels with --preview-column F6 or --preview-column NIR")

    us_idx = header.index("us")
    t0_us: Optional[float] = None
    last_draw = 0.0

    maxlen = 6000
    tbuf = deque(maxlen=maxlen)
    usbuf = deque(maxlen=maxlen)
    series = {col: deque(maxlen=maxlen) for col in data_columns}

    csv_writer = None
    csv_file = None
    if args.csv_out:
        csv_path = Path(args.csv_out)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["pc_time_s", *header])

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    raw_ln, = ax1.plot([], [], label="raw")
    flt_ln, = ax2.plot([], [], label="heart-band filtered")
    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")
    ax1.set_ylabel("ADC counts")
    ax2.set_ylabel("filtered")
    ax2.set_xlabel("time (s)")
    status_txt = ax1.text(0.02, 0.90, "selected=-- | fs=-- | bpm_fft=-- | score=--", transform=ax1.transAxes)

    try:
        while True:
            raw_line = ser.readline().decode(errors="ignore").strip()
            if not raw_line:
                continue
            if raw_line.startswith("#"):
                print(raw_line)
                continue
            parts = raw_line.split(",")
            if len(parts) != len(header):
                continue

            now_s = time.time()
            if csv_writer is not None:
                csv_writer.writerow([now_s, *parts])

            try:
                us_val = float(parts[us_idx])
            except ValueError:
                continue

            if t0_us is None:
                t0_us = us_val

            t_s = (us_val - t0_us) / 1_000_000.0
            tbuf.append(t_s)
            usbuf.append(us_val)

            for col in data_columns:
                idx = header.index(col)
                try:
                    val = float(parts[idx])
                except ValueError:
                    val = np.nan
                series[col].append(val)

            if now_s - last_draw < 0.10:
                continue
            last_draw = now_s

            t = np.asarray(tbuf, dtype=float)
            us_arr = np.asarray(usbuf, dtype=float)
            if len(t) < 10:
                continue

            fs = estimate_fs_us(us_arr)
            if fs is None:
                continue

            keep = t >= (t[-1] - args.window_sec)
            t_win = t[keep]

            selected_col = args.preview_column
            best_score = -1.0
            best_y = None
            best_yf = None
            best_bpm = np.nan
            best_sat_frac = np.nan
            best_raw_range = np.nan

            columns_to_consider = [selected_col] if selected_col else data_columns
            for col in columns_to_consider:
                y = np.asarray(series[col], dtype=float)
                if len(y) != len(t):
                    continue
                y_win = y[keep]
                finite = np.isfinite(y_win)
                if np.sum(finite) < 30:
                    continue
                y_eval = y_win[finite]
                t_eval = t_win[finite]
                score, yf, bpm, sat_frac, raw_range = score_column(y_eval, fs, args.saturation_threshold)
                if args.preview_column is not None:
                    selected_col = args.preview_column
                    best_score = score
                    best_y = (t_eval, y_eval)
                    best_yf = (t_eval, yf) if yf is not None else None
                    best_bpm = bpm
                    best_sat_frac = sat_frac
                    best_raw_range = raw_range
                    break
                if score > best_score:
                    selected_col = col
                    best_score = score
                    best_y = (t_eval, y_eval)
                    best_yf = (t_eval, yf) if yf is not None else None
                    best_bpm = bpm
                    best_sat_frac = sat_frac
                    best_raw_range = raw_range

            if best_y is None or selected_col is None:
                continue

            raw_ln.set_data(best_y[0], best_y[1])
            if best_yf is not None:
                flt_ln.set_data(best_yf[0], best_yf[1])
            else:
                flt_ln.set_data([], [])

            ax1.relim()
            ax1.autoscale_view()
            ax2.relim()
            ax2.autoscale_view()
            status_txt.set_text(
                f"selected={selected_col} | fs={fs:.2f} Hz | bpm_fft={best_bpm:.1f} | "
                f"score={best_score:.2f} | sat={100.0 * best_sat_frac:.1f}% | range={best_raw_range:.1f}"
            )
            fig.canvas.draw_idle()
            plt.pause(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        if csv_file is not None:
            csv_file.close()
        plt.close(fig)
        ser.close()


if __name__ == "__main__":
    main()
