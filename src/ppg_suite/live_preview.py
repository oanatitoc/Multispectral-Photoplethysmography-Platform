from __future__ import annotations

from collections import deque
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from .signal import butter_bandpass


class LivePreview:
    def __init__(
        self,
        header: list[str],
        channel: str = "NIR_diff",
        max_samples: int = 700,
        heart_band: tuple[float, float] = (0.7, 4.0),
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.header = header
        self.channel = channel
        self.max_samples = max_samples
        self.heart_band = heart_band

        if not enabled:
            return

        if channel not in header:
            raise ValueError(f"Preview channel not found in header: {channel}")

        self.channel_idx = header.index(channel)
        self.us_idx = header.index("us") if "us" in header else None
        self.ms_idx = header.index("ms") if "ms" in header else None

        self.usbuf = deque(maxlen=max_samples)
        self.tbuf = deque(maxlen=max_samples)
        self.ybuf = deque(maxlen=max_samples)
        self.t0_us: Optional[float] = None
        self.t0_ms: Optional[float] = None
        self.last_draw = 0.0

        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        self.raw_ln, = self.ax1.plot([], [], label=f"{channel} raw")
        self.ax1.set_ylabel("raw ADC")
        self.ax1.legend(loc="upper right")
        self.ax1.set_title("PPG live preview")

        self.flt_ln, = self.ax2.plot([], [], label=f"{channel} filtered")
        self.pk_ln, = self.ax2.plot([], [], "o", label="detected peaks")
        self.ax2.set_xlabel("time (s)")
        self.ax2.set_ylabel("filtered")
        self.ax2.legend(loc="upper right")

        self.txt = self.ax1.text(
            0.02,
            0.90,
            "HR_peak: -- | HR_fft: -- | fs: -- | contact: --",
            transform=self.ax1.transAxes,
            fontsize=12,
        )
        plt.show(block=False)

    def close(self) -> None:
        if not self.enabled:
            return
        plt.close(self.fig)

    @staticmethod
    def estimate_fs_us(us_array: np.ndarray) -> Optional[float]:
        if len(us_array) < 5:
            return None
        dt = np.diff(us_array) / 1_000_000.0
        dt = dt[(dt > 0.005) & (dt < 0.2)]
        if len(dt) < 3:
            return None
        return float(1.0 / np.median(dt))

    @staticmethod
    def estimate_fs_t(t_array: np.ndarray) -> Optional[float]:
        if len(t_array) < 5:
            return None
        dt = np.diff(t_array)
        dt = dt[(dt > 0.005) & (dt < 0.2)]
        if len(dt) < 3:
            return None
        return float(1.0 / np.median(dt))

    @staticmethod
    def bpm_fft(x: np.ndarray, fs: Optional[float], band: tuple[float, float]) -> tuple[Optional[float], float]:
        if fs is None or len(x) < int(6 * fs):
            return None, 0.0
        x = np.asarray(x, float)
        x = x - np.mean(x)
        win = np.hanning(len(x))
        X = np.abs(np.fft.rfft(x * win))
        f = np.fft.rfftfreq(len(x), d=1.0 / fs)
        lo, hi = band
        mask = (f >= lo) & (f <= hi)
        if not np.any(mask):
            return None, 0.0
        fb = f[mask]
        Xb = X[mask]
        k = int(np.argmax(Xb))
        peak_f = float(fb[k])
        peak = float(Xb[k])
        noise = float(np.median(Xb) + 1e-9)
        q = peak / noise
        return peak_f * 60.0, q

    @staticmethod
    def bpm_from_peaks(t: np.ndarray, y: np.ndarray, fs: Optional[float]) -> tuple[Optional[float], np.ndarray]:
        if fs is None or len(y) < int(5 * fs):
            return None, np.array([], dtype=int)
        prominence = max(0.5, 0.35 * np.std(y))
        min_distance = max(1, int(0.35 * fs))
        peaks, _ = find_peaks(y, distance=min_distance, prominence=prominence)
        if len(peaks) < 2:
            return None, peaks
        ibi = np.diff(t[peaks])
        valid = (ibi >= 60.0 / 180.0) & (ibi <= 60.0 / 40.0)
        ibi_valid = ibi[valid]
        if len(ibi_valid) < 2:
            return None, peaks
        bpm = 60.0 / np.median(ibi_valid)
        return float(bpm), peaks

    @staticmethod
    def contact_good(raw_window: np.ndarray) -> bool:
        raw_window = np.asarray(raw_window, float)
        if len(raw_window) < 20:
            return False
        p1 = np.percentile(raw_window, 1)
        p99 = np.percentile(raw_window, 99)
        if p1 < 100 or p99 > 3000:
            return False
        return True

    def _compute_time_s(self, parts: list[str]) -> Optional[float]:
        try:
            if self.us_idx is not None:
                us = float(parts[self.us_idx])
                if self.t0_us is None:
                    self.t0_us = us
                self.usbuf.append(us)
                return (us - self.t0_us) / 1_000_000.0
            if self.ms_idx is not None:
                ms = float(parts[self.ms_idx])
                if self.t0_ms is None:
                    self.t0_ms = ms
                return (ms - self.t0_ms) / 1000.0
        except Exception:
            return None
        return None

    def update(self, parts: list[str], now_s: float) -> None:
        if not self.enabled:
            return
        try:
            val = float(parts[self.channel_idx])
        except Exception:
            return

        t_s = self._compute_time_s(parts)
        if t_s is None:
            return

        self.tbuf.append(t_s)
        self.ybuf.append(val)

        if now_s - self.last_draw < 0.05:
            return
        self.last_draw = now_s

        t = np.asarray(self.tbuf, dtype=float)
        y = np.asarray(self.ybuf, dtype=float)
        us_arr = np.asarray(self.usbuf, dtype=float)
        fs = self.estimate_fs_us(us_arr) if len(us_arr) >= 5 else self.estimate_fs_t(t)
        ok = self.contact_good(y)

        self.raw_ln.set_data(t, y)

        hr_peak = None
        hr_fft = None
        q_fft = 0.0
        peak_x = np.array([])
        peak_y = np.array([])

        if ok:
            yf = butter_bandpass(y, fs, self.heart_band[0], self.heart_band[1], order=2)
            if yf is not None:
                self.flt_ln.set_data(t, yf)
                hr_peak, peaks = self.bpm_from_peaks(t, yf, fs)
                if len(peaks) > 0:
                    peak_x = t[peaks]
                    peak_y = yf[peaks]
                hr_fft, q_fft = self.bpm_fft(yf, fs, self.heart_band)
            else:
                self.flt_ln.set_data([], [])
        else:
            self.flt_ln.set_data([], [])

        self.pk_ln.set_data(peak_x, peak_y)
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()

        peak_txt = "--" if hr_peak is None else f"{hr_peak:.1f}"
        fft_txt = "--" if hr_fft is None else f"{hr_fft:.1f}"
        fs_txt = "--" if fs is None else f"{fs:.1f}"
        contact_txt = "good" if ok else "bad"

        self.txt.set_text(
            f"HR_peak: {peak_txt} bpm | HR_fft: {fft_txt} bpm | fs: {fs_txt} Hz | q={q_fft:.1f} | contact: {contact_txt}"
        )
        self.fig.canvas.draw_idle()
        plt.pause(0.001)