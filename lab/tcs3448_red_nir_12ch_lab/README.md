# TCS3448 Red/NIR 12-Channel Lab

This PlatformIO project is used to test the TCS3448 channels that are most relevant for SpO2-style and hemoglobin-related exploratory features. It is separate from the main firmware entry point so that experimental settings can be changed without disturbing the stable acquisition path.

## Purpose

The first target is to obtain a clear pulsatile signal from:

- `NIR`, approximately 855 nm;
- `F6`, approximately 636 nm and used as the red-like channel.

This firmware uses `auto_smux=2`, which provides 12 channels. The mode includes the stable first bank and the bank containing `F6`.

## Illumination Mode

The firmware keeps the LEDs on continuously during acquisition. This simplifies the timing and improves the sampling rate while validating whether the selected spectral channels contain a clear pulse component.

This mode is useful for early testing, but it also means that the recorded signal can include ambient-light contribution. Cover the sensor with the finger as consistently as possible during testing.

## CSV Columns

The CSV header is:

```text
ms,us,astatus,FZ,FY,FXL,NIR,VIS2_C1,FD_C1,F2,F3,F4,F6,VIS2_C2,FD_C2
```

The most important channels for this experiment are:

- `NIR`: near-infrared channel, approximately 855 nm;
- `F6`: red-like channel, approximately 636 nm;
- `FXL`: amber/orange channel, approximately 596 nm;
- `FY`: green-yellow channel, approximately 560 nm.

## Firmware Upload

From this folder:

```powershell
platformio run --target upload --environment esp32dev
```

If the ESP32 does not enter download mode automatically, hold `BOOT`, start the upload, press `EN/RST` briefly, and release `BOOT` when the upload begins.

## Live Plot With Automatic Channel Selection

```powershell
python tools\live_plot_12ch.py --port COM5
```

The plot title reports:

- `sat`: percentage of samples close to ADC saturation;
- `range`: approximate difference between the 95th and 5th percentiles.

For `F6`, `sat` should remain close to `0%`. If saturation is high, reduce `LED_DRIVE` or `ALS_AGAIN`.

## Live Plot With a Fixed Channel

```powershell
python tools\live_plot_12ch.py --port COM5 --preview-column NIR
python tools\live_plot_12ch.py --port COM5 --preview-column F6
python tools\live_plot_12ch.py --port COM5 --preview-column FXL
```

## Save a CSV Recording

```powershell
python tools\live_plot_12ch.py --port COM5 --csv-out logs\red_nir_12ch_session.csv
```

The `logs/` folder is intended for local recordings and is ignored by Git.

## Analyze F6/NIR Features

After saving approximately 60-100 seconds:

```powershell
python tools\analyze_red_nir_12ch.py --input logs\red_nir_12ch_session.csv
```

The outputs are written to:

```text
logs\analysis_red_nir_12ch\red_nir_12ch_summary.json
logs\analysis_red_nir_12ch\red_nir_12ch_beats.csv
```

The main ratio feature is:

```text
selected_ratio_of_ratios = (AC_F6/DC_F6) / (AC_NIR/DC_NIR)
```

To save one pulse-oximeter anchor for the current run:

```powershell
python tools\analyze_red_nir_12ch.py --input logs\red_nir_12ch_session.csv --spo2-reference-pct-for-this-run 98.5
```

To estimate an exploratory SpO2-style value from an existing anchor:

```powershell
python tools\analyze_red_nir_12ch.py --input logs\red_nir_12ch_session.csv --spo2-anchor-ratio <anchor_ratio> --spo2-anchor-pct 98.5 --spo2-slope 25
```

## Quick Interpretation

- `fs` should remain high enough for PPG analysis, ideally above 20 Hz.
- `bpm_fft` should be close to the expected heart rate.
- `score` should be higher for channels with a visible pulse component.

If both `F6` and `NIR` show clear pulse components, the next development step is to evaluate LED off/on/difference acquisition for ambient-light compensation.

## Troubleshooting

If the signal saturates or remains close to the ADC maximum:

- reduce `LED_DRIVE`;
- reduce `ALS_AGAIN`.

If the signal is too small:

- increase `LED_DRIVE`;
- increase `ALS_AGAIN`;
- increase `ALS_ATIME` or `ALS_ASTEP`.

If `NIR` works but `F6` does not show a clear pulse component:

- reduce finger pressure;
- cover the sensor more consistently from ambient light;
- test `FXL` and `FY` as fallback visible channels.
