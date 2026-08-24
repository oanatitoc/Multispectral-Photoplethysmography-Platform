# TCS3448 18-Channel Lab

This folder contains an experimental PlatformIO project for the TCS3448 `auto_smux=3` mode, which reads all 18 spectral channels. It is separate from the main firmware so that full-channel experiments can be performed without changing the stable acquisition path.

## Purpose

The 18-channel firmware:

- reads the complete TCS3448 channel set;
- captures LED-off and LED-on frames;
- writes one CSV row containing on, off, and difference values for all channels;
- discards frames after LED switching to reduce unstable or stale samples.

## Channel Order

The automatic 18-channel mode follows this order:

1. `FZ`
2. `FY`
3. `FXL`
4. `NIR`
5. `VIS2_C1`
6. `FD_C1`
7. `F2`
8. `F3`
9. `F4`
10. `F6`
11. `VIS2_C2`
12. `FD_C2`
13. `F1`
14. `F7`
15. `F8`
16. `F5`
17. `VIS2_C3`
18. `FD_C3`

## Firmware Upload

From this folder:

```powershell
platformio run --target upload --environment esp32dev
```

## Live Plot

```powershell
python tools\live_plot_18ch.py --port COM5 --field diff
```

To preview a fixed column:

```powershell
python tools\live_plot_18ch.py --port COM5 --preview-column F6_diff
python tools\live_plot_18ch.py --port COM5 --preview-column NIR_diff
python tools\live_plot_18ch.py --port COM5 --preview-column F7_diff
```

## Save a CSV Recording

```powershell
python tools\live_plot_18ch.py --port COM5 --field diff --csv-out logs\18ch_session.csv
```

The `logs/` folder is intended for local recordings and is ignored by Git.

## First Calibration Checks

1. Check whether `NIR_diff`, `F6_diff`, `F7_diff`, and `F8_diff` contain a clear pulse component.
2. Compare `diff` with the raw `on` values.
3. If `diff` remains close to zero, increase `DISCARD_FRAMES_AFTER_LED_TOGGLE` or `EXTRA_SETTLE_MS`.
4. If the signal saturates, reduce `LED_DRIVE` or `ALS_AGAIN`.
5. If the signal is too small, increase `LED_DRIVE`, `ALS_ATIME`, or `ALS_ASTEP`.

## Research Goal

This laboratory firmware is used to identify a robust configuration for:

- complete 18-channel multispectral access;
- automatic selection of the channel with the clearest pulse component;
- LED off/on/difference validation;
- selection of useful spectral pairs for SpO2-style features, tissue-oxygenation trends, hemoglobin-related exploratory features, and pigmentation-aware sensing.
