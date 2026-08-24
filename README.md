# Multispectral Photoplethysmography Platform

This repository contains the firmware, desktop acquisition software, and offline analysis tools for a custom multispectral reflective photoplethysmography (PPG) platform. The system combines a TCS3448 multispectral optical front end, an ESP32-based acquisition board, a Python desktop graphical user interface (GUI), and a structured dataset pipeline.

The code is intended for research and engineering reproducibility. It is not a medical device and it is not intended for clinical diagnosis or patient monitoring.

## Repository Contents

```text
apps/
  acquire_record.py              Command-line acquisition utility
  analyze_run.py                 Offline analysis for one recording
  analyze_subject.py             Batch analysis for one subject folder
  build_cohort_report.py         Cohort-level summary tables
  build_validation_table.py      Validation-table builder
  calibrate_red_nir_12ch_cohort.py
  create_subject.py              Subject-folder creation helper
  import_legacy_data.py          Import helper for older recordings
  ppg_gui.py                     Desktop acquisition GUI
  train_cohort_models.py         Cohort model-comparison pipeline

src/
  main.cpp                       Main ESP32 firmware entry point
  firmware/                      Firmware support headers
  ppg_suite/                     Python processing package

lab/
  tcs3448_red_nir_12ch_lab/      12-channel firmware and live-plot tools
  tcs3448_18ch_lab/              18-channel experimental firmware and tools

hardware/
  Exported schematic, PCB views, bill of materials, and hardware renders

docs/
  Dataset and hardware documentation
```

Human-subject recordings, calibration tables derived from the pilot cohort, and generated manuscript figures are intentionally not included in this public release.

## Hardware Summary

The optical module uses the ams-OSRAM TCS3448 multispectral sensor with four white LEDs in a reflective PPG geometry. The sensor board connects to an ESP32-Sparrow Rev. 2 acquisition board through I2C and streams CSV-formatted samples to the desktop software over USB serial.

The `hardware/` folder contains exported design documentation that supports reproduction of the prototype:

- optical PCB schematic;
- PCB front and bottom layout views;
- bill of materials;
- annotated PCB/enclosure renders.

The ESP32-Sparrow Rev. 2 controller board is an open-hardware design available at: <https://github.com/dantudose/ESP32-Sparrow-rev2>.

## Software Requirements

Python 3.11 is recommended.

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The ESP32 firmware is built with PlatformIO.

## Firmware Upload

From the repository root:

```powershell
platformio run --target upload --environment esp32dev_red_nir_12ch
```

Other available environments are defined in `platformio.ini`, including dim, bright, and 18-channel experimental configurations.

## Desktop Acquisition GUI

Start the GUI with:

```bash
python apps/ppg_gui.py
```

The GUI supports serial-port connection, subject and protocol selection, live waveform preview, live metric display, event marking, and timestamped reference-snapshot entry.

## Command-Line Acquisition

For a new recording without the GUI:

```bash
python apps/acquire_record.py --subject-id subject_0001 --port COM5 --duration 180 --preview-channel NIR
```

This creates a structured run folder containing raw samples and metadata.

## Dataset Layout

The processing tools expect recordings to follow this structure:

```text
dataset/
  subject_0001/
    subject_metadata.json
    run_0001_YYYY-MM-DD_HH-MM-SS/
      raw/
        tcs3448_raw.csv
      meta/
        run_metadata.json
      analysis/
```

Private participant data should remain outside the public repository. A synthetic or anonymized example can be added later if it is approved for public release.

## Offline Analysis

Analyze one run:

```bash
python apps/analyze_run.py --input "dataset/subject_0001/run_0001_YYYY-MM-DD_HH-MM-SS/raw/tcs3448_raw.csv"
```

Analyze all runs from one subject:

```bash
python apps/analyze_subject.py --subject-id subject_0001
```

Build validation tables:

```bash
python apps/build_validation_table.py --dataset-dir dataset --subject-id subject_0001
```

## Main Processing Modules

Implemented modules include:

- heart-rate and beat detection;
- heart-rate variability features;
- respiratory-rate tracking from baseline, amplitude, and inter-beat interval modulation;
- perfusion-index proxy estimation;
- SpO2-style red/NIR ratio-of-ratios features;
- tissue-oxygenation trend features;
- perfusion-response metrics;
- vasomotion-like slow modulation features;
- waveform morphology and exploratory stiffness descriptors.

## Reproducibility Notes

The public release contains the source code and hardware documentation needed to inspect and reproduce the acquisition and analysis pipeline. Raw pilot recordings are excluded because they contain human-subject physiological data and participant metadata.

For manuscript use, this repository should be cited as the firmware and software availability resource, while the dataset availability statement should describe any anonymized derived tables or processed examples that can be shared under consent and institutional constraints.
