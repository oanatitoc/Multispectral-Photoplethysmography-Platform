# Dataset Organization

The software expects recordings to be organized as one folder per participant and one folder per acquisition run.

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

## Raw Samples

Raw samples are stored as CSV files in each run's `raw/` folder. The exact columns depend on the firmware mode, but the files always contain time columns followed by optical-channel values.

## Metadata

Each run can contain a `meta/run_metadata.json` file with acquisition settings, protocol information, selected channel, sampling information, and operator notes.

Subject-level metadata can be stored in `subject_metadata.json`. Public releases should not include identifying information or private participant metadata.

## Analysis Outputs

Offline analysis writes results under each run's `analysis/` folder. Typical outputs include beat tables, summary JSON files, respiratory-rate traces, perfusion-proxy tables, and validation snapshots.

## Public Data Policy

Raw pilot recordings are intentionally excluded from this repository because they contain human-subject physiological data and participant metadata. If data are shared publicly later, use anonymized or synthetic examples and remove direct identifiers, private notes, local paths, and any metadata not approved for release.

Derived validation tables can be shared only if they are compatible with the study consent and institutional constraints.
