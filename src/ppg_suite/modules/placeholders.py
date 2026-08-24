from __future__ import annotations


def placeholder_modules() -> dict:
    return {
        "arterial_stiffness_waveform_morphology": {
            "status": "placeholder",
            "next_step": "Add rise time, crest time, pulse width, derivative features, and surrogate stiffness indices.",
        },
        "skin_tone_aware_compensation": {
            "status": "placeholder",
            "next_step": "Needs subject metadata and multispectral normalization/adaptive weighting experiments.",
        },
    }
