# FastSpeech 2-Based Text-to-Speech Synthesis

Separate English (Svarah) and Telugu (IndicTTS Telugu) FastSpeech 2 implementations.

## Structure
- `English/` and `Telugu/`: source code, configurations, checkpoints, and logs.
- `Results/`: five final WAV samples per language, separate training/validation loss plots, and per-test duration/pitch/energy plots.
- `Report/`: technical report in PDF and, if supplied, DOCX format.

Pitch and energy analysis values are model-output units and are not calibrated physical Hz/energy measurements.

## Reproducibility
See each language folder's README, config.yaml, and requirements.txt. Ensure dataset and vocoder assets are available before running preprocessing, training, or inference.
