# whisper-app

End-to-end pipeline to:
- Transcribe audio with OpenAI Whisper
- Classify speaker segments via Ollama (default: mistral:7b)
- Generate a SOAP note from the conversation

Outputs are saved under `transcription_outputs/` and `soap_note_outputs/`.

## Install

This project uses `uv` with a `pyproject.toml`.

Install in editable mode (recommended for CLI):

```
uv pip install -e .
```

- Create `.env` (optional):
  - `OLLAMA_BASE_URL=http://localhost:11434`
  - `OLLAMA_MODEL=mistral:7b`

## CLI

- Transcribe:
  - `whisper-app transcribe doctor_patient_example2.wav`
- Classify:
  - `whisper-app classify transcription_outputs/doctor_patient_example2_segments.json`
- SOAP note:
  - `whisper-app soap transcription_outputs/classified_segments.json`
- Full run:
  - `whisper-app run doctor_patient_example2.wav`

If you prefer the executable name, install the project in editable mode and use `whisper-app` directly.

## Notebooks

Original notebooks were moved to `notebooks/` and remain runnable:
- `notebooks/Step1_whisper_pipeline.ipynb`
- `notebooks/Step2_participant_classification.ipynb`
- `notebooks/Step3_Chart_Generation.ipynb`

## Notes
- Default Whisper model: `medium.en` (auto-selects `cuda` if available).
- Default Ollama model: `mistral:7b` (override with `--model` or env).
- Outputs will not overwrite existing files unless the same names are used.
