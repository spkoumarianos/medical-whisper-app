from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import whisper  # type: ignore

from .io_utils import ensure_dir, write_json, write_text
from .paths import DEFAULT_TRANSCRIPTS_DIR


@dataclass
class TranscribeResult:
    model_name: str
    device: str
    audio_path: Path
    transcript_txt: Path
    segments_json: Path
    language: Optional[str]
    segment_count: int


def _resolve_device(device: str | None = None) -> str:
    if device in {"cuda", "cpu"}:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(name: str = "medium.en", device: str | None = None):
    device_name = _resolve_device(device)
    if device_name == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    return whisper.load_model(name, device=device_name)


def transcribe_audio(
    audio_path: Path,
    model_name: str = "medium.en",
    device: str | None = None,
    output_dir: Path = DEFAULT_TRANSCRIPTS_DIR,
) -> TranscribeResult:
    audio_path = Path(audio_path)
    assert audio_path.exists(), f"Audio file not found: {audio_path}"

    ensure_dir(output_dir)

    model = load_model(model_name, device=device)
    result: Dict[str, Any] = model.transcribe(str(audio_path))

    # Save plain text transcript
    full_text = (result.get("text") or "").strip()
    txt_path = output_dir / f"{audio_path.stem}_transcript.txt"
    write_text(txt_path, full_text)

    # Build simplified segments and save JSON
    segments_out: List[Dict[str, Any]] = []
    for seg in result.get("segments", []) or []:
        segments_out.append(
            {
                "start": seg.get("start"),
                "end": seg.get("end"),
                "text": (seg.get("text") or "").strip(),
                "source": "whisper",
            }
        )

    json_out = {
        "audio_path": str(audio_path),
        "language": result.get("language"),
        "model_name": model_name,
        "engine": "openai-whisper",
        "segments": segments_out,
    }

    segments_path = output_dir / f"{audio_path.stem}_segments.json"
    write_json(segments_path, json_out)

    return TranscribeResult(
        model_name=model_name,
        device=_resolve_device(device),
        audio_path=audio_path,
        transcript_txt=txt_path,
        segments_json=segments_path,
        language=result.get("language"),
        segment_count=len(segments_out),
    )
