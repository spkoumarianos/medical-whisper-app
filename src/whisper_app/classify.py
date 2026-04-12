from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import os
import requests

from .io_utils import ensure_dir, read_json, write_json
from .paths import DEFAULT_TRANSCRIPTS_DIR


@dataclass
class ClassificationResult:
    model_name: str
    base_url: str
    input_segments_json: Path
    output_classified_json: Path
    output_classified_csv: Optional[Path]
    segment_count: int


def check_ollama_server(base_url: str = "http://localhost:11434") -> Dict[str, Any]:
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        models = [m.get("name") for m in data.get("models", [])]
        return {"ok": True, "available_models": models}
    except Exception as e:
        return {"ok": False, "error": str(e), "available_models": []}


def _build_classify_prompt() -> str:
    return (
        "You will receive JSON conversation segments with start,end,text. "
        "Label each as doctor or patient. Return JSON with key 'classified_segments' "
        "containing objects with start,end,text,speaker."
    )


def _ollama_json(prompt: str, payload: Dict[str, Any], model: str, base_url: str) -> Dict[str, Any]:
    body = {
        "model": model,
        "prompt": f"{prompt}\n\nINPUT:\n" + json.dumps(payload, ensure_ascii=False),
        "stream": False,
        "format": "json",
    }
    resp = requests.post(f"{base_url}/api/generate", json=body, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    raw_text = data.get("response") or "{}"
    try:
        return json.loads(raw_text)
    except Exception:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw_text[start : end + 1])
        raise


def classify_segments(
    segments_json: Path,
    model: str = os.getenv("OLLAMA_MODEL", "mistral:7b"),
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    output_dir: Path = DEFAULT_TRANSCRIPTS_DIR,
) -> ClassificationResult:
    segments_json = Path(segments_json)
    assert segments_json.exists(), f"Segments JSON not found: {segments_json}"
    ensure_dir(output_dir)

    src = read_json(segments_json)
    segments = src.get("segments") or []

    prompt = _build_classify_prompt()
    payload = {"segments": segments}
    parsed = _ollama_json(prompt, payload, model=model, base_url=base_url)

    out_segments = parsed.get("classified_segments") or []
    out = {
        "audio_path": src.get("audio_path"),
        "model_name": model,
        "classifier_backend": "ollama",
        "segments": out_segments,
    }

    classified_json = output_dir / "classified_segments.json"
    write_json(classified_json, out)

    csv_path: Optional[Path] = None
    try:
        import pandas as pd  # type: ignore
        rows = [
            {
                "start": s.get("start"),
                "end": s.get("end"),
                "speaker": s.get("speaker"),
                "text": s.get("text"),
            }
            for s in out_segments
        ]
        df = pd.DataFrame(rows)
        csv_path = output_dir / "classified_segments.csv"
        df.to_csv(csv_path, index=False)
    except Exception:
        csv_path = None

    return ClassificationResult(
        model_name=model,
        base_url=base_url,
        input_segments_json=segments_json,
        output_classified_json=classified_json,
        output_classified_csv=csv_path,
        segment_count=len(out_segments),
    )
