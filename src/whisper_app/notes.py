from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import json
import os
import textwrap
import requests

from .io_utils import ensure_dir, read_json, write_json, write_text
from .paths import DEFAULT_NOTES_DIR


@dataclass
class SoapResult:
    model_name: str
    base_url: str
    input_classified_json: Path
    soap_json: Path
    soap_txt: Path
    raw_response_json: Path


def _soap_prompt() -> str:
    return textwrap.dedent(
        """
        You are a clinical documentation assistant. Given a JSON list of dialogue segments
        labeled with speaker (doctor or patient), produce a concise SOAP note as strict JSON:
        {
          "subjective": str,
          "objective": str,
          "assessment": str,
          "plan": str
        }
        Use professional medical tone, no PHI, and avoid hallucination. If information is missing, say so.
        """
    ).strip()


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


def generate_soap_note(
    classified_json: Path,
    model: str = os.getenv("OLLAMA_MODEL", "mistral:7b"),
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    output_dir: Path = DEFAULT_NOTES_DIR,
) -> SoapResult:
    classified_json = Path(classified_json)
    assert classified_json.exists(), f"Classified JSON not found: {classified_json}"
    ensure_dir(output_dir)

    data = read_json(classified_json)
    segments = data.get("segments") or []

    prompt = _soap_prompt()
    payload = {"segments": segments}
    parsed = _ollama_json(prompt, payload, model=model, base_url=base_url)

    soap = {
        "subjective": parsed.get("subjective", ""),
        "objective": parsed.get("objective", ""),
        "assessment": parsed.get("assessment", ""),
        "plan": parsed.get("plan", ""),
    }

    stem = classified_json.stem
    soap_json_path = output_dir / f"{stem}_soap_note.json"
    write_json(soap_json_path, soap)

    # Pretty TXT
    txt = (
        "SOAP Note\n\n"
        f"Subjective:\n{soap['subjective']}\n\n"
        f"Objective:\n{soap['objective']}\n\n"
        f"Assessment:\n{soap['assessment']}\n\n"
        f"Plan:\n{soap['plan']}\n"
    )
    soap_txt_path = output_dir / f"{stem}_soap_note.txt"
    write_text(soap_txt_path, txt)

    raw_path = output_dir / f"{stem}_soap_note_raw_ollama_response.json"
    write_json(raw_path, {"raw_ollama_response": parsed})

    return SoapResult(
        model_name=model,
        base_url=base_url,
        input_classified_json=classified_json,
        soap_json=soap_json_path,
        soap_txt=soap_txt_path,
        raw_response_json=raw_path,
    )
