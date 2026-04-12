from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> Path:
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def write_text(path: Path, text: str) -> Path:
    ensure_dir(path.parent)
    path.write_text(text, encoding='utf-8')
    return path


def coerce_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    s = str(val).strip().lower()
    return s in {1, true, t, yes, y, on}
