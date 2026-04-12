from __future__ import annotations

from pathlib import Path

# Project default directories
CWD = Path.cwd()
DEFAULT_TRANSCRIPTS_DIR = CWD / 'transcription_outputs'
DEFAULT_NOTES_DIR = CWD / 'soap_note_outputs'

__all__ = [
    'CWD',
    'DEFAULT_TRANSCRIPTS_DIR',
    'DEFAULT_NOTES_DIR',
]
