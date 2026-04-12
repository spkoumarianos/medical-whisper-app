from __future__ import annotations
from pathlib import Path
from typing import Optional
import os, json
import typer
from rich import print

from .transcribe import transcribe_audio
from .classify import classify_segments, check_ollama_server
from .notes import generate_soap_note
from .paths import DEFAULT_TRANSCRIPTS_DIR, DEFAULT_NOTES_DIR

app = typer.Typer(add_completion=False, help='Whisper pipeline: transcribe -> classify -> SOAP')

@app.command()
def transcribe(
    audio_path: Path = typer.Argument(..., exists=True, readable=True, help='Path to audio file'),
    model: str = typer.Option('medium.en', help='Whisper model name'),
    device: Optional[str] = typer.Option(None, help='Force device: cpu or cuda'),
    outdir: Path = typer.Option(DEFAULT_TRANSCRIPTS_DIR, help='Output directory'),
):
    """Transcribe audio to text + segments JSON."""
    result = transcribe_audio(audio_path=audio_path, model_name=model, device=device, output_dir=outdir)
    print({
        'transcript_txt': str(result.transcript_txt),
        'segments_json': str(result.segments_json),
        'language': result.language,
        'segment_count': result.segment_count,
    })

@app.command()
def classify(
    segments_json: Path = typer.Argument(..., exists=True, help='Segments JSON from transcribe'),
    model: str = typer.Option(os.getenv('OLLAMA_MODEL', 'mistral:7b'), help='Ollama model name'),
    base_url: str = typer.Option(os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'), help='Ollama base URL'),
    outdir: Path = typer.Option(DEFAULT_TRANSCRIPTS_DIR, help='Output directory for classification'),
):
    """Classify segments with speaker labels using Ollama."""
    status = check_ollama_server(base_url)
    if not status.get('ok'):
        print('[red]Ollama not reachable at {}: {}[/red]'.format(base_url, status.get('error')))
    
        raise typer.Exit(2)
    result = classify_segments(segments_json=segments_json, model=model, base_url=base_url, output_dir=outdir)
    print({
        'classified_json': str(result.output_classified_json),
        'classified_csv': str(result.output_classified_csv) if result.output_classified_csv else None,
        'count': result.segment_count,
    })

@app.command()
def soap(
    classified_json: Path = typer.Argument(..., exists=True, help='Classified segments JSON'),
    model: str = typer.Option(os.getenv('OLLAMA_MODEL', 'mistral:7b'), help='Ollama model name'),
    base_url: str = typer.Option(os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'), help='Ollama base URL'),
    outdir: Path = typer.Option(DEFAULT_NOTES_DIR, help='Output directory for SOAP notes'),
):
    """Generate a SOAP note from classified segments using Ollama."""
    status = check_ollama_server(base_url)
    if not status.get('ok'):
        print('[red]Ollama not reachable at {}: {}[/red]'.format(base_url, status.get('error')))
        raise typer.Exit(2)
    result = generate_soap_note(classified_json=classified_json, model=model, base_url=base_url, output_dir=outdir)
    print({
        'soap_json': str(result.soap_json),
        'soap_txt': str(result.soap_txt),
        'raw_response': str(result.raw_response_json),
    })

@app.command()
def run(
    audio_path: Path = typer.Argument(..., exists=True, readable=True, help='Path to audio file'),
    whisper_model: str = typer.Option('medium.en', help='Whisper model name'),
    device: Optional[str] = typer.Option(None, help='Force device: cpu or cuda'),
    ollama_model: str = typer.Option(os.getenv('OLLAMA_MODEL', 'mistral:7b'), help='Ollama model name'),
    base_url: str = typer.Option(os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'), help='Ollama base URL'),
):
    """Full pipeline: transcribe -> classify -> SOAP."""
    t = transcribe_audio(audio_path=audio_path, model_name=whisper_model, device=device, output_dir=DEFAULT_TRANSCRIPTS_DIR)
    status = check_ollama_server(base_url)
    if not status.get('ok'):
        print('[red]Ollama not reachable at {}: {}[/red]'.format(base_url, status.get('error')))
        raise typer.Exit(2)
    c = classify_segments(segments_json=t.segments_json, model=ollama_model, base_url=base_url, output_dir=DEFAULT_TRANSCRIPTS_DIR)
    n = generate_soap_note(classified_json=c.output_classified_json, model=ollama_model, base_url=base_url, output_dir=DEFAULT_NOTES_DIR)
    print(json.dumps({
        'transcript_txt': str(t.transcript_txt),
        'segments_json': str(t.segments_json),
        'classified_json': str(c.output_classified_json),
        'classified_csv': str(c.output_classified_csv) if c.output_classified_csv else None,
        'soap_json': str(n.soap_json),
        'soap_txt': str(n.soap_txt),
        'raw_response': str(n.raw_response_json),
    }, indent=2))
