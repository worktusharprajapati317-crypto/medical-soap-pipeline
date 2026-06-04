"""
Medical Transcription & SOAP Note Pipeline — standalone script.

End-to-end flow:
  1. ASR  : HuggingFace ``transformers`` Whisper pipeline (chunked long-form audio).
  2. SOAP : Anthropic Claude ``tool_use`` (primary) or Ollama prompt (fallback).
  3. Val  : Subjective→Objective leakage check (regex + sentence embeddings).

Outputs are written under ``outputs/`` (transcript, SOAP JSON/Markdown, validation report).
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Optional, TypedDict

import numpy as np
import nltk
import requests
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from transformers import pipeline as hf_pipeline

# Download NLTK sentence tokenizer data required by ``sent_tokenize``.
nltk.download("punkt_tab", quiet=True)


# ── Type aliases (structured dict shapes used across the pipeline) ─────────────

class TranscriptSegment(TypedDict):
    """One timed chunk from the ASR pipeline."""

    id: int
    start: float
    end: float
    text: str


class TranscriptionResult(TypedDict):
    """Normalized transcription payload returned by ``transcribe_audio``."""

    full_text: str
    segments: list[TranscriptSegment]
    language: str
    duration_s: float


# SOAP note keys: subjective, objective, assessment, plan (all string values).
SoapNote = dict[str, str]

# Single validation flag (linguistic or semantic layer).
LeakageFlag = dict[str, Any]

# Full validation report written to ``validation_report.json``.
ValidationReport = dict[str, Any]

# HuggingFace ASR pipeline instance (dynamic transformers type).
ASRPipeline = Any


# ── Paths ──────────────────────────────────────────────────────────────────────
# Project root = directory containing this script.
BASE_DIR: Final[Path] = Path(__file__).parent
# Default input audio for the assignment sample dictation.
AUDIO_FILE: Final[Path] = BASE_DIR / "sample_dictation.mp3"
# All generated artifacts go here.
OUTPUT_DIR: Final[Path] = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TRANSCRIPT_FILE: Final[Path] = OUTPUT_DIR / "raw_transcript.txt"
SOAP_JSON_FILE: Final[Path] = OUTPUT_DIR / "soap_note.json"
SOAP_MD_FILE: Final[Path] = OUTPUT_DIR / "soap_note.md"
VALIDATION_FILE: Final[Path] = OUTPUT_DIR / "validation_report.json"

# ── ASR config ─────────────────────────────────────────────────────────────────
# Override with env var WHISPER_MODEL, e.g. openai/whisper-small.
WHISPER_HF_MODEL: str = os.environ.get("WHISPER_MODEL", "openai/whisper-base")
WHISPER_LANGUAGE: Final[str] = "en"

# ── LLM config ─────────────────────────────────────────────────────────────────
ANTHROPIC_MODEL: Final[str] = "claude-sonnet-4-6"
OLLAMA_BASE_URL: Final[str] = "http://localhost:11434"
OLLAMA_MODEL: Final[str] = "gemma2:2b"
OLLAMA_TIMEOUT: Final[int] = 300

# ── Validation config ──────────────────────────────────────────────────────────
# Cosine similarity above this between O and S sentences triggers a semantic flag.
SEMANTIC_LEAKAGE_THRESHOLD: Final[float] = 0.80

# ── Anthropic tool schema ──────────────────────────────────────────────────────
# Forces structured SOAP fields via Claude tool_use / function calling.
SOAP_TOOL: dict[str, Any] = {
    "name": "create_soap_note",
    "description": (
        "Create a structured SOAP note from a physician-patient encounter transcript. "
        "Carefully separate patient-reported information (Subjective) from clinician-observed "
        "findings (Objective). Do NOT repeat the same fact in two sections."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "subjective": {
                "type": "string",
                "description": (
                    "Patient-reported information ONLY: chief complaint, symptom onset/duration, "
                    "pain quality and scale (e.g. 4/10), aggravating/relieving factors, past medical "
                    "and surgical history as stated by patient, current medications, allergies, "
                    "family history, social history (occupation, smoking, alcohol, travel). "
                    "Use: 'Patient reports', 'Patient states', 'Patient denies'."
                ),
            },
            "objective": {
                "type": "string",
                "description": (
                    "Clinician-performed observations and measurements ONLY: palpation findings, "
                    "range-of-motion and provocative test results performed by the clinician, "
                    "inspection findings, vital signs, lab and imaging results. "
                    "Do NOT include any patient self-reports."
                ),
            },
            "assessment": {
                "type": "string",
                "description": "Clinical diagnosis or impression. One to three sentences.",
            },
            "plan": {
                "type": "string",
                "description": (
                    "Treatment plan: medications with doses and duration, activity modifications, "
                    "referrals, imaging orders, follow-up timeline, patient education."
                ),
            },
        },
        "required": ["subjective", "objective", "assessment", "plan"],
    },
}

# System instructions for Anthropic (prepended to user message).
SOAP_SYSTEM_PROMPT: Final[str] = (
    "You are a board-certified clinical documentation specialist. "
    "Extract a structured SOAP note from the medical encounter transcript. "
    "The transcript is a real doctor-patient conversation — extract clinical "
    "information from BOTH speakers. "
    "Rules: (1) Patient-reported symptoms -> Subjective ONLY. "
    "(2) Clinician exam findings -> Objective ONLY. "
    "(3) No duplication across sections."
)

# Ollama fallback prompt; ``{{`` / ``}}`` escape JSON braces for ``str.format``.
SOAP_OLLAMA_PROMPT: Final[str] = (
    "You are a clinical documentation specialist. "
    "Convert the transcript below into a structured SOAP note.\n\n"
    "DEFINITIONS:\n"
    "  S - SUBJECTIVE: Patient-reported symptoms, history, pain scale, medications.\n"
    "  O - OBJECTIVE: Clinician exam findings, palpation, ROM tests, vitals, imaging.\n"
    "  A - ASSESSMENT: Clinical diagnosis or impression.\n"
    "  P - PLAN: Treatments, prescriptions, referrals, follow-up.\n\n"
    "RULES: No duplication. Output valid JSON only. Each value is a plain string.\n\n"
    "EXAMPLE:\n"
    '{{"subjective": "Patient reports sharp right knee pain 7/10 for 2 days. Ibuprofen minimal relief.",\n'
    ' "objective": "Tenderness on medial joint line. Mild swelling. Limited ROM. X-ray: no fracture.",\n'
    ' "assessment": "Suspected medial meniscus tear.",\n'
    ' "plan": "Rest, ice, PT x 4 weeks. MRI if no improvement."}}\n\n'
    "Transcript:\n\"\"\"\n{transcript}\n\"\"\"\n\n"
    "Output (JSON only):"
)

# (regex pattern, severity) pairs scanned in the Objective section.
LINGUISTIC_PATTERNS: Final[list[tuple[str, str]]] = [
    (r"patient\s+(reports|states|says|feels|complains|describes|denies|mentions)", "HIGH"),
    (r"(he|she|they)\s+(reports|states|says|feels|complains)", "HIGH"),
    (r"\d+\s*(out\s+of|/)\s*10", "HIGH"),
    (r"per\s+patient", "MEDIUM"),
    (r"rates?\s+(the\s+)?pain", "MEDIUM"),
    (r"(no\s+relief|minimal\s+relief|hasn.t\s+helped)", "LOW"),
    (r"(ibuprofen|tylenol|advil|aspirin)\s+(hasn.t|didn.t|not|hasn)", "LOW"),
]


# ── ASR helpers ────────────────────────────────────────────────────────────────

def load_asr_pipeline(model_name: str) -> ASRPipeline:
    """
    Load a HuggingFace automatic-speech-recognition pipeline for Whisper.

    Uses 30 s chunks with 5 s stride for long audio. ``device=-1`` means CPU.

    Args:
        model_name: HuggingFace hub model id (e.g. ``openai/whisper-base``).

    Returns:
        Configured ``transformers`` ASR pipeline callable on file paths or arrays.
    """
    # Log which hub model is being loaded and start a wall-clock timer.
    print(f"\n[Part A] Loading HuggingFace Whisper: '{model_name}' ...")
    t0 = time.time()
    # Build seq2seq Whisper pipeline with chunked inference for long dictations.
    asr = hf_pipeline(
        "automatic-speech-recognition",
        model=model_name,
        chunk_length_s=30,
        stride_length_s=5,
        device=-1,
    )
    print(f"  ✓ Ready in {time.time()-t0:.1f}s")
    return asr


def transcribe_audio(asr: ASRPipeline, audio_path: Path) -> TranscriptionResult:
    """
    Transcribe an audio file and normalize pipeline output into segments.

    Args:
        asr: Pipeline from ``load_asr_pipeline``.
        audio_path: Path to audio (e.g. MP3); decoded via system ``ffmpeg``.

    Returns:
        Full text, per-chunk segments with timestamps, language, and duration.
    """
    # Single-line progress prefix; flush so it appears before the long ASR call.
    print(f"  Transcribing {audio_path.name} ...", end="", flush=True)
    t0 = time.time()
    # Run ASR; timestamps enable chunk-level start/end times.
    result = asr(
        str(audio_path),
        return_timestamps=True,
        generate_kwargs={"language": WHISPER_LANGUAGE, "task": "transcribe"},
    )
    full_text = result["text"].strip()
    chunks = result.get("chunks", [])
    # Map HF chunk dicts to a stable segment list for saving/display.
    segments: list[TranscriptSegment] = [
        {
            "id": i,
            "start": round(float(c["timestamp"][0] or 0.0), 2),
            "end": round(float(c["timestamp"][1] or 0.0), 2),
            "text": c["text"].strip(),
        }
        for i, c in enumerate(chunks)
    ]
    duration = segments[-1]["end"] if segments else 0.0
    print(f" ✓ {time.time()-t0:.1f}s | {len(segments)} chunks | {len(full_text.split())} words")
    return {
        "full_text": full_text,
        "segments": segments,
        "language": WHISPER_LANGUAGE,
        "duration_s": round(duration, 2),
    }


def save_transcript(t: TranscriptionResult, path: Path) -> None:
    """
    Write human-readable transcript file with metadata and timed chunks.

    Args:
        t: Result from ``transcribe_audio``.
        path: Destination file (typically ``outputs/raw_transcript.txt``).
    """
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("MEDICAL DICTATION — RAW TRANSCRIPT\n")
        fh.write("=" * 60 + "\n")
        fh.write(f"Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write(f"ASR Model  : {WHISPER_HF_MODEL} (HuggingFace transformers)\n")
        fh.write(f"Language   : {t['language']}\n")
        fh.write(f"Duration   : {t['duration_s']}s\n")
        fh.write(f"Chunks     : {len(t['segments'])}\n")
        fh.write("=" * 60 + "\n\n")
        fh.write(t["full_text"])
        fh.write("\n\n")
        fh.write("=" * 60 + "\nTIMED CHUNKS\n" + "=" * 60 + "\n")
        for s in t["segments"]:
            fh.write(f"[{s['start']:>7.2f}s → {s['end']:>7.2f}s]  {s['text']}\n")
    print(f"  ✓ Saved → {path}")


# ── LLM helpers ────────────────────────────────────────────────────────────────

def check_anthropic_available() -> bool:
    """
    Return whether ``ANTHROPIC_API_KEY`` is set in the environment.

    Returns:
        True if the primary Claude backend can be used.
    """
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def check_ollama_available() -> bool:
    """
    Probe the local Ollama HTTP API (``/api/tags``).

    Returns:
        True if Ollama responds with HTTP 200 within 5 seconds.
    """
    try:
        return requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5).status_code == 200
    except Exception:
        return False


def generate_soap_anthropic(transcript: str) -> SoapNote:
    """
    Generate a SOAP note via Anthropic Messages API with forced ``tool_use``.

    The model must call ``create_soap_note``; ``block.input`` is the SOAP dict.

    Args:
        transcript: Full ASR text from the encounter.

    Returns:
        SOAP sections as string fields (subjective, objective, assessment, plan).

    Raises:
        RuntimeError: If the response contains no matching tool_use block.
    """
    import anthropic

    client = anthropic.Anthropic()
    user_content = (
        f"{SOAP_SYSTEM_PROMPT}\n\n"
        f"Transcript:\n\"\"\"\n{transcript}\n\"\"\""
    )
    print(f"\n[Part B] Generating SOAP via Anthropic ({ANTHROPIC_MODEL}) with tool_use ...")
    t0 = time.time()
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        tools=[SOAP_TOOL],
        tool_choice={"type": "tool", "name": "create_soap_note"},
        messages=[{"role": "user", "content": user_content}],
    )
    print(
        f"  ✓ Response in {time.time()-t0:.1f}s | "
        f"tokens in={response.usage.input_tokens} out={response.usage.output_tokens}"
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "create_soap_note":
            return block.input
    raise RuntimeError("No tool_use block in Anthropic response")


def extract_json_from_response(raw: str) -> SoapNote:
    """
    Parse SOAP JSON from a free-form LLM string (Ollama fallback).

    Strips markdown fences, tries full parse, then first ``{...}`` block,
    then a trailing-comma cleanup pass.

    Args:
        raw: Model text response.

    Returns:
        Parsed SOAP dict.

    Raises:
        ValueError: If JSON cannot be recovered.
    """
    cleaned = re.sub(r"```(?:json)?\s*", "", raw, flags=re.IGNORECASE).replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]+\}", cleaned)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(re.sub(r",\s*([}\]])", r"\1", cleaned))
    except json.JSONDecodeError:
        raise ValueError(f"Could not parse JSON:\n{raw[:400]}")


def generate_soap_ollama(transcript: str) -> SoapNote:
    """
    Generate SOAP JSON via local Ollama ``/api/generate`` (non-streaming).

    Args:
        transcript: Full ASR text.

    Returns:
        Parsed SOAP dict from model JSON output.
    """
    prompt = SOAP_OLLAMA_PROMPT.format(transcript=transcript)
    print(f"\n[Part B] Generating SOAP via Ollama ({OLLAMA_MODEL}) [fallback] ...")
    t0 = time.time()
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 1024},
        },
        timeout=OLLAMA_TIMEOUT,
    )
    resp.raise_for_status()
    raw = resp.json()["response"]
    print(f"  ✓ Response in {time.time()-t0:.1f}s")
    return extract_json_from_response(raw)


def save_soap_json(
    soap: SoapNote,
    transcript: str,
    llm: str,
    backend: str,
    path: Path,
) -> None:
    """
    Persist SOAP note, transcript, and generation metadata as JSON.

    Args:
        soap: SOAP section strings.
        transcript: Full ASR text included for traceability.
        llm: Model name used (Anthropic or Ollama).
        backend: ``anthropic_tool_use`` or ``ollama_prompt``.
        path: Output JSON path.
    """
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "asr_model": WHISPER_HF_MODEL,
            "asr_backend": "HuggingFace transformers pipeline",
            "llm_model": llm,
            "llm_backend": backend,
            "source_audio": AUDIO_FILE.name,
        },
        "transcript": transcript,
        "soap_note": soap,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved → {path}")


def save_soap_markdown(soap: SoapNote, llm: str, path: Path) -> None:
    """
    Write a Markdown rendering of the SOAP note for human review.

    Args:
        soap: SOAP section strings.
        llm: Model name for the footer line.
        path: Output ``.md`` path.
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# SOAP Note",
        f"_Generated: {ts} | ASR: {WHISPER_HF_MODEL} | LLM: {llm}_",
        "",
        "---",
        "",
        "## S — Subjective",
        soap.get("subjective", ""),
        "",
        "## O — Objective",
        soap.get("objective", ""),
        "",
        "## A — Assessment",
        soap.get("assessment", ""),
        "",
        "## P — Plan",
        soap.get("plan", ""),
        "",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"  ✓ Saved → {path}")


# ── Validation helpers ─────────────────────────────────────────────────────────

_semantic_encoder: Optional[SentenceTransformer] = None


def get_semantic_encoder() -> SentenceTransformer:
    """
    Lazy-load and cache the sentence-transformer used for semantic leakage.

    Returns:
        Shared ``all-MiniLM-L6-v2`` encoder instance.
    """
    global _semantic_encoder
    if _semantic_encoder is None:
        print("  Loading sentence-transformer (all-MiniLM-L6-v2) ...")
        _semantic_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _semantic_encoder


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Pairwise cosine similarity between rows of ``a`` and rows of ``b``.

    Args:
        a: Embedding matrix shape ``(n, dim)``.
        b: Embedding matrix shape ``(m, dim)``.

    Returns:
        Matrix shape ``(n, m)`` with values in approximately [-1, 1].
    """
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a_norm @ b_norm.T


def check_subjective_leakage(soap: SoapNote) -> ValidationReport:
    """
    Detect patient-reported (Subjective) content leaking into Objective.

    Layer 1 — regex ``LINGUISTIC_PATTERNS`` on the whole Objective section.
    Layer 2 — sentence embeddings: flag Objective lines similar to Subjective.

    Pass/fail: FAIL if any HIGH or MEDIUM flag exists (LOW alone still passes).

    Args:
        soap: Generated SOAP note (uses ``objective`` and ``subjective`` keys).

    Returns:
        Validation report dict with ``flags``, counts, and ``overall_status``.
    """
    # Pull S and O narrative text; default to empty string if key missing.
    obj_text = soap.get("objective", "")
    subj_text = soap.get("subjective", "")
    flags: list[LeakageFlag] = []

    # Layer 1: scan entire Objective for patient-report phrasing (regex).
    for pattern, severity in LINGUISTIC_PATTERNS:
        matches = re.findall(pattern, obj_text, flags=re.IGNORECASE)
        if matches:
            flags.append({
                "layer": "linguistic",
                "pattern": pattern,
                "matches": [m if isinstance(m, str) else str(m) for m in matches],
                "severity": severity,
                "message": f"Patient-report language pattern in Objective. Pattern: '{pattern}'",
            })

    # Layer 2: only if both sections have non-whitespace content.
    if obj_text.strip() and subj_text.strip():
        # Tokenize into sentences; drop very short fragments (noise).
        obj_sents = [s for s in sent_tokenize(obj_text) if len(s.split()) >= 4]
        subj_sents = [s for s in sent_tokenize(subj_text) if len(s.split()) >= 4]
        if obj_sents and subj_sents:
            encoder = get_semantic_encoder()
            # Embed each sentence list into R^384 (MiniLM output dim).
            obj_emb = encoder.encode(obj_sents, convert_to_numpy=True)
            subj_emb = encoder.encode(subj_sents, convert_to_numpy=True)
            # Rows = Objective sentences, cols = Subjective sentences.
            sim_matrix = cosine_similarity_matrix(obj_emb, subj_emb)
            for i, obj_sent in enumerate(obj_sents):
                max_sim = float(sim_matrix[i].max())
                best_j = int(sim_matrix[i].argmax())
                if max_sim >= SEMANTIC_LEAKAGE_THRESHOLD:
                    severity = "HIGH" if max_sim >= 0.90 else "MEDIUM"
                    flags.append({
                        "layer": "semantic",
                        "objective_sentence": obj_sent,
                        "similar_subjective": subj_sents[best_j],
                        "similarity_score": round(max_sim, 3),
                        "severity": severity,
                        "message": (
                            f"Objective sentence semantically similar to Subjective content "
                            f"(cosine {max_sim:.2f} >= {SEMANTIC_LEAKAGE_THRESHOLD})."
                        ),
                    })

    # Aggregate severities for summary and pass/fail rule.
    high = sum(1 for f in flags if f["severity"] == "HIGH")
    medium = sum(1 for f in flags if f["severity"] == "MEDIUM")
    low = sum(1 for f in flags if f["severity"] == "LOW")

    return {
        "validation_timestamp": datetime.now().isoformat(),
        "validation_method": "linguistic_patterns + semantic_similarity (all-MiniLM-L6-v2)",
        "semantic_threshold": SEMANTIC_LEAKAGE_THRESHOLD,
        "overall_status": "PASS" if (high == 0 and medium == 0) else "FAIL",
        "high_severity_count": high,
        "medium_severity_count": medium,
        "low_severity_count": low,
        "total_flags": len(flags),
        "flags": flags,
        "objective_section": obj_text,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Run the full pipeline: transcribe → SOAP → validate → print summary.

    Exits with code 1 if no LLM backend is available or SOAP sections are missing.
    """
    # Banner: pipeline title and configured models.
    print("=" * 65)
    print("MEDICAL TRANSCRIPTION & SOAP NOTE PIPELINE (v2)")
    print("=" * 65)
    print(f"ASR  : {WHISPER_HF_MODEL} (HuggingFace transformers)")
    print(f"LLM  : {ANTHROPIC_MODEL} primary | Ollama {OLLAMA_MODEL} fallback")

    # Part A — Transcription: load Whisper, run on sample audio, save txt.
    asr = load_asr_pipeline(WHISPER_HF_MODEL)
    transcription = transcribe_audio(asr, AUDIO_FILE)
    save_transcript(transcription, TRANSCRIPT_FILE)

    # Part B — SOAP generation: prefer Anthropic if API key present.
    anthropic_ok = check_anthropic_available()
    ollama_ok = check_ollama_available()
    print(f"\n  Anthropic API key: {anthropic_ok} | Ollama: {ollama_ok}")

    if anthropic_ok:
        # Structured output via tool_use (schema-enforced fields).
        soap_note = generate_soap_anthropic(transcription["full_text"])
        llm_used = ANTHROPIC_MODEL
        backend_used = "anthropic_tool_use"
    elif ollama_ok:
        # Local LLM; JSON parsed from free-form response.
        print("  Anthropic key not set — falling back to Ollama")
        soap_note = generate_soap_ollama(transcription["full_text"])
        llm_used = OLLAMA_MODEL
        backend_used = "ollama_prompt"
    else:
        # Neither backend reachable — cannot continue.
        print(
            "  ✗ No LLM available. Set ANTHROPIC_API_KEY or start Ollama.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Ensure all four SOAP keys exist before saving.
    missing = {"subjective", "objective", "assessment", "plan"} - set(soap_note)
    if missing:
        print(f"  ✗ SOAP note missing sections: {missing}", file=sys.stderr)
        sys.exit(1)

    save_soap_json(soap_note, transcription["full_text"], llm_used, backend_used, SOAP_JSON_FILE)
    save_soap_markdown(soap_note, llm_used, SOAP_MD_FILE)

    # Bonus — Validation: S/O leakage report to JSON + console summary.
    print("\n[Bonus] Running S/O leakage validation (semantic + linguistic) ...")
    report = check_subjective_leakage(soap_note)
    with open(VALIDATION_FILE, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    status = "✓ PASS" if report["overall_status"] == "PASS" else "✗ FAIL"
    print(
        f"  {status} — {report['total_flags']} flag(s) [{report['high_severity_count']} HIGH, "
        f"{report['medium_severity_count']} MEDIUM, {report['low_severity_count']} LOW]"
    )

    # Print SOAP note to stdout (wrapped for readability).
    print("\n" + "=" * 65)
    print("GENERATED SOAP NOTE")
    print("=" * 65)
    for key, label in [
        ("subjective", "S — Subjective"),
        ("objective", "O — Objective"),
        ("assessment", "A — Assessment"),
        ("plan", "P — Plan"),
    ]:
        print(f"\n{label}")
        print("-" * len(label))
        print(textwrap.fill(soap_note.get(key, ""), width=63))

    # File inventory: byte sizes of all output artifacts.
    print("\n" + "=" * 65)
    print("OUTPUT FILES")
    print("=" * 65)
    for p in [TRANSCRIPT_FILE, SOAP_JSON_FILE, SOAP_MD_FILE, VALIDATION_FILE]:
        size = p.stat().st_size if p.exists() else 0
        print(f"  {p.name:<35} {size:>8} bytes")

    # Final one-line run summary.
    print(f"\n  Pipeline: {WHISPER_HF_MODEL} | {llm_used} ({backend_used})")
    print(f"  Validation: {report['overall_status']} via {report['validation_method']}")


if __name__ == "__main__":
    main()
