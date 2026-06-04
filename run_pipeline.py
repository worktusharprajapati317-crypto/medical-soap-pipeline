"""
Standalone execution script - Medical Transcription & SOAP Note Pipeline.

Alternative approach vs run_pipeline.py:
  ASR  : HuggingFace transformers Whisper pipeline (not faster-whisper)
  SOAP : Anthropic Claude tool_use / function calling (primary)
         Ollama prompt engineering (fallback)
  Val  : Sentence-transformer semantic similarity + linguistic patterns
"""

import json
import os
import re
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

import requests
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer

nltk.download("punkt_tab", quiet=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
AUDIO_FILE      = BASE_DIR / "sample_dictation.mp3"
OUTPUT_DIR      = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TRANSCRIPT_FILE = OUTPUT_DIR / "raw_transcript.txt"
SOAP_JSON_FILE  = OUTPUT_DIR / "soap_note.json"
SOAP_MD_FILE    = OUTPUT_DIR / "soap_note.md"
VALIDATION_FILE = OUTPUT_DIR / "validation_report.json"

# ── ASR config ─────────────────────────────────────────────────────────────────
WHISPER_HF_MODEL = os.environ.get("WHISPER_MODEL", "openai/whisper-base")
WHISPER_LANGUAGE = "en"

# ── LLM config ─────────────────────────────────────────────────────────────────
ANTHROPIC_MODEL = "claude-sonnet-4-6"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "gemma2:2b"
OLLAMA_TIMEOUT  = 300

# ── Validation config ──────────────────────────────────────────────────────────
SEMANTIC_LEAKAGE_THRESHOLD = 0.80

# ── Anthropic tool schema ──────────────────────────────────────────────────────
SOAP_TOOL = {
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

SOAP_SYSTEM_PROMPT = (
    "You are a board-certified clinical documentation specialist. "
    "Extract a structured SOAP note from the medical encounter transcript. "
    "The transcript is a real doctor-patient conversation — extract clinical "
    "information from BOTH speakers. "
    "Rules: (1) Patient-reported symptoms -> Subjective ONLY. "
    "(2) Clinician exam findings -> Objective ONLY. "
    "(3) No duplication across sections."
)

SOAP_OLLAMA_PROMPT = (
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

LINGUISTIC_PATTERNS = [
    (r"patient\s+(reports|states|says|feels|complains|describes|denies|mentions)", "HIGH"),
    (r"(he|she|they)\s+(reports|states|says|feels|complains)", "HIGH"),
    (r"\d+\s*(out\s+of|/)\s*10", "HIGH"),
    (r"per\s+patient", "MEDIUM"),
    (r"rates?\s+(the\s+)?pain", "MEDIUM"),
    (r"(no\s+relief|minimal\s+relief|hasn.t\s+helped)", "LOW"),
    (r"(ibuprofen|tylenol|advil|aspirin)\s+(hasn.t|didn.t|not|hasn)", "LOW"),
]


# ── ASR helpers ────────────────────────────────────────────────────────────────

def load_asr_pipeline(model_name: str):
    print(f"\n[Part A] Loading HuggingFace Whisper: '{model_name}' ...")
    t0 = time.time()
    asr = hf_pipeline(
        "automatic-speech-recognition",
        model=model_name,
        chunk_length_s=30,
        stride_length_s=5,
        device=-1,
    )
    print(f"  ✓ Ready in {time.time()-t0:.1f}s")
    return asr


def transcribe_audio(asr, audio_path: Path) -> dict:
    print(f"  Transcribing {audio_path.name} ...", end="", flush=True)
    t0 = time.time()
    result = asr(
        str(audio_path),
        return_timestamps=True,
        generate_kwargs={"language": WHISPER_LANGUAGE, "task": "transcribe"},
    )
    full_text = result["text"].strip()
    chunks    = result.get("chunks", [])
    segments  = [
        {
            "id":    i,
            "start": round(float(c["timestamp"][0] or 0.0), 2),
            "end":   round(float(c["timestamp"][1] or 0.0), 2),
            "text":  c["text"].strip(),
        }
        for i, c in enumerate(chunks)
    ]
    duration = segments[-1]["end"] if segments else 0.0
    print(f" ✓ {time.time()-t0:.1f}s | {len(segments)} chunks | {len(full_text.split())} words")
    return {
        "full_text":  full_text,
        "segments":   segments,
        "language":   WHISPER_LANGUAGE,
        "duration_s": round(duration, 2),
    }


def save_transcript(t: dict, path: Path) -> None:
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
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def check_ollama_available() -> bool:
    try:
        return requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5).status_code == 200
    except Exception:
        return False


def generate_soap_anthropic(transcript: str) -> dict:
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


def extract_json_from_response(raw: str) -> dict:
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


def generate_soap_ollama(transcript: str) -> dict:
    prompt = SOAP_OLLAMA_PROMPT.format(transcript=transcript)
    print(f"\n[Part B] Generating SOAP via Ollama ({OLLAMA_MODEL}) [fallback] ...")
    t0 = time.time()
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model":  OLLAMA_MODEL,
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


def save_soap_json(soap: dict, transcript: str, llm: str, backend: str, path: Path) -> None:
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "asr_model":    WHISPER_HF_MODEL,
            "asr_backend":  "HuggingFace transformers pipeline",
            "llm_model":    llm,
            "llm_backend":  backend,
            "source_audio": AUDIO_FILE.name,
        },
        "transcript": transcript,
        "soap_note":  soap,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved → {path}")


def save_soap_markdown(soap: dict, llm: str, path: Path) -> None:
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

_semantic_encoder = None


def get_semantic_encoder() -> SentenceTransformer:
    global _semantic_encoder
    if _semantic_encoder is None:
        print("  Loading sentence-transformer (all-MiniLM-L6-v2) ...")
        _semantic_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _semantic_encoder


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a_norm @ b_norm.T


def check_subjective_leakage(soap: dict) -> dict:
    obj_text  = soap.get("objective", "")
    subj_text = soap.get("subjective", "")
    flags = []

    for pattern, severity in LINGUISTIC_PATTERNS:
        matches = re.findall(pattern, obj_text, flags=re.IGNORECASE)
        if matches:
            flags.append({
                "layer":    "linguistic",
                "pattern":  pattern,
                "matches":  [m if isinstance(m, str) else str(m) for m in matches],
                "severity": severity,
                "message":  f"Patient-report language pattern in Objective. Pattern: '{pattern}'",
            })

    if obj_text.strip() and subj_text.strip():
        obj_sents  = [s for s in sent_tokenize(obj_text)  if len(s.split()) >= 4]
        subj_sents = [s for s in sent_tokenize(subj_text) if len(s.split()) >= 4]
        if obj_sents and subj_sents:
            encoder    = get_semantic_encoder()
            obj_emb    = encoder.encode(obj_sents,  convert_to_numpy=True)
            subj_emb   = encoder.encode(subj_sents, convert_to_numpy=True)
            sim_matrix = cosine_similarity_matrix(obj_emb, subj_emb)
            for i, obj_sent in enumerate(obj_sents):
                max_sim = float(sim_matrix[i].max())
                best_j  = int(sim_matrix[i].argmax())
                if max_sim >= SEMANTIC_LEAKAGE_THRESHOLD:
                    severity = "HIGH" if max_sim >= 0.90 else "MEDIUM"
                    flags.append({
                        "layer":              "semantic",
                        "objective_sentence": obj_sent,
                        "similar_subjective": subj_sents[best_j],
                        "similarity_score":   round(max_sim, 3),
                        "severity":           severity,
                        "message": (
                            f"Objective sentence semantically similar to Subjective content "
                            f"(cosine {max_sim:.2f} >= {SEMANTIC_LEAKAGE_THRESHOLD})."
                        ),
                    })

    high   = sum(1 for f in flags if f["severity"] == "HIGH")
    medium = sum(1 for f in flags if f["severity"] == "MEDIUM")
    low    = sum(1 for f in flags if f["severity"] == "LOW")

    return {
        "validation_timestamp":  datetime.now().isoformat(),
        "validation_method":     "linguistic_patterns + semantic_similarity (all-MiniLM-L6-v2)",
        "semantic_threshold":    SEMANTIC_LEAKAGE_THRESHOLD,
        "overall_status":        "PASS" if (high == 0 and medium == 0) else "FAIL",
        "high_severity_count":   high,
        "medium_severity_count": medium,
        "low_severity_count":    low,
        "total_flags":           len(flags),
        "flags":                 flags,
        "objective_section":     obj_text,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("MEDICAL TRANSCRIPTION & SOAP NOTE PIPELINE (v2)")
    print("=" * 65)
    print(f"ASR  : {WHISPER_HF_MODEL} (HuggingFace transformers)")
    print(f"LLM  : {ANTHROPIC_MODEL} primary | Ollama {OLLAMA_MODEL} fallback")

    # Part A — Transcription
    asr           = load_asr_pipeline(WHISPER_HF_MODEL)
    transcription = transcribe_audio(asr, AUDIO_FILE)
    save_transcript(transcription, TRANSCRIPT_FILE)

    # Part B — SOAP generation
    anthropic_ok = check_anthropic_available()
    ollama_ok    = check_ollama_available()
    print(f"\n  Anthropic API key: {anthropic_ok} | Ollama: {ollama_ok}")

    if anthropic_ok:
        soap_note    = generate_soap_anthropic(transcription["full_text"])
        llm_used     = ANTHROPIC_MODEL
        backend_used = "anthropic_tool_use"
    elif ollama_ok:
        print("  Anthropic key not set — falling back to Ollama")
        soap_note    = generate_soap_ollama(transcription["full_text"])
        llm_used     = OLLAMA_MODEL
        backend_used = "ollama_prompt"
    else:
        print(
            "  ✗ No LLM available. Set ANTHROPIC_API_KEY or start Ollama.",
            file=sys.stderr,
        )
        sys.exit(1)

    missing = {"subjective", "objective", "assessment", "plan"} - set(soap_note)
    if missing:
        print(f"  ✗ SOAP note missing sections: {missing}", file=sys.stderr)
        sys.exit(1)

    save_soap_json(soap_note, transcription["full_text"], llm_used, backend_used, SOAP_JSON_FILE)
    save_soap_markdown(soap_note, llm_used, SOAP_MD_FILE)

    # Bonus — Validation
    print("\n[Bonus] Running S/O leakage validation (semantic + linguistic) ...")
    report = check_subjective_leakage(soap_note)
    with open(VALIDATION_FILE, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    status = "✓ PASS" if report["overall_status"] == "PASS" else "✗ FAIL"
    print(f"  {status} — {report['total_flags']} flag(s) [{report['high_severity_count']} HIGH, "
          f"{report['medium_severity_count']} MEDIUM, {report['low_severity_count']} LOW]")

    # Print SOAP note
    print("\n" + "=" * 65)
    print("GENERATED SOAP NOTE")
    print("=" * 65)
    for key, label in [
        ("subjective", "S — Subjective"),
        ("objective",  "O — Objective"),
        ("assessment", "A — Assessment"),
        ("plan",       "P — Plan"),
    ]:
        print(f"\n{label}")
        print("-" * len(label))
        print(textwrap.fill(soap_note.get(key, ""), width=63))

    # File inventory
    print("\n" + "=" * 65)
    print("OUTPUT FILES")
    print("=" * 65)
    for p in [TRANSCRIPT_FILE, SOAP_JSON_FILE, SOAP_MD_FILE, VALIDATION_FILE]:
        size = p.stat().st_size if p.exists() else 0
        print(f"  {p.name:<35} {size:>8} bytes")

    print(f"\n  Pipeline: {WHISPER_HF_MODEL} | {llm_used} ({backend_used})")
    print(f"  Validation: {report['overall_status']} via {report['validation_method']}")


if __name__ == "__main__":
    main()
