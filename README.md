# Medical Transcription & SOAP Note Generation Pipeline

A privacy-first pipeline that transcribes physician audio dictations and structures them into standardised SOAP notes.

Two implementations are provided — same assignment, different engineering approach:

| | v1 |
|-|--------------|-----------------|
| **Notebook** | `medical_transcription_soap.ipynb` |
| **Script** | `run_pipeline.py` |
| **ASR** |  HuggingFace `transformers` pipeline |
| **SOAP LLM** | Anthropic Claude `tool_use` (primary) |
| **Output extraction** | Function calling — schema guaranteed |
| **Validation** | Semantic similarity + linguistic patterns |

---

## Architecture


### HuggingFace transformers + Anthropic tool_use

```
sample_dictation.mp3
        │
        ▼
┌───────────────────────┐
│  transformers pipeline │  ← HuggingFace Whisper (chunk-and-stride)
│  openai/whisper-base   │
└──────────┬────────────┘
           │  raw_transcript.txt
           ▼
┌───────────────────────┐
│  Anthropic Claude      │  ← tool_use / function calling (schema-guaranteed)
│  tool_use primary      │    Ollama gemma2:2b as fallback
└──────────┬────────────┘
           │
           ├──→ outputs/soap_note.json
           ├──→ outputs/soap_note.md
           └──→ outputs/validation_report.json   (semantic similarity + linguistic)
```

### Model Choices & Rationale

| Component | Model | Why |
|-----------|-------|-----|
| ASR | HuggingFace `openai/whisper-base` | Unified HuggingFace hub API; built-in chunk-and-stride for long audio; swap model in one config line; PyTorch-native |
| SOAP LLM | Anthropic Claude `tool_use` | `tool_choice` forces schema-compliant output — no regex JSON extraction; type-safe structured fields; swap `claude-sonnet-4-6` for any Claude model |
| Fallback LLM | Ollama `gemma2:2b` | Fully-local fallback when no API key present |
| Validation | `sentence-transformers` + regex | Cosine similarity catches paraphrased patient-reported content that regex alone misses |

---

## Prerequisites
- Python 3.9+, `ANTHROPIC_API_KEY` env var (or Ollama as fallback)

---

## Installation

### dependencies

```bash
pip install transformers torch torchaudio \
    sentence-transformers nltk requests anthropic jupyter
```

### Ollama (optional)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma2:2b
ollama serve
```

### Anthropic API key (primary)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Running the Notebooks

```bash
jupyter notebook medical_transcription_soap.ipynb
```

Both notebooks: **Run All Cells** (`Kernel → Restart & Run All`).

Each notebook will:
1. Transcribe `sample_dictation.mp3` (first run downloads ASR model weights)
2. Generate a structured SOAP note from the transcript
3. Run the subjective/objective leakage validation check
4. Save all outputs to the `outputs/` directory

### Standalone scripts

```bash

python run_pipeline.py

# Override ASR model
WHISPER_MODEL=openai/whisper-small python run_pipeline.py
```

---

## Output Files

| File | Description |
|------|-------------|
| `outputs/raw_transcript.txt` | Verbatim transcript with per-segment timestamps |
| `outputs/soap_note.json` | SOAP note as JSON with generation metadata |
| `outputs/soap_note.md` | SOAP note in Markdown format |
| `outputs/validation_report.json` | Bonus: S/O leakage check results |

### Example `soap_note.json` structure

```json
{
  "metadata": {
    "generated_at": "2026-03-25T10:00:00",
    "asr_model": "large-v2",
    "llm_model": "gemma2:2b",
    "source_audio": "sample_dictation.mp3"
  },
  "transcript": "...",
  "soap_note": {
    "subjective":  "Patient reports ...",
    "objective":   "On exam ...",
    "assessment":  "Suspected ...",
    "plan":        "Prescribe ..."
  }
}
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_HF_MODEL` | `"openai/whisper-base"` | HuggingFace hub model ID |
| `ANTHROPIC_MODEL` | `"claude-sonnet-4-6"` | Any Anthropic model |
| `OLLAMA_MODEL` | `"gemma2:2b"` | Fallback Ollama model |
| `SEMANTIC_LEAKAGE_THRESHOLD` | `0.80` | Cosine similarity threshold for leakage detection |

---

## Validation Logic
### Semantic similarity + linguistic patterns

**Layer 1 (linguistic):** Same regex scan as v1 for explicit markers.

**Layer 2 (semantic):** Encodes each Objective and Subjective sentence with `sentence-transformers` (`all-MiniLM-L6-v2`) and computes cosine similarity. Sentences scoring above `SEMANTIC_LEAKAGE_THRESHOLD` are flagged — catches *paraphrased* patient-reported content that regex misses.

Both versions classify flags as **HIGH**, **MEDIUM**, or **LOW** severity and write to `outputs/validation_report.json`.

---

## Performance Notes

| Scenario | Approximate Time |
|----------|-----------------|
| whisper-base on CPU (13-min audio) | 10–20 minutes |
| whisper-small on CPU (13-min audio) | 20–40 minutes |
| Anthropic Claude SOAP generation | 5–15 seconds |
| Semantic validation (all-MiniLM-L6-v2) | 1–3 seconds |

This is faster for SOAP generation (Anthropic API vs local Ollama).

For CPU-only with no API key: set `WHISPER_HF_MODEL = "openai/whisper-base"` and start Ollama.
