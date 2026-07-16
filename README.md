# TextSummaryML

FastAPI-powered text summarization service using Hugging Face transformers. Supports GPU acceleration (AMD ROCm / NVIDIA CUDA) and handles long documents via intelligent chunking.

## Stack

- **Python 3.12** — managed via `uv`
- **FastAPI** — REST API with Pydantic v2 validation
- **Transformers** — `sshleifer/distilbart-cnn-12-6` model
- **PyTorch** — with ROCm index for AMD GPUs

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Health check, model status, GPU availability |
| `POST` | `/api/summarize/text` | Summarize text (50–50,000 chars) |
| `POST` | `/api/summarize/web` | Summarize web page |

### Summarize Request

```json
{
  "text": "Your long text here...",
  "length": "medium"
}
```

`length` accepts: `short`, `medium`, `long`, `extra-long`.

### Summarize Response

```json
{
  "summary": "...",
  "input_characters": 2847,
  "summary_characters": 142,
  "model": "sshleifer/distilbart-cnn-12-6",
  "total_chunks": 2,
  "duration": "1.23s"
}
```

## How It Works

1. Text is tokenized and split into 900-token chunks with 80-token overlap
   - Web pages have html -> markdown before this step
2. Each chunk is summarized independently (batch size: 8)
3. If combined summaries still exceed the token limit, the process recurses
4. A final pass produces the summary at the requested length

## Quick Start

```bash
uv sync
./run-dev.sh
```

## GPU Setup

ROCm 6.4 index is configured in `pyproject.toml` for Linux. The `run-dev.sh` script sets `HSA_OVERRIDE_GFX_VERSION=10.3.0` for MI250-class GPUs; adjust for your hardware.

## Development

```bash
uv run uvicorn server.src.api.app:app --reload
```

Test endpoints via `server/tests/tests.http` (REST Client compatible).