# AGENTS.md — TextSummaryML

## First read

- `README.md` — full project overview
- `server/src/config.py` — model name, chunk config, length settings
- `server/src/api/app.py` — API entrypoint, lifespan lifecycle
- `server/src/services/summarizer.py` — chunking + recursive merge logic

## Dev commands

| Action | Command |
|--------|---------|
| Install deps | `uv sync` |
| Dev server | `uv run uvicorn server.src.api.app:app --reload` |
| Quick test | `./run-dev.sh` (sets `HSA_OVERRIDE_GFX_VERSION=10.3.0` for ROCm) |
| Test endpoints | `server/tests/tests.http` (VS Code REST Client) |

No lint, typecheck, or test framework configured.

## Project structure

```
server/src/
  api/app.py     — FastAPI app, lifespan (model load), /api/health, /api/summarize
  services/
    summarizer.py — chunk_text → summarize_chunks → recursive merge → final summary
    logger.py      — logging.basicConfig level=INFO
  models/
    SummaryResult.py — dataclass (summary, total_chunks, summarization_duration)
  config.py        — MODEL_NAME, LENGTH_SETTINGS, chunk params
  tests/tests.http — smoke tests
```

## Architecture notes

- **Model**: `sshleifer/distilbart-cnn-12-6` (distilbart, ~600MB)
- **Chunking**: 900-token chunks with 80-token overlap via tokenizer stride. If combined summaries exceed token limit, recursion until fit.
- **Batch summarization**: `summarizer(chunks, batch_size=8)` — process chunks in parallel
- **GPU**: auto-detects CUDA or ROCm via `torch.cuda.is_available()`. ROCm index pinned in `pyproject.toml` under `[[tool.uv.index]]`
- **Length settings**: short (20-60), medium (40-130), long (80-220), extra-long (175-500) tokens

## Gotchas

- `SummaryResult` has a bug: `__init__` assigns `self.total_chunks` but dataclass field is named `total_total_chunks`. Field name is self.total_chunks.
- `app.py` sets `MODEL_NAME` import but uses module-level `gpu_enabled` variable — `summarizer` and `tokenizer` are also module-level. State is duplicated between module level and `app.state`.
- No CI/CD, no pre-commit hooks.
- Lockfile: `uv.lock` (uv-native).
