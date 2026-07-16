from contextlib import asynccontextmanager
from typing import Literal
from time import perf_counter

from fastapi import FastAPI, HTTPException
import torch
from transformers import pipeline, AutoTokenizer
from pydantic import BaseModel, Field

from server.src.config import MODEL_NAME, format_duration, CHUNK_BATCH_SIZE
from server.src.services.summarizer import summarize_text
from server.src.services.logger import logger

summarizer = None
tokenizer = None
gpu_enabled: bool = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("TextSummaryML is starting up!")

    stopwatch_start = perf_counter()

    device = 0 
    backend = "CPU"
    device_name = "CPU"
    if torch.cuda.is_available():
        device = 0
        app.state.gpu_enabled = True

        backend = "AMD ROCm" if torch.version.hip else "NVIDIA CUDA"
        device_name = torch.cuda.get_device_name(0)
    else:
        device = -1
        CHUNK_BATCH_SIZE = 1

    logger.info(f"Using {"GPU" if gpu_enabled else "CPU"}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    logger.info("Tokenizer ready!")

    summarizer = pipeline(
        task="summarization",
        model=MODEL_NAME,
        device=device,
    )

    logger.info("Summarizer ready!")

    training_seconds = perf_counter() - stopwatch_start
    training_duration = format_duration(training_seconds)

    logger.info(f"TextSummaryML started up in {training_duration}")
    logger.info(f"\nModel: {MODEL_NAME} \n{"GPU" if gpu_enabled else "CPU"} Enabled\nBackend {backend}\nDevice {device_name}")

    app.state.tokenizer = tokenizer
    app.state.summarizer = summarizer

    yield

    app.state.tokenizer = None
    app.state.summarizer = None

app = FastAPI(
    title="TextSummaryML",
    version="0.0.1",
    lifespan=lifespan,
)

class Request(BaseModel):
    text: str = Field(
        min_length=50,
        max_length=50_000,
        description="Text to summarize.",
    )
    length: Literal["short", "medium", "long", "extra-long"] = "medium"

class Response(BaseModel):
    summary: str
    input_characters: int
    summary_characters: int
    model: str
    total_chunks: int
    duration: str

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": summarizer is not None,
        "model": MODEL_NAME,
        "gpu_enabled": gpu_enabled
    }

@app.post("/api/summarize", response_model=Response)
def summarize(request: Request):
    if app.state.summarizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available",
        )

    logger.info(f"New summary request of {len(request.text)} characters")

    try:
        summary = summarize_text(
            request.text, 
            request.length, 
            app.state.summarizer, 
            app.state.tokenizer,
        )

        return Response(
            summary=summary.summary,
            input_characters=len(request.text),
            summary_characters=len(summary.summary),
            model=MODEL_NAME,
            total_chunks=summary.total_chunks,
            duration=summary.summarization_duration
        )

    except Exception as ex:
        raise HTTPException(
            status_code=500,
            detail=f"Summarization error: {ex}"
        ) from ex