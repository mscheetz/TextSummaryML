from contextlib import asynccontextmanager
from typing import Literal
from time import perf_counter

from fastapi import FastAPI, HTTPException
import torch
from transformers import pipeline
from pydantic import BaseModel, Field

from server.src.config import MODEL_NAME, LENGTH_SETTINGS, format_duration
from server.src.services.summarizer import summarize_text

summarizer = None
gpu_enabled: bool = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global summarizer
    global gpu_enabled

    print("TextSummaryML is starting up!")

    stopwatch_start = perf_counter()

    device = 0 
    backend = "CPU"
    device_name = "CPU"
    if torch.cuda.is_available():
        device = 0
        gpu_enabled = True

        backend = "AMD ROCm" if torch.version.hip else "NVIDIA CUDA"
        device_name = torch.cuda.get_device_name(0)
    else:
        device = -1

    print(f"Using {"GPU" if gpu_enabled else "CPU"}")

    summarizer = pipeline(
        task="summarization",
        model=MODEL_NAME,
        device=device,
    )

    training_seconds = perf_counter() - stopwatch_start
    training_duration = format_duration(training_seconds)

    print(f"TextSummaryML is start up in {training_duration}")
    print(f"\nModel: {MODEL_NAME} \n{"GPU" if gpu_enabled else "CPU"} Enabled\nBackend {backend}\nDevice {device_name}")

    yield

    summarizer = None

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
    length: Literal["short", "medium", "long"] = "medium"

class Response(BaseModel):
    summary: str
    input_characters: int
    summary_characters: int
    model: str

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
    if summarizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available",
        )

    print(f"New summary request of {len(request.text)} characters")

    try:
        summary = summarize_text(request.text, request.length, summarizer)

        return Response(
            summary=summary,
            input_characters=len(request.text),
            summary_characters=len(summary),
            model=MODEL_NAME
        )

    except Exception as ex:
        raise HTTPException(
            status_code=500,
            detail=f"Summarization error: {ex}"
        ) from ex