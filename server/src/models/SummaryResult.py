from dataclasses import dataclass
from typing import Optional

@dataclass
class SummaryResult:
    summary: str
    total_chunks: int

    summarization_duration: str = ""
    summarization_seconds: float = 0.0
    summarization_passes: int = 0

    url: Optional[str] = None
    md_characters: Optional[int] = None

    def __init__(self, summary: str, chunks: int):
        self.summary = summary
        self.total_chunks = chunks
    