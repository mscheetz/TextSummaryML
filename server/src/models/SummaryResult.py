from dataclasses import dataclass

@dataclass
class SummaryResult:
    summary: str
    total_total_chunks: int
    summarization_duration: str

    def __init__(self, summary: str, chunks: int):
        self.summary = summary
        self.total_chunks = chunks
    