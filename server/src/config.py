MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

LENGTH_SETTINGS = {
    "short": {"min_length": 20, "max_length": 60},
    "medium": {"min_length": 40, "max_length": 130},
    "long": {"min_length": 80, "max_length": 220},
    "extra-long": {"min_length": 175, "max_length": 500},
}

MAX_CHUNK_TOKENS: int = 900
CHUNK_OVERLAP_TOKENS: int = 80

MAX_CHUNK_SUMMARY_TOKENS: int = 150
MIN_CHUNK_SUMMARY_TOKENS: int = 40

CHUNK_BATCH_SIZE: int = 8

def format_duration(seconds: float) -> str:
    minutes, remaining_seconds = divmod(seconds, 60)
    hours, remaining_minutes = divmod(int(minutes), 60)

    if hours:
        return (
            f"{hours}h {remaining_minutes}m "
            f"{remaining_seconds:.2f}s"
        )

    if minutes >= 1:
        return f"{int(minutes)}m {remaining_seconds:.2f}s"

    return f"{seconds:.2f}s"