MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

LENGTH_SETTINGS = {
    "short": {"min_length": 20, "max_length": 60},
    "medium": {"min_length": 40, "max_length": 130},
    "long": {"min_length": 80, "max_length": 220},
}

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