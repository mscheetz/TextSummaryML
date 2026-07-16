from server.src.config import LENGTH_SETTINGS

def summarize_text(text: str, summary_size: str, summarizer):
    settings = LENGTH_SETTINGS[summary_size]

    try:
        result = summarizer(
            text,
            min_length=settings["min_length"],
            max_length=settings["max_length"],
            do_sample=False,
            truncation=True,
        )

        summary = result[0]["summary_text"].strip()

        return summary

    except Exception as ex:
        raise Exception(ex)
