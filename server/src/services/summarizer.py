from time import perf_counter

from server.src.config import LENGTH_SETTINGS, MAX_CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS, MAX_CHUNK_SUMMARY_TOKENS, MIN_CHUNK_SUMMARY_TOKENS, CHUNK_BATCH_SIZE, format_duration
from server.src.models.SummaryResult import SummaryResult

def summarize_text(text: str, summary_length: str, summarizer, tokenizer) -> SummaryResult:
    stopwatch_start = perf_counter()

    chunks = chunk_text(text, tokenizer)

    if len(chunks) > 1:
        result = summarize_long_text(
            text,
            summary_length,
            summarizer,
            tokenizer,
            len(chunks)
        )
        
        summary_seconds = perf_counter() - stopwatch_start
        summary_duration = format_duration(summary_seconds)

        result.summarization_duration = summary_duration

        return result

    settings = LENGTH_SETTINGS[summary_length]

    summary = get_text_summary(
        text, 
        settings["min_length"], 
        settings["max_length"], 
        summarizer,
        tokenizer,
    )

    result = SummaryResult(summary, 1)

    summary_seconds = perf_counter() - stopwatch_start
    summary_duration = format_duration(summary_seconds)

    result.summarization_duration = summary_duration

    return result

def summarize_long_text(text: str, summary_length: str, summarizer, tokenizer, total_chunks: int = 1) -> SummaryResult:
    chunks = chunk_text(text, tokenizer)

    chunk_summaries = summarize_chunks(
        chunks,
        summarizer,
        tokenizer,
    )

    total_chunks = max(total_chunks, len(chunks))

    combined = "\n\n".join(chunk_summaries)

    reduced_chunks = chunk_text(combined, tokenizer)

    if len(reduced_chunks) == 1:
        settings = LENGTH_SETTINGS[summary_length]

        summary = get_text_summary(
            combined,
            settings["min_length"],
            settings["max_length"],
            summarizer,
            tokenizer,
        )

        return SummaryResult(summary, total_chunks)

    return summarize_long_text(
        combined,
        summary_length,
        summarizer,
        tokenizer,
        total_chunks
    )

def summarize_chunks(chunks: list[str], summarizer, tokenizer) -> list[str]:
    input_token_counts = [
        len(
            tokenizer.encode(
                chunk,
                add_special_tokens=False,
                truncation=True,
                max_length=MAX_CHUNK_TOKENS,
            )
        )
        for chunk in chunks
    ]

    shortest_chunk = min(input_token_counts)

    max_length_guard = min(
        MAX_CHUNK_SUMMARY_TOKENS,
        max(20, shortest_chunk // 2),
    )

    min_length_guard = min(
        MIN_CHUNK_SUMMARY_TOKENS,
        max(10, max_length_guard // 2),
    )

    results = summarizer(
        chunks,
        min_length=min_length_guard,
        max_length=max_length_guard,
        do_sample=False,
        truncation=True,
        batch_size=CHUNK_BATCH_SIZE,
    )

    return [
        result["summary_text"].strip()
        for result in results
    ]

def chunk_text(text: str, tokenizer) -> list[str]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_CHUNK_TOKENS,
        stride=CHUNK_OVERLAP_TOKENS,
        return_overflowing_tokens=True,
        return_attention_mask=False,
    )

    input_chunks = encoded["input_ids"]

    return [
        tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaced=True,
        )
        for chunk_ids in input_chunks
    ]

def summarize_chunk(text: str, summarizer, tokenizer) -> str:
        return get_text_summary(text, MIN_CHUNK_SUMMARY_TOKENS, MAX_CHUNK_SUMMARY_TOKENS, summarizer, tokenizer)

def get_text_summary(text, min_length: int, max_length: int, summarizer, tokenizer) -> str:
    input_token_count = len(
        tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_CHUNK_TOKENS,
        )
    )

    max_length_guard = min(
        max_length,
        max(20, input_token_count // 2),
    )

    min_length_guard = min(
        min_length,
        max(10, max_length_guard // 2),
    )

    try:
        result = summarizer(
            text,
            min_length=min_length_guard,
            max_length=max_length_guard,
            do_sample=False,
            truncation=True,
        )

        summary = result[0]["summary_text"].strip()

        return summary

    except Exception as ex:
        raise Exception(ex)
