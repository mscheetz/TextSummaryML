from time import perf_counter

from server.src.config import LENGTH_SETTINGS, MAX_CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS, MAX_CHUNK_SUMMARY_TOKENS, MIN_CHUNK_SUMMARY_TOKENS, CHUNK_BATCH_SIZE, format_duration
from server.src.models.SummaryResult import SummaryResult
from server.src.services.WebHandler import url_to_md

def summarize_text(text: str, summary_length: str, summarizer, tokenizer, precount_seconds: int = 0) -> SummaryResult:
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
        
        summary_seconds = (perf_counter() - stopwatch_start) + precount_seconds
        summary_duration = format_duration(summary_seconds)
        result.summarization_seconds = summary_seconds

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

    summary_seconds = (perf_counter() - stopwatch_start) + precount_seconds
    summary_duration = format_duration(summary_seconds)

    result.summarization_duration = summary_duration
    result.summarization_seconds = summary_seconds
    result.summarization_passes = 0

    return result

async def summarize_web_page(url: str, summary_length: str, summarizer, tokenizer) -> SummaryResult:
    stopwatch_start = perf_counter()

    md = await url_to_md(url)

    md_characters = len(md)

    web_scrape_seconds = perf_counter() - stopwatch_start

    result = summarize_text(md, summary_length, summarizer, tokenizer, web_scrape_seconds)

    result.summarization_seconds += web_scrape_seconds
    result.summarization_duration = format_duration(result.summarization_seconds)
    result.url = url
    result.md_characters = md_characters
    
    return result

def summarize_long_text(
    text: str, 
    summary_length: str, 
    summarizer, 
    tokenizer, 
    total_chunks: int = 1,
    reduction_depth: int = 0,
) -> SummaryResult:
    chunks = chunk_text(text, tokenizer)

    if reduction_depth == 0:
        min_length = MIN_CHUNK_SUMMARY_TOKENS
        max_length = MAX_CHUNK_SUMMARY_TOKENS

    else:
        settings = LENGTH_SETTINGS[summary_length]
        min_length = max(
            MIN_CHUNK_SUMMARY_TOKENS,
            settings["min_length"] // len(chunks),
        )
        max_length = max(
            MAX_CHUNK_SUMMARY_TOKENS,
            settings["max_length"] // len(chunks),
        )
        

    chunk_summaries = summarize_chunks(
        chunks=chunks,
        summarizer=summarizer,
        tokenizer=tokenizer,
        min_length=min_length,
        max_length=max_length,
    )

    total_chunks = max(total_chunks, len(chunks))

    combined = "\n\n".join(chunk_summaries)

    reduced_chunks = chunk_text(combined, tokenizer)

    if len(reduced_chunks) == 1:
        settings = LENGTH_SETTINGS[summary_length]

        summary = get_text_summary(
            text=combined,
            min_length=settings["min_length"],
            max_length=settings["max_length"],
            summarizer=summarizer,
            tokenizer=tokenizer,
            max_output_ratio=0.8,
        )

        result = SummaryResult(summary, total_chunks)
        result.summarization_passes = reduction_depth

        return result

    return summarize_long_text(
        text=combined,
        summary_length=summary_length,
        summarizer=summarizer,
        tokenizer=tokenizer,
        total_chunks=total_chunks,
        reduction_depth=reduction_depth + 1,
    )

def summarize_chunks(
    chunks: list[str], 
    summarizer, 
    tokenizer,
    min_length: int,
    max_length: int,
) -> list[str]:
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
        max_length,
        max(20, int(shortest_chunk * 0.5)),
    )

    min_length_guard = min(
        min_length,
        max(10, max_length_guard - 20),
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

def get_text_summary(
    text: str, 
    min_length: int, 
    max_length: int, 
    summarizer, 
    tokenizer, 
    max_output_ratio: float = 0.25,
) -> str:
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
        max(20, int(input_token_count * max_output_ratio) // 2),
    )

    min_length_guard = min(
        min_length,
        max(10, max_length_guard // 2),
    )

    if max_output_ratio == 0.8:
        print(f"\nFinal Text:\n{text}")

    try:
        result = summarizer(
            text,
            min_length=min_length_guard,
            max_length=max_length_guard,
            do_sample=False,
            truncation=True,
            num_beams=4,
        )

        summary = result[0]["summary_text"].strip()

        return summary

    except Exception as ex:
        raise Exception(ex)
