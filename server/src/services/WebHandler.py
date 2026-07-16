import httpx
from trafilatura import extract

from server.src.services.logger import logger

async def url_to_md(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; TextSummaryML/1.0)"
    }

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(20.0),
            headers=headers,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

    except httpx.TimeoutException as ex:
        raise WebsiteExtractionError(
            "The website request timed out."
        ) from exc

    except httpx.HTTPStatusError as exc:
        raise WebsiteExtractionError(
            f"Website returned HTTP {exc.response.status_code}."
        ) from exc

    except httpx.RequestError as exc:
        raise WebsiteExtractionError(
            f"Could not retrieve the website: {exc}"
        ) from exc

    content_type = response.headers.get("content-type", "")

    if "text/html" not in content_type.lower():
        raise WebsiteExtractionError(
            f"Expected HTML but received {content_type or 'unknown content type'}."
        )
        
    markdown = extract(
        response.text,
        url=str(response.url),
        output_format="markdown",
        include_links=False,
        include_images=False,
        include_tables=True,
        include_comments=False,
        favor_precision=True,
    )

    if not markdown:
        raise WebsiteExtractionError(
            "No readable page content could be extracted."
        )

    logger.info(f"{url} converted to markdown with {len(markdown)} characters.")

    return markdown.strip()
