from langchain_core.tools import tool
import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

_crawler: AsyncWebCrawler | None = None
_lock = asyncio.Lock()

async def get_crawler() -> AsyncWebCrawler:
    global _crawler
    if _crawler is None:
        async with _lock:
            if _crawler is None:
                browser_config = BrowserConfig(
                    headless=True,
                    text_mode=True,
                    light_mode=True,
                    avoid_ads=True,
                    avoid_css=True,
                    extra_args=["--disable-gpu", "--disable-software-rasterizer"],
                )
                _crawler = AsyncWebCrawler(config=browser_config)
                await _crawler.start()
    return _crawler

@tool
async def web_crawler(url: str) -> str:
    """
    Use this tool only when you want to scrape content live from the questohive site using https eg https://www.questohive.com/ or any other questohive related link
    """
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.ENABLED,
        wait_until="domcontentloaded",
        page_timeout=15000,
        excluded_tags=["script", "style", "nav", "footer", "svg"],
        # no fixed session_id → safe under concurrent calls
    )
    try:
        crawler = await get_crawler()
        result = await crawler.arun(url=url, config=run_config)
        if result.success:
            markdown = getattr(result, 'markdown_v2', None)
            if markdown and hasattr(markdown, 'raw_markdown'):
                return markdown.raw_markdown.strip()
            return (result.markdown or "No content extracted").strip()
        return f"Scrape failed: {result.error_message or 'Unknown error'}"
    except Exception as e:
        return f"Deep scrape error: {str(e)}"


if __name__ == "__main__":
    print("Set up is complete")
