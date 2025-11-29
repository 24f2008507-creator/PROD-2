import asyncio
import base64
import logging
from typing import Optional
from playwright.async_api import async_playwright, Browser, Page, Playwright

from app.config import settings

logger = logging.getLogger(__name__)


class BrowserService:
    """Singleton service for managing Playwright browser."""

    _playwright: Optional[Playwright] = None
    _browser: Optional[Browser] = None
    _lock = asyncio.Lock()

    @classmethod
    async def initialize(cls) -> None:
        """Initialize the Playwright browser."""
        async with cls._lock:
            if cls._browser is None:
                logger.info("Initializing Playwright browser...")
                cls._playwright = await async_playwright().start()
                cls._browser = await cls._playwright.chromium.launch(
                    headless=settings.BROWSER_HEADLESS,
                    args=[
                        "--no-sandbox",
                        "--disable-setuid-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-gpu",
                    ],
                )
                logger.info("Browser initialized successfully")

    @classmethod
    async def cleanup(cls) -> None:
        """Cleanup browser resources."""
        async with cls._lock:
            if cls._browser:
                await cls._browser.close()
                cls._browser = None
            if cls._playwright:
                await cls._playwright.stop()
                cls._playwright = None
            logger.info("Browser resources cleaned up")

    @classmethod
    async def is_healthy(cls) -> bool:
        """Check if browser is healthy."""
        return cls._browser is not None and cls._browser.is_connected()

    @classmethod
    async def get_page(cls) -> Page:
        """Create a new page in a fresh context."""
        if cls._browser is None:
            await cls.initialize()

        context = await cls._browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        )
        page = await context.new_page()
        return page

    @classmethod
    async def get_page_content(cls, url: str, wait_selector: str = "#result") -> str:
        """
        Navigate to URL and get rendered HTML content.

        This handles JavaScript-rendered pages by waiting for the content
        to be injected into the DOM.
        """
        page = await cls.get_page()
        try:
            logger.info(f"Navigating to {url}")
            await page.goto(url, wait_until="networkidle", timeout=30000)

            # Wait for the result element to be populated
            try:
                await page.wait_for_selector(wait_selector, timeout=10000)
                # Additional wait for JavaScript to execute
                await asyncio.sleep(1)
            except Exception:
                # If no #result selector, try body
                await page.wait_for_load_state("domcontentloaded")

            # Get the full page HTML
            content = await page.content()
            logger.info(f"Retrieved content from {url} ({len(content)} chars)")
            return content

        finally:
            await page.context.close()

    @classmethod
    async def get_element_content(cls, url: str, selector: str = "#result") -> str:
        """Get the inner HTML of a specific element after JavaScript execution."""
        page = await cls.get_page()
        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)

            try:
                await page.wait_for_selector(selector, timeout=10000)
                await asyncio.sleep(1)  # Wait for JS to complete
                element = await page.query_selector(selector)
                if element:
                    content = await element.inner_html()
                    return content
            except Exception:
                pass

            # Fallback to body content
            return await page.inner_html("body")

        finally:
            await page.context.close()

    @classmethod
    async def take_screenshot(cls, url: str) -> str:
        """Take a screenshot and return as base64."""
        page = await cls.get_page()
        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)
            await asyncio.sleep(2)  # Wait for full render

            screenshot_bytes = await page.screenshot(full_page=True)
            return base64.b64encode(screenshot_bytes).decode("utf-8")

        finally:
            await page.context.close()

    @classmethod
    async def download_file(cls, url: str) -> bytes:
        """Download a file and return its bytes."""
        page = await cls.get_page()
        try:
            response = await page.request.get(url)
            return await response.body()
        finally:
            await page.context.close()

    @classmethod
    async def execute_and_get_content(cls, url: str) -> dict:
        """
        Navigate to URL, execute JavaScript, and extract quiz content.

        Returns dict with:
        - rendered_html: The fully rendered HTML after JS execution
        - screenshot_base64: Screenshot of the page
        """
        page = await cls.get_page()
        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)

            # Wait for content to render
            try:
                await page.wait_for_selector("#result", timeout=10000)
            except Exception:
                pass

            await asyncio.sleep(2)  # Ensure JavaScript has executed

            rendered_html = await page.content()
            screenshot = await page.screenshot(full_page=True)

            return {
                "rendered_html": rendered_html,
                "screenshot_base64": base64.b64encode(screenshot).decode("utf-8"),
            }

        finally:
            await page.context.close()
