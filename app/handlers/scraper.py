import logging
import re
from typing import Any

from bs4 import BeautifulSoup

from app.services.llm import LLMService
from app.services.browser import BrowserService

logger = logging.getLogger(__name__)


class ScraperHandler:
    """Handle web scraping quiz tasks."""

    def __init__(self, llm: LLMService):
        self.llm = llm

    async def solve(self, question: str, raw_html: str) -> Any:
        """
        Solve a web scraping quiz.

        Args:
            question: The quiz question
            raw_html: The raw HTML content to analyze

        Returns:
            The computed answer
        """
        # Extract meaningful content from HTML
        extracted_content = self._extract_content(raw_html)

        # Use LLM to analyze and answer
        answer = await self.llm.analyze_quiz(
            question=question,
            context=f"Scraped Content:\n{extracted_content}",
        )

        return answer

    def _extract_content(self, html: str) -> str:
        """Extract meaningful content from HTML."""
        soup = BeautifulSoup(html, "html.parser")

        # Remove script and style elements
        for elem in soup(["script", "style", "meta", "link"]):
            elem.decompose()

        content_parts = []

        # Extract text from main content areas
        for selector in ["main", "article", "#content", ".content", "#result", "body"]:
            elem = soup.select_one(selector)
            if elem:
                text = elem.get_text(separator="\n", strip=True)
                if text and len(text) > 50:
                    content_parts.append(text)
                    break

        # Extract all links
        links = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            text = link.get_text(strip=True)
            if text and href.startswith("http"):
                links.append(f"{text}: {href}")

        if links:
            content_parts.append("\nLinks found:\n" + "\n".join(links[:20]))

        # Extract tables
        for table in soup.find_all("table"):
            rows = []
            for tr in table.find_all("tr"):
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if cells:
                    rows.append(" | ".join(cells))
            if rows:
                content_parts.append("\nTable:\n" + "\n".join(rows))

        # Extract lists
        for ul in soup.find_all(["ul", "ol"]):
            items = [li.get_text(strip=True) for li in ul.find_all("li")]
            if items:
                content_parts.append("\nList:\n" + "\n".join(f"- {item}" for item in items))

        return "\n\n".join(content_parts)

    async def scrape_url(self, url: str) -> str:
        """Scrape content from a URL using the browser service."""
        content = await BrowserService.get_element_content(url)
        return self._extract_content(content)

    async def extract_specific_data(
        self,
        html: str,
        selector: str,
    ) -> list:
        """Extract data using a specific CSS selector."""
        soup = BeautifulSoup(html, "html.parser")
        elements = soup.select(selector)
        return [elem.get_text(strip=True) for elem in elements]
