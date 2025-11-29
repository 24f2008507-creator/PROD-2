import re
import logging
from typing import Optional, Tuple
from bs4 import BeautifulSoup

from app.models import QuizContent

logger = logging.getLogger(__name__)


class QuizParser:
    """Parse quiz content from rendered HTML pages."""

    @staticmethod
    def parse_quiz_page(html_content: str, quiz_url: str) -> QuizContent:
        """
        Parse the quiz page HTML and extract relevant information.

        Args:
            html_content: The fully rendered HTML content
            quiz_url: The original quiz URL

        Returns:
            QuizContent with parsed question, submission URL, etc.
        """
        soup = BeautifulSoup(html_content, "html.parser")

        # Get the main content (usually in #result div)
        result_div = soup.find(id="result")
        if result_div:
            content_text = result_div.get_text(separator="\n", strip=True)
            content_html = str(result_div)
        else:
            content_text = soup.get_text(separator="\n", strip=True)
            content_html = html_content

        logger.info(f"Parsed content: {content_text[:500]}...")

        # Extract submission URL
        submission_url = QuizParser._extract_submission_url(content_text, content_html)

        # Extract download URL if present
        download_url = QuizParser._extract_download_url(content_html)

        # Determine quiz type
        quiz_type = QuizParser._determine_quiz_type(content_text, download_url)

        # Extract the question
        question = QuizParser._extract_question(content_text)

        return QuizContent(
            question=question,
            submission_url=submission_url,
            download_url=download_url,
            quiz_type=quiz_type,
            raw_html=content_html,
        )

    @staticmethod
    def _extract_submission_url(text: str, html: str) -> str:
        """Extract the submission URL from the quiz content."""
        # First, normalize text by removing newlines within URLs
        # Join lines that look like they're part of a URL
        normalized_text = re.sub(r'\n(?=\/)', '', text)  # Join /path after newline
        normalized_text = re.sub(r'(https?://[^\s]+)\n([^\s]+)', r'\1\2', normalized_text)

        # Also try the raw HTML for href attributes
        soup = BeautifulSoup(html, "html.parser")

        # First check href attributes (most reliable)
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if "submit" in href.lower():
                logger.info(f"Found submission URL in href: {href}")
                return href

        # Look for URLs in the normalized text
        patterns = [
            r'https?://[^\s<>"\'\{\}\[\]]+/submit(?:[^\s<>"\'\{\}\[\]]*)?',
            r'https?://[^\s<>"\'\{\}\[\]]+/answer(?:[^\s<>"\'\{\}\[\]]*)?',
            r'Post your answer to\s+(https?://[^\s<>"\'\{\}\[\]]+)',
            r'POST.*?to\s+(https?://[^\s<>"\'\{\}\[\]]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, normalized_text, re.IGNORECASE)
            if match:
                url = match.group(1) if match.lastindex else match.group(0)
                # Clean any trailing non-URL characters
                url = re.sub(r'[{}\[\]<>"\'\s.,;:]+$', '', url)
                logger.info(f"Found submission URL: {url}")
                return url

        # Try to reconstruct URL from base domain + /submit
        base_url_match = re.search(r'(https?://[^\s<>"\'/\n]+)', text)
        if base_url_match:
            base_url = base_url_match.group(1)
            if "/submit" in text.lower():
                constructed_url = f"{base_url}/submit"
                # Clean any trailing non-URL characters
                constructed_url = re.sub(r'[{}\[\]<>"\'\s]+$', '', constructed_url)
                logger.info(f"Constructed submission URL: {constructed_url}")
                return constructed_url

        # Fallback: look for any https URL with submit
        all_urls = re.findall(r'https?://[^\s<>"\']+', normalized_text)
        for url in all_urls:
            if "submit" in url.lower() or "answer" in url.lower():
                return url.rstrip(".,;:\"'")

        logger.warning("Could not find submission URL")
        return ""

    @staticmethod
    def make_absolute_url(url: str, base_url: str) -> str:
        """Convert a relative URL to absolute using the base URL."""
        if url.startswith("http"):
            return url

        from urllib.parse import urljoin, urlparse
        parsed = urlparse(base_url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        return urljoin(base, url)

    @staticmethod
    def _extract_download_url(html: str) -> Optional[str]:
        """Extract download URL for files (PDFs, data files, etc.)."""
        soup = BeautifulSoup(html, "html.parser")

        # Look for links with file extensions
        file_extensions = [".pdf", ".csv", ".json", ".xlsx", ".txt", ".zip"]

        for link in soup.find_all("a", href=True):
            href = link["href"]
            for ext in file_extensions:
                if ext in href.lower():
                    # Make absolute URL if needed
                    if href.startswith("http"):
                        return href
                    logger.info(f"Found download URL: {href}")
                    return href

        # Also check in text
        text = soup.get_text()
        for ext in file_extensions:
            pattern = rf'(https?://[^\s<>"\']+{ext}[^\s<>"\']*)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).rstrip(".,;:\"'")

        return None

    @staticmethod
    def _determine_quiz_type(text: str, download_url: Optional[str]) -> str:
        """Determine the type of quiz based on content."""
        text_lower = text.lower()

        if download_url:
            if ".pdf" in download_url.lower():
                return "pdf"
            if any(ext in download_url.lower() for ext in [".csv", ".xlsx", ".json"]):
                return "data"

        if "api" in text_lower or "endpoint" in text_lower:
            return "api"

        if any(word in text_lower for word in ["chart", "plot", "graph", "visualiz"]):
            return "visualization"

        if any(word in text_lower for word in ["sum", "average", "count", "filter", "sort", "aggregate"]):
            return "data"

        if any(word in text_lower for word in ["scrape", "extract", "website", "page"]):
            return "scraping"

        return "general"

    @staticmethod
    def _extract_question(text: str) -> str:
        """Extract the question from the quiz text."""
        # Remove submission instructions
        lines = text.split("\n")
        question_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Skip JSON payload examples
            if line.startswith("{") or line.startswith("}"):
                continue
            if "email" in line.lower() and "secret" in line.lower():
                continue
            if "post your answer" in line.lower():
                continue

            question_lines.append(line)

        question = "\n".join(question_lines[:10])  # First 10 meaningful lines
        return question

    @staticmethod
    def extract_json_payload_format(text: str) -> dict:
        """Extract the expected JSON payload format from the quiz."""
        # Look for JSON-like structures in the text
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)

        for match in matches:
            try:
                # Try to parse as JSON
                import json
                return json.loads(match)
            except:
                continue

        return {}
