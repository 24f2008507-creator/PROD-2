import base64
import io
import logging
from typing import Any, Optional

import httpx
import pdfplumber

from app.services.llm import LLMService

logger = logging.getLogger(__name__)


class PDFHandler:
    """Handle PDF-related quiz tasks."""

    def __init__(self, llm: LLMService):
        self.llm = llm
        self.client = httpx.AsyncClient(timeout=60.0)

    async def solve(self, question: str, download_url: str) -> Any:
        """
        Solve a PDF-related quiz.

        Args:
            question: The quiz question
            download_url: URL to download the PDF

        Returns:
            The computed answer
        """
        logger.info(f"Downloading PDF from {download_url}")

        try:
            # Download the PDF
            response = await self.client.get(download_url)
            response.raise_for_status()
            pdf_bytes = response.content

            # Extract text from PDF
            text_content = await self._extract_text(pdf_bytes)
            logger.info(f"Extracted {len(text_content)} chars from PDF")

            # Use LLM to analyze and answer
            answer = await self.llm.analyze_quiz(
                question=question,
                context=f"PDF Content:\n{text_content}",
            )

            return answer

        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            # Try vision-based approach
            return await self._solve_with_vision(question, download_url)

    async def _extract_text(self, pdf_bytes: bytes) -> str:
        """Extract text content from PDF bytes."""
        text_parts = []

        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for i, page in enumerate(pdf.pages):
                    # Extract text
                    text = page.extract_text() or ""
                    text_parts.append(f"=== Page {i + 1} ===\n{text}")

                    # Extract tables
                    tables = page.extract_tables()
                    for j, table in enumerate(tables):
                        if table:
                            table_text = self._format_table(table)
                            text_parts.append(f"--- Table {j + 1} on Page {i + 1} ---\n{table_text}")

        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")

        return "\n\n".join(text_parts)

    def _format_table(self, table: list) -> str:
        """Format a table as text."""
        if not table:
            return ""

        lines = []
        for row in table:
            if row:
                # Clean None values
                clean_row = [str(cell) if cell else "" for cell in row]
                lines.append(" | ".join(clean_row))

        return "\n".join(lines)

    async def _solve_with_vision(self, question: str, download_url: str) -> Any:
        """Fallback: Use vision to analyze PDF pages as images."""
        try:
            # Download PDF
            response = await self.client.get(download_url)
            pdf_bytes = response.content

            # Convert first page to image
            from pdf2image import convert_from_bytes

            images = convert_from_bytes(pdf_bytes, first_page=1, last_page=3)

            # Analyze with vision
            for i, img in enumerate(images):
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                img_base64 = base64.b64encode(buffer.getvalue()).decode()

                answer = await self.llm.analyze_with_vision(
                    img_base64,
                    f"This is page {i + 1} of a PDF. {question}",
                )

                if answer and answer.strip():
                    return self.llm._parse_answer(answer)

        except ImportError:
            logger.warning("pdf2image not available for vision fallback")
        except Exception as e:
            logger.error(f"Vision fallback failed: {e}")

        return None

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
