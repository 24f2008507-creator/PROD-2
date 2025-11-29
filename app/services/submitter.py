import logging
from typing import Any, Optional
import httpx

from app.models import AnswerSubmission, QuizResult

logger = logging.getLogger(__name__)


class AnswerSubmitter:
    """Service for submitting answers to quiz endpoints."""

    def __init__(self, email: str, secret: str):
        self.email = email
        self.secret = secret
        self.client = httpx.AsyncClient(timeout=30.0)

    async def submit_answer(
        self,
        submission_url: str,
        quiz_url: str,
        answer: Any,
    ) -> QuizResult:
        """
        Submit an answer to the quiz endpoint.

        Args:
            submission_url: URL to submit the answer to
            quiz_url: Original quiz URL
            answer: The answer (can be bool, number, string, or dict)

        Returns:
            QuizResult with correct status and next URL if available
        """
        payload = {
            "email": self.email,
            "secret": self.secret,
            "url": quiz_url,
            "answer": answer,
        }

        logger.info(f"Submitting answer to {submission_url}")
        logger.info(f"Payload: {payload}")

        try:
            response = await self.client.post(
                submission_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            logger.info(f"Response status: {response.status_code}")
            logger.info(f"Response body: {response.text[:500]}...")

            if response.status_code == 200:
                data = response.json()
                return QuizResult(
                    correct=data.get("correct", False),
                    url=data.get("url"),
                    reason=data.get("reason"),
                )
            else:
                logger.error(f"Submission failed: {response.status_code} - {response.text}")
                return QuizResult(
                    correct=False,
                    reason=f"HTTP {response.status_code}: {response.text}",
                )

        except Exception as e:
            logger.error(f"Error submitting answer: {e}")
            return QuizResult(
                correct=False,
                reason=str(e),
            )

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
