from pydantic import BaseModel, EmailStr
from typing import Any, Optional, Union


class QuizTaskRequest(BaseModel):
    """Incoming quiz task request from the evaluation system."""

    email: str
    secret: str
    url: str


class QuizResponse(BaseModel):
    """Response after processing a quiz task."""

    status: str
    message: str


class AnswerSubmission(BaseModel):
    """Payload for submitting answers to the quiz endpoint."""

    email: str
    secret: str
    url: str
    answer: Any  # Can be bool, number, string, base64 URI, or JSON object


class QuizResult(BaseModel):
    """Result from submitting an answer."""

    correct: bool
    url: Optional[str] = None
    reason: Optional[str] = None


class QuizContent(BaseModel):
    """Parsed content from a quiz page."""

    question: str
    submission_url: str
    download_url: Optional[str] = None
    quiz_type: str  # scraping, api, pdf, data, visualization
    raw_html: str
