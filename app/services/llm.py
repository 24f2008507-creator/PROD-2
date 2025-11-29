import base64
import json
import logging
import re
from typing import Any, Optional, Union
from openai import AsyncOpenAI

from app.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Service for interacting with OpenAI GPT-4o."""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = "gpt-4o"

    async def analyze_quiz(
        self,
        question: str,
        context: str = "",
        image_base64: Optional[str] = None,
    ) -> Any:
        """
        Analyze a quiz question and return the answer.

        Args:
            question: The quiz question text
            context: Additional context (HTML, data, etc.)
            image_base64: Optional base64 image for vision analysis

        Returns:
            The answer in the appropriate format (bool, number, string, or dict)
        """
        system_prompt = """You are an expert quiz solver. Analyze the question carefully and provide ONLY the answer value.

CRITICAL RULES:
1. Return ONLY the answer value that should go in the "answer" field
2. DO NOT return the entire JSON payload structure - just the answer value itself
3. For numeric answers: return just the number (e.g., 42 or 3.14)
4. For boolean answers: return true or false (lowercase)
5. For string answers: return just the string without quotes
6. For list/object answers: return valid JSON
7. If the question shows an example like "answer": "something", return just: something
8. Be precise and accurate
9. If the question asks for a sum, calculate it exactly
10. DO NOT include explanations, just the answer value

Example: If asked "What is 2+2?" return: 4
Example: If the quiz shows "answer": "anything you want", return: anything you want"""

        messages = [{"role": "system", "content": system_prompt}]

        # Build user message
        user_content = []

        if image_base64:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}",
                        "detail": "high",
                    },
                }
            )

        text_content = f"Question: {question}"
        if context:
            text_content += f"\n\nContext/Data:\n{context}"

        user_content.append({"type": "text", "text": text_content})

        messages.append({"role": "user", "content": user_content})

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=4096,
                temperature=0,
            )

            answer_text = response.choices[0].message.content.strip()
            logger.info(f"LLM response: {answer_text[:200]}...")

            return self._parse_answer(answer_text)

        except Exception as e:
            logger.error(f"LLM error: {e}")
            raise

    def _parse_answer(self, answer_text: str) -> Any:
        """Parse the LLM response into the appropriate type."""
        answer_text = answer_text.strip()

        # Remove markdown code blocks if present
        if answer_text.startswith("```"):
            lines = answer_text.split("\n")
            # Remove first and last lines (``` markers)
            lines = [l for l in lines if not l.startswith("```")]
            answer_text = "\n".join(lines).strip()

        # Check if it looks like a JSON with an 'answer' field - extract just the answer
        if answer_text.startswith("{") and '"answer"' in answer_text:
            try:
                parsed = json.loads(answer_text)
                if isinstance(parsed, dict) and "answer" in parsed:
                    # Extract just the answer value
                    return parsed["answer"]
            except json.JSONDecodeError:
                pass

        # Try boolean
        if answer_text.lower() in ("true", "false"):
            return answer_text.lower() == "true"

        # Try integer
        try:
            if "." not in answer_text and answer_text.lstrip("-").isdigit():
                return int(answer_text)
        except ValueError:
            pass

        # Try float
        try:
            if re.match(r"^-?\d+\.?\d*$", answer_text):
                return float(answer_text)
        except ValueError:
            pass

        # Try JSON
        try:
            return json.loads(answer_text)
        except json.JSONDecodeError:
            pass

        # Return as string
        return answer_text

    async def analyze_with_vision(
        self,
        image_base64: str,
        prompt: str,
    ) -> Any:
        """Analyze an image (screenshot, chart, PDF page) with a specific prompt."""
        messages = [
            {
                "role": "system",
                "content": "You are an expert at analyzing images and extracting information. Provide precise, accurate answers.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=4096,
            temperature=0,
        )

        return response.choices[0].message.content.strip()

    async def extract_structured_data(
        self,
        content: str,
        schema_description: str,
    ) -> dict:
        """Extract structured data from text content."""
        messages = [
            {
                "role": "system",
                "content": f"Extract data according to this schema: {schema_description}. Return valid JSON only.",
            },
            {"role": "user", "content": content},
        ]

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=4096,
            temperature=0,
            response_format={"type": "json_object"},
        )

        return json.loads(response.choices[0].message.content)

    async def solve_data_task(
        self,
        task_description: str,
        data: str,
    ) -> Any:
        """Solve a data analysis task."""
        prompt = f"""Task: {task_description}

Data:
{data}

Analyze the data and provide ONLY the answer. No explanations."""

        return await self.analyze_quiz(prompt)

    async def generate_visualization(
        self,
        data: str,
        chart_type: str,
        requirements: str,
    ) -> str:
        """Generate Python code for visualization, execute it, and return base64 image."""
        # This would generate matplotlib code and execute it
        # For now, return a placeholder
        prompt = f"""Generate Python matplotlib code to create a {chart_type} chart.
Data: {data}
Requirements: {requirements}

Return ONLY the Python code, no explanations."""

        return await self.analyze_quiz(prompt)
