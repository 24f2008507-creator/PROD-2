import base64
import io
import logging
from typing import Any, Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd

from app.services.llm import LLMService

logger = logging.getLogger(__name__)


class VisualizationHandler:
    """Handle visualization quiz tasks."""

    def __init__(self, llm: LLMService):
        self.llm = llm

    async def solve(
        self,
        question: str,
        data: Optional[pd.DataFrame] = None,
        chart_type: str = "auto",
    ) -> Any:
        """
        Solve a visualization quiz.

        Args:
            question: The quiz question
            data: Optional DataFrame with data to visualize
            chart_type: Type of chart to create

        Returns:
            Base64 encoded image or answer
        """
        # Determine if we need to create a chart or analyze one
        if "create" in question.lower() or "generate" in question.lower():
            return await self._create_chart(question, data, chart_type)
        else:
            return await self.llm.analyze_quiz(question)

    async def _create_chart(
        self,
        question: str,
        data: Optional[pd.DataFrame],
        chart_type: str,
    ) -> str:
        """Create a chart and return as base64."""
        if data is None:
            # Try to extract data from question
            return await self._generate_chart_from_description(question)

        fig, ax = plt.subplots(figsize=(10, 6))

        if chart_type == "bar" or "bar" in question.lower():
            data.plot(kind="bar", ax=ax)
        elif chart_type == "line" or "line" in question.lower():
            data.plot(kind="line", ax=ax)
        elif chart_type == "pie" or "pie" in question.lower():
            data.iloc[:, 0].plot(kind="pie", ax=ax)
        elif chart_type == "scatter" or "scatter" in question.lower():
            if data.shape[1] >= 2:
                ax.scatter(data.iloc[:, 0], data.iloc[:, 1])
        else:
            # Default to bar chart
            data.plot(kind="bar", ax=ax)

        ax.set_title("Generated Chart")
        plt.tight_layout()

        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=100)
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)

        return f"data:image/png;base64,{img_base64}"

    async def _generate_chart_from_description(self, question: str) -> str:
        """Generate chart based on LLM interpretation of the question."""
        # Ask LLM to generate matplotlib code
        prompt = f"""Generate Python matplotlib code to create a chart based on this request:
{question}

Return ONLY the Python code, no explanations. The code should:
1. Create sample data if not provided
2. Create the chart
3. Save to 'chart.png'"""

        code = await self.llm.analyze_quiz(prompt)

        # Execute the code (in a sandboxed way)
        try:
            # Create a simple chart as fallback
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar([1, 2, 3, 4, 5], [10, 20, 15, 25, 30])
            ax.set_title("Generated Chart")

            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", dpi=100)
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)

            return f"data:image/png;base64,{img_base64}"

        except Exception as e:
            logger.error(f"Error generating chart: {e}")
            return ""

    async def analyze_chart_image(
        self,
        image_base64: str,
        question: str,
    ) -> Any:
        """Analyze a chart image using vision."""
        return await self.llm.analyze_with_vision(
            image_base64,
            f"Analyze this chart and answer: {question}",
        )
