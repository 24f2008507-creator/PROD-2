import io
import json
import logging
import re
from typing import Any, Optional, Tuple

import httpx
import pandas as pd

from app.services.llm import LLMService

logger = logging.getLogger(__name__)


class DataHandler:
    """Handle data analysis quiz tasks."""

    def __init__(self, llm: LLMService):
        self.llm = llm
        self.client = httpx.AsyncClient(timeout=60.0)

    async def solve(
        self,
        question: str,
        download_url: Optional[str],
        raw_html: str,
    ) -> Any:
        """
        Solve a data-related quiz.

        Args:
            question: The quiz question
            download_url: Optional URL to download data file
            raw_html: The raw HTML content (may contain inline data)

        Returns:
            The computed answer
        """
        df = None
        data_content = ""

        # Try to get data from download URL as DataFrame
        if download_url:
            try:
                df, data_content = await self._fetch_data_as_df(download_url)
                logger.info(f"Loaded DataFrame with shape: {df.shape if df is not None else 'None'}")
            except Exception as e:
                logger.error(f"Error fetching data: {e}")

        # If no data from URL, try to extract from HTML
        if not data_content:
            data_content = self._extract_data_from_html(raw_html)

        # Try to compute answer directly if we have a DataFrame and detect a cutoff pattern
        if df is not None:
            direct_answer = await self._try_direct_computation(question, df)
            if direct_answer is not None:
                logger.info(f"Computed answer directly: {direct_answer}")
                return direct_answer

        # Fall back to LLM analysis
        enhanced_prompt = f"""Analyze this data-related quiz carefully.

QUIZ QUESTION:
{question}

DATA FROM FILE:
{data_content[:10000]}  # Truncated if too long

CRITICAL INSTRUCTIONS:
1. Read the question VERY carefully to understand what answer format is expected
2. MOST IMPORTANT: Return a SINGLE value, not an array or list, unless explicitly asked for a list
3. If the question mentions a CUTOFF value and data, you need to:
   - Filter the data to find values GREATER THAN the cutoff
   - Calculate the SUM of those filtered values
   - Return just that sum as a single number
4. Keywords that indicate COUNT (return a number):
   - "how many" = return COUNT as a single number
   - "number of" = return COUNT as a single number
   - "count" = return COUNT as a single number
5. Keywords that indicate SUM (return a number):
   - "sum", "total", "add up", "cutoff" = return SUM as a single number
6. Return ONLY the answer value - NO explanations

EXAMPLE for cutoff questions:
- If cutoff is 100 and data is [50, 150, 200, 80], the answer is 150 + 200 = 350"""

        answer = await self.llm.analyze_quiz(
            question=enhanced_prompt,
            context="",  # Already included in prompt
        )

        return answer

    async def _try_direct_computation(self, question: str, df: pd.DataFrame) -> Optional[Any]:
        """Try to compute the answer directly using pandas if we detect a pattern."""
        question_lower = question.lower()

        # Detect cutoff/threshold pattern
        cutoff_match = re.search(r'cutoff[:\s]*(\d+)', question_lower)
        if not cutoff_match:
            cutoff_match = re.search(r'threshold[:\s]*(\d+)', question_lower)
        if not cutoff_match:
            cutoff_match = re.search(r'greater than[:\s]*(\d+)', question_lower)
        if not cutoff_match:
            cutoff_match = re.search(r'above[:\s]*(\d+)', question_lower)

        if cutoff_match:
            cutoff = int(cutoff_match.group(1))
            logger.info(f"Detected cutoff value: {cutoff}")

            # Get numeric values from DataFrame
            numeric_values = []
            for col in df.columns:
                try:
                    numeric_values.extend(pd.to_numeric(df[col], errors='coerce').dropna().tolist())
                except:
                    pass

            # If DataFrame has no columns (single column CSV), try the first column
            if not numeric_values and len(df.columns) == 1:
                try:
                    numeric_values = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().tolist()
                except:
                    pass

            # Also try if the DataFrame index contains numbers (headerless CSV)
            if not numeric_values:
                try:
                    # Re-read as headerless
                    numeric_values = df.iloc[:, 0].tolist()
                except:
                    pass

            if numeric_values:
                # Filter values greater than cutoff and sum
                filtered = [v for v in numeric_values if isinstance(v, (int, float)) and v > cutoff]
                result = sum(filtered)
                logger.info(f"Filtered {len(filtered)} values > {cutoff}, sum = {result}")
                return int(result) if result == int(result) else result

        return None

    async def _fetch_data_as_df(self, url: str) -> Tuple[Optional[pd.DataFrame], str]:
        """Fetch data from URL and return both DataFrame and string representation."""
        response = await self.client.get(url)
        response.raise_for_status()
        content = response.content

        df = None
        data_str = ""

        # Determine file type from URL or content
        if ".csv" in url.lower():
            # Try with header first
            try:
                df = pd.read_csv(io.BytesIO(content))
                # Check if first row looks like data (all numeric)
                if df.shape[1] == 1 and pd.to_numeric(df.columns[0], errors='coerce') is not pd.NA:
                    # Headerless CSV - re-read
                    df = pd.read_csv(io.BytesIO(content), header=None)
            except:
                df = pd.read_csv(io.BytesIO(content), header=None)
            data_str = df.to_string()

        elif ".json" in url.lower():
            data = json.loads(content)
            data_str = json.dumps(data, indent=2)
            if isinstance(data, list):
                df = pd.DataFrame(data)

        elif ".xlsx" in url.lower() or ".xls" in url.lower():
            df = pd.read_excel(io.BytesIO(content))
            data_str = df.to_string()

        else:
            # Try to parse as JSON first, then CSV
            try:
                data = json.loads(content)
                data_str = json.dumps(data, indent=2)
                if isinstance(data, list):
                    df = pd.DataFrame(data)
            except:
                try:
                    df = pd.read_csv(io.BytesIO(content), header=None)
                    data_str = df.to_string()
                except:
                    data_str = content.decode("utf-8", errors="ignore")

        return df, data_str

    async def _fetch_and_parse_data(self, url: str) -> str:
        """Fetch and parse data from URL."""
        response = await self.client.get(url)
        response.raise_for_status()
        content = response.content

        # Determine file type from URL or content
        if ".csv" in url.lower():
            df = pd.read_csv(io.BytesIO(content))
            return df.to_string()

        if ".json" in url.lower():
            data = json.loads(content)
            return json.dumps(data, indent=2)

        if ".xlsx" in url.lower() or ".xls" in url.lower():
            df = pd.read_excel(io.BytesIO(content))
            return df.to_string()

        # Try to parse as JSON first, then CSV
        try:
            data = json.loads(content)
            return json.dumps(data, indent=2)
        except:
            pass

        try:
            df = pd.read_csv(io.BytesIO(content))
            return df.to_string()
        except:
            pass

        # Return as text
        return content.decode("utf-8", errors="ignore")

    def _extract_data_from_html(self, html: str) -> str:
        """Extract data tables or JSON from HTML content."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        data_parts = []

        # Extract tables
        for table in soup.find_all("table"):
            try:
                df = pd.read_html(str(table))[0]
                data_parts.append(df.to_string())
            except Exception as e:
                logger.debug(f"Could not parse table: {e}")

        # Extract pre/code blocks that might contain data
        for elem in soup.find_all(["pre", "code"]):
            text = elem.get_text()
            if text.strip():
                data_parts.append(text)

        # Look for JSON in script tags
        for script in soup.find_all("script"):
            text = script.get_text()
            if "{" in text and "}" in text:
                # Try to extract JSON
                import re

                json_matches = re.findall(r'\{[^{}]+\}', text)
                for match in json_matches:
                    try:
                        data = json.loads(match)
                        data_parts.append(json.dumps(data, indent=2))
                    except:
                        pass

        return "\n\n".join(data_parts) if data_parts else ""

    async def compute_statistics(self, data: pd.DataFrame, operation: str) -> Any:
        """Compute statistics on a DataFrame."""
        operation = operation.lower()

        if "sum" in operation:
            return data.sum().sum()
        if "mean" in operation or "average" in operation:
            return data.mean().mean()
        if "count" in operation:
            return len(data)
        if "max" in operation:
            return data.max().max()
        if "min" in operation:
            return data.min().min()

        return None

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
