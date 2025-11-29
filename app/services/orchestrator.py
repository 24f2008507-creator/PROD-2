import asyncio
import logging
from typing import Any, List, Optional

from app.services.browser import BrowserService
from app.services.quiz_parser import QuizParser
from app.services.llm import LLMService
from app.services.submitter import AnswerSubmitter
from app.handlers.pdf import PDFHandler
from app.handlers.data import DataHandler
from app.handlers.scraper import ScraperHandler
from app.models import QuizContent, QuizResult

logger = logging.getLogger(__name__)


class QuizOrchestrator:
    """Orchestrates the quiz solving workflow."""

    def __init__(self, email: str, secret: str):
        self.email = email
        self.secret = secret
        self.llm = LLMService()
        self.submitter = AnswerSubmitter(email, secret)
        self.pdf_handler = PDFHandler(self.llm)
        self.data_handler = DataHandler(self.llm)
        self.scraper_handler = ScraperHandler(self.llm)
        self.results: List[QuizResult] = []

    async def solve_quiz_chain(self, initial_url: str, max_questions: int = 20):
        """
        Solve a chain of quizzes, following next URLs until complete.

        Args:
            initial_url: The first quiz URL
            max_questions: Maximum number of questions to solve
        """
        current_url = initial_url
        question_count = 0

        while current_url and question_count < max_questions:
            question_count += 1
            logger.info(f"=== Solving question {question_count}: {current_url} ===")

            try:
                result = await self.solve_single_quiz(current_url)
                self.results.append(result)

                if result.correct:
                    logger.info(f"Answer correct! Next URL: {result.url}")
                else:
                    logger.warning(f"Answer incorrect. Reason: {result.reason}")
                    # Still try to proceed if there's a next URL
                    if result.url:
                        logger.info(f"Proceeding to next question: {result.url}")

                current_url = result.url

            except Exception as e:
                logger.error(f"Error solving quiz {current_url}: {e}")
                break

        logger.info(f"Quiz chain complete. Solved {question_count} questions.")
        await self.submitter.close()

    async def solve_single_quiz(self, url: str) -> QuizResult:
        """
        Solve a single quiz question.

        Args:
            url: The quiz URL

        Returns:
            QuizResult with the submission outcome
        """
        # Step 1: Get and render the page content
        logger.info("Step 1: Fetching and rendering page...")
        page_data = await BrowserService.execute_and_get_content(url)
        html_content = page_data["rendered_html"]
        screenshot = page_data["screenshot_base64"]

        # Step 2: Parse the quiz content
        logger.info("Step 2: Parsing quiz content...")
        quiz_content = QuizParser.parse_quiz_page(html_content, url)
        logger.info(f"Quiz type: {quiz_content.quiz_type}")
        logger.info(f"Question: {quiz_content.question[:200]}...")
        logger.info(f"Submission URL: {quiz_content.submission_url}")

        if not quiz_content.submission_url:
            logger.error("No submission URL found!")
            return QuizResult(correct=False, reason="No submission URL found")

        # Make submission URL absolute if needed
        submission_url = QuizParser.make_absolute_url(quiz_content.submission_url, url)
        logger.info(f"Absolute submission URL: {submission_url}")

        # Step 3: Solve based on quiz type
        logger.info(f"Step 3: Solving quiz (type: {quiz_content.quiz_type})...")
        answer = await self._solve_by_type(quiz_content, screenshot, url)
        logger.info(f"Computed answer: {answer}")

        # Step 4: Submit the answer
        logger.info("Step 4: Submitting answer...")
        result = await self.submitter.submit_answer(
            submission_url,
            url,
            answer,
        )

        # Step 5: Handle retry if incorrect
        if not result.correct and result.url is None:
            logger.info("Answer incorrect, attempting retry with different approach...")
            # Try with vision analysis
            answer = await self._solve_with_vision(quiz_content, screenshot)
            result = await self.submitter.submit_answer(
                submission_url,
                url,
                answer,
            )

        return result

    async def _solve_by_type(
        self,
        quiz_content: QuizContent,
        screenshot: str,
        base_url: str,
    ) -> Any:
        """Solve the quiz based on its type."""

        if quiz_content.quiz_type == "pdf" and quiz_content.download_url:
            download_url = QuizParser.make_absolute_url(quiz_content.download_url, base_url)
            return await self.pdf_handler.solve(
                quiz_content.question,
                download_url,
            )

        if quiz_content.quiz_type == "data":
            download_url = None
            if quiz_content.download_url:
                download_url = QuizParser.make_absolute_url(quiz_content.download_url, base_url)

            # For data quizzes, also use vision to understand the question better
            vision_analysis = await self.llm.analyze_with_vision(
                screenshot,
                "Describe this quiz page. What exactly is the question asking? What data operation is needed? What format should the answer be in?"
            )

            # Combine the question with vision analysis for better context
            enhanced_question = f"{quiz_content.question}\n\nPage Analysis: {vision_analysis}"

            return await self.data_handler.solve(
                enhanced_question,
                download_url,
                quiz_content.raw_html,
            )

        if quiz_content.quiz_type == "scraping":
            return await self._solve_scraping_task(quiz_content, base_url, screenshot)

        if quiz_content.quiz_type == "api":
            return await self._solve_api_task(quiz_content, base_url)

        # Default: use LLM with both text and vision
        return await self._solve_with_vision(quiz_content, screenshot)

    async def _solve_scraping_task(
        self,
        quiz_content: QuizContent,
        base_url: str,
        screenshot: str,
    ) -> Any:
        """Solve a web scraping task by actually visiting and scraping the URL."""
        import re

        # Extract the URL to scrape from the question
        question = quiz_content.question
        raw_html = quiz_content.raw_html

        # Look for relative URLs to scrape in the question
        url_patterns = [
            r'[Ss]crape\s+(/[^\s<>"\']+)',
            r'[Vv]isit\s+(/[^\s<>"\']+)',
            r'[Gg]et.*?from\s+(/[^\s<>"\']+)',
            r'href="(/[^"]+)"',
        ]

        scrape_url = None
        for pattern in url_patterns:
            match = re.search(pattern, question + raw_html)
            if match:
                scrape_url = match.group(1)
                break

        if scrape_url:
            # Make absolute
            absolute_url = QuizParser.make_absolute_url(scrape_url, base_url)
            logger.info(f"Scraping URL: {absolute_url}")

            # Actually visit and scrape the page
            scraped_content = await BrowserService.get_element_content(absolute_url)
            logger.info(f"Scraped content: {scraped_content[:500]}...")

            # Use LLM to analyze the scraped content and answer the question
            answer = await self.llm.analyze_quiz(
                question=quiz_content.question,
                context=f"Scraped content from {absolute_url}:\n{scraped_content}",
            )
            return answer

        # Fallback to vision if we can't find a URL to scrape
        return await self._solve_with_vision(quiz_content, screenshot)

    async def _solve_with_vision(
        self,
        quiz_content: QuizContent,
        screenshot: str,
    ) -> Any:
        """Solve using GPT-4o vision capabilities."""
        prompt = f"""Look at this quiz page screenshot and answer the question.

Question from page:
{quiz_content.question}

Instructions:
1. Analyze the screenshot carefully
2. Follow any instructions in the quiz
3. Compute the exact answer requested
4. Return ONLY the answer value (number, string, boolean, or JSON)
5. Do NOT include explanations"""

        answer = await self.llm.analyze_with_vision(screenshot, prompt)
        return self.llm._parse_answer(answer)

    async def _solve_api_task(self, quiz_content: QuizContent, base_url: str) -> Any:
        """Solve an API-related task."""
        # Extract API details from the question
        return await self.llm.analyze_quiz(
            quiz_content.question,
            context=quiz_content.raw_html,
        )
