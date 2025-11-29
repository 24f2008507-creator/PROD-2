import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.config import settings
from app.models import QuizTaskRequest, QuizResponse
from app.services.browser import BrowserService
from app.services.orchestrator import QuizOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - initialize and cleanup browser."""
    logger.info("Starting up - initializing browser service...")
    await BrowserService.initialize()
    yield
    logger.info("Shutting down - cleaning up browser service...")
    await BrowserService.cleanup()


app = FastAPI(
    title="LLM Analysis Quiz API",
    description="Automated quiz solver using LLMs",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle invalid JSON with HTTP 400."""
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": "Invalid JSON payload", "errors": str(exc.errors())},
    )


@app.get("/")
async def root():
    """Root endpoint - basic info."""
    return {"status": "ok", "message": "LLM Analysis Quiz API"}


@app.get("/health")
async def health_check():
    """Health check endpoint for deployment monitoring."""
    browser_healthy = await BrowserService.is_healthy()
    return {
        "status": "healthy" if browser_healthy else "degraded",
        "browser": browser_healthy,
    }


@app.post("/", response_model=QuizResponse)
@app.post("/api/quiz", response_model=QuizResponse)
async def solve_quiz(task: QuizTaskRequest):
    """
    Main endpoint to receive and solve quiz tasks.

    - Validates the secret
    - Visits the quiz URL
    - Solves the quiz using LLM
    - Submits the answer
    """
    logger.info(f"Received quiz task for URL: {task.url}")

    # Validate secret
    if task.secret != settings.STORED_SECRET:
        logger.warning(f"Invalid secret provided for email: {task.email}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid secret",
        )

    # Start quiz solving in background (don't block the response)
    # The quiz must be solved within 3 minutes of receiving the request
    asyncio.create_task(
        solve_quiz_task(task.email, task.secret, task.url)
    )

    return QuizResponse(
        status="accepted",
        message=f"Quiz task accepted. Solving quiz at {task.url}",
    )


async def solve_quiz_task(email: str, secret: str, url: str):
    """Background task to solve the quiz."""
    try:
        orchestrator = QuizOrchestrator(email, secret)
        await asyncio.wait_for(
            orchestrator.solve_quiz_chain(url),
            timeout=settings.QUIZ_TIMEOUT_SECONDS - 10,  # 10s buffer
        )
        logger.info(f"Quiz chain completed for {url}")
    except asyncio.TimeoutError:
        logger.error(f"Quiz solving timed out for {url}")
    except Exception as e:
        logger.error(f"Error solving quiz {url}: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
