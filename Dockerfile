FROM mcr.microsoft.com/playwright/python:v1.41.0-jammy

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers (Chromium only for smaller image)
RUN playwright install chromium
RUN playwright install-deps chromium

# Copy application code
COPY . .

# Expose port (Railway will set PORT env var)
EXPOSE 8000

# Run the application
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
