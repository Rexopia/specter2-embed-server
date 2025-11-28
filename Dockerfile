FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 
ENV HF_HOME=/app/model_cache

# Copy project definition
COPY pyproject.toml uv.lock ./

# Install dependencies
# --system installs directly into the system Python environment
RUN uv pip install --system .

# Copy preload script and app code
COPY preload.py .
COPY app ./app

# Preload model during build
RUN python preload.py

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
