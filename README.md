# Specter2 Embedding API

This project provides an OpenAI-compatible API for generating embeddings using the [allenai/specter2](https://huggingface.co/allenai/specter2) model. It uses `FastAPI` and supports Docker deployment with `uv` for efficient dependency management.

## Features

- **Model**: `allenai/specter2_base` with `allenai/specter2` adapter (optimized for proximity/retrieval).
- **API**: OpenAI-compatible `/v1/embeddings` endpoint.
- **Batching**: Automatically handles large batch requests (default batch size: 32).
- **Containerization**: Docker support with preloaded model.

## Quick Start

### Run Locally

1. Install `uv` (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Run the server:
   ```bash
   uv run uvicorn app.main:app --reload
   ```

4. Test the API:
   ```bash
   curl http://localhost:8000/v1/embeddings \
     -H "Content-Type: application/json" \
     -d '{
       "input": "This is a scientific paper abstract.",
       "model": "specter2"
     }'
   ```

### Build and Run with Docker

1. **Build the image**:
   ```bash
   docker build -t specter2-embed-server .
   ```
   *Note: This will download the model (~1-2GB) and package it into the image.*

2. **Run the container**:
   ```bash
   # Map port 8000 inside container to port 8000 on host
   docker run -p 8000:8000 specter2-embed-server
   ```
   
   To run multiple models on different ports (e.g., port 8001):
   ```bash
   docker run -p 8001:8000 specter2-embed-server
   ```

3. **Clean up old images** (Optional):
   If you rebuild the image, you can remove the old dangling images with:
   ```bash
   docker image prune -f
   ```

## API Usage

### Single Input
**Endpoint**: `POST /v1/embeddings`

```json
{
  "input": "Your text here",
  "model": "specter2"
}
```

### Batch Input
You can send a list of texts. The server automatically batches processing (default: 32 items per batch) to manage memory usage.

```json
{
  "input": [
    "Paper 1: Attention is all you need",
    "Paper 2: BERT: Pre-training of Deep Bidirectional Transformers",
    "..."
  ],
  "model": "specter2"
}
```

**Response**:
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.1, 0.2, ...],
      "index": 0
    },
    {
      "object": "embedding",
      "embedding": [0.3, 0.4, ...],
      "index": 1
    }
  ],
  "model": "specter2",
  "usage": {
    "prompt_tokens": 128,
    "total_tokens": 128
  }
}
```
