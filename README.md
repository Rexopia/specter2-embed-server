# Specter2 Embedding Server

An OpenAI-compatible embedding API server for [AllenAI's Specter 2.0](https://github.com/allenai/specter), designed for generating high-quality embeddings of scientific documents.

This server wraps the `allenai/specter2_base` model (with the `proximity` adapter) in a FastAPI application, exposing a drop-in replacement for OpenAI's embedding API (`/v1/embeddings`). This allows you to easily integrate scientific document embeddings into existing RAG (Retrieval-Augmented Generation) pipelines or semantic search applications.

## Features

- 🚀 **OpenAI Compatible**: Fully compatible with the `/v1/embeddings` endpoint format. Use it with the official OpenAI Python/Node.js SDKs, LangChain, or LlamaIndex.
- 🐳 **Docker Ready**: Pre-built Docker configuration with model caching support for easy deployment.
- ⚡ **High Performance**: Supports GPU acceleration (CUDA) automatically if available, with efficient batch processing.
- 🧠 **Specter 2.0**: Uses the state-of-the-art Specter 2.0 model, optimized for scientific literature similarity and retrieval.

## Quick Start

### Option 1: Docker (Recommended)

The easiest way to run the server is using Docker. This ensures all dependencies and model weights are handled correctly.

1. **Build the image:**
   ```bash
   docker build -t specter2-embed-server .
   ```
   *(Note: The first build will download the model weights, which may take a few minutes.)*

2. **Run the container:**
   ```bash
   docker run -p 8000:8000 --gpus all specter2-embed-server
   ```
   *(Remove `--gpus all` if you are running on a machine without NVIDIA GPUs.)*

### Option 2: Local Development

If you prefer to run the Python code directly:

1. **Install dependencies** (using `uv` for fast package management):
   ```bash
   # Install uv if needed
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Install dependencies
   uv sync
   ```

2. **Run the server:**
   ```bash
   uv run uvicorn app.main:app --reload --port 8000
   ```

## Usage

### Using `curl`

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "The impact of attention mechanisms in deep learning.",
    "model": "specter2"
  }'
```

### Using OpenAI Python SDK

You can use the standard `openai` library by changing the `base_url`.

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy",  # No API key required
    base_url="http://localhost:8000/v1"
)

response = client.embeddings.create(
    model="specter2",
    input="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
)

print(f"Embedding dimension: {len(response.data[0].embedding)}")
# Output: Embedding dimension: 768
```

### Using with LangChain

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="specter2",
    openai_api_base="http://localhost:8000/v1",
    openai_api_key="dummy"
)

vector = embeddings.embed_query("Scientific document retrieval")
```

## API Reference

### `POST /v1/embeddings`

Generates embeddings for the input text.

**Request Body:**
- `input`: string or array of strings. The text(s) to embed.
- `model`: string (optional). ID of the model to use (default: "specter2").
- `user`: string (optional).

**Response:**
Returns a standard OpenAI embedding response object containing the embedding vector(s) (768 dimensions for Specter 2).

### `GET /health`

Health check endpoint to verify the service status and model loading.

## System Requirements

- **RAM**: At least 4GB of RAM is recommended.
- **GPU**: Optional but recommended for higher throughput (NVIDIA GPU with CUDA). The server automatically detects CUDA availability.
