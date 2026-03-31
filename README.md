# Premium Embeddings API

OpenAI-compatible embeddings API with **per-token billing** via [Mainlayer](https://mainlayer.fr).

This is a production-ready drop-in replacement for the OpenAI embeddings endpoint with automatic per-call billing and volume discounts.

## Features

- OpenAI-compatible request/response schema (`POST /embeddings`)
- Per-call billing: **$0.001 per 1,000 tokens** (standard) / **$0.0008 per 1,000 tokens** (batch)
- Deterministic mock embeddings for testing and development
- Batch endpoint for bulk processing (20% discount)
- Mainlayer payment integration with HTTP 402 Payment Required
- Full async support with FastAPI
- Ready for OpenAI SDK (`client.embeddings.create(...)`)

## 5-Minute Quickstart

### 1. Prerequisites

```bash
git clone <this-repo>
cd premium-embeddings-api
python -m venv venv
source venv/bin/activate  # or: .\venv\Scripts\activate on Windows
```

### 2. Install and configure

```bash
pip install -r requirements.txt
cp .env.example .env

# Edit .env with your Mainlayer credentials:
# MAINLAYER_API_KEY=sk_live_...
# MAINLAYER_ENABLED=true  # or 'false' for local testing
```

### 3. Run the server

```bash
# Development with auto-reload
uvicorn src.main:app --reload

# Open http://localhost:8000/docs for interactive API docs
```

### 4. Generate embeddings

```python
import httpx

client = httpx.Client()

# Create embeddings for a single text
response = client.post(
    "http://localhost:8000/embeddings",
    json={
        "model": "text-embedding-3-small",
        "input": "Hello, world!",
    },
    headers={"X-Payer-Wallet": "user_123"}  # Your Mainlayer wallet
)

print(response.json())
# {
#   "data": [
#       {"embedding": [...1536 floats...], "index": 0}
#   ],
#   "model": "text-embedding-3-small",
#   "usage": {"prompt_tokens": 3, "total_tokens": 3}
# }
```

## API Endpoints

### `POST /embeddings`

OpenAI-compatible embeddings endpoint with per-call billing.

**Request:**
```json
{
  "model": "text-embedding-3-small",
  "input": "Hello, world!"
}
```

**Headers:**
- `X-Payer-Wallet` (required) — Your Mainlayer wallet address
- `Content-Type: application/json`

**Response (200 OK):**
```json
{
  "data": [
    {"embedding": [...], "index": 0}
  ],
  "model": "text-embedding-3-small",
  "usage": {"prompt_tokens": 3, "total_tokens": 3}
}
```

**Error (402 Payment Required):**
```json
{
  "error": "payment_required",
  "message": "Insufficient balance. Fund your account at mainlayer.fr",
  "amount_required": 0.000003,
  "currency": "USD"
}
```

### `POST /embeddings/batch`

Bulk embeddings endpoint with 20% volume discount ($0.0008 per 1K tokens).

**Request:**
```json
{
  "requests": [
    {
      "model": "text-embedding-3-small",
      "input": "Text 1"
    },
    {
      "model": "text-embedding-3-large",
      "input": ["Text 2a", "Text 2b"]
    }
  ]
}
```

**Response (200 OK):**
```json
{
  "results": [
    {"data": [...], "model": "...", "usage": {...}},
    {"data": [...], "model": "...", "usage": {...}}
  ],
  "total_usage": {"prompt_tokens": N, "total_tokens": N},
  "batch_size": 2
}
```

### `GET /pricing`

Get current pricing information.

**Response:**
```json
{
  "models": {
    "text-embedding-3-small": {
      "dimensions": 1536,
      "pricing_per_1k_tokens": 0.001
    },
    "text-embedding-3-large": {
      "dimensions": 3072,
      "pricing_per_1k_tokens": 0.001
    }
  },
  "batch_discount_percent": 20,
  "batch_pricing_per_1k_tokens": 0.0008
}
```

### `GET /models`

List available embedding models.

## Available Models

| Model | Dimensions | Speed | Quality | Cost |
|-------|-----------|-------|---------|------|
| `text-embedding-3-small` | 1536 | Fast | Good | $0.001/1K tokens |
| `text-embedding-3-large` | 3072 | Slower | Excellent | $0.001/1K tokens |
| `text-embedding-ada-002` | 1536 | Fast | Good | $0.001/1K tokens |

All models produce deterministic, L2-normalized embeddings.

## Pricing

| Endpoint | Rate | Best for |
|----------|------|----------|
| `/embeddings` | $0.001 per 1,000 tokens | Single requests |
| `/embeddings/batch` | $0.0008 per 1,000 tokens | Bulk processing (20% discount) |

**Example costs:**
- 1 request, 100 tokens: $0.0001
- 100 requests, 100 tokens each: $0.01
- 1 batch, 10,000 tokens: $0.008 (vs. $0.01 standard)

## Using with OpenAI SDK

The API is fully OpenAI-compatible. Update your OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy",  # Not used for billing; use X-Payer-Wallet header instead
    base_url="http://localhost:8000",
)

# Note: OpenAI SDK doesn't support custom headers out of the box
# For production, use httpx or requests directly (see examples above)
```

## Local Development (no payment)

For testing without Mainlayer integration:

```bash
MAINLAYER_ENABLED=false uvicorn src.main:app --reload

# Now any request works, even without X-Payer-Wallet header
curl http://localhost:8000/embeddings \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"model": "text-embedding-3-small", "input": "hello"}'
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MAINLAYER_API_KEY` | Yes* | — | Your Mainlayer API key |
| `MAINLAYER_ENABLED` | No | `true` | Set to `false` for local development |
| `MAINLAYER_BASE_URL` | No | `https://api.mainlayer.fr` | Override Mainlayer endpoint |
| `MAINLAYER_TIMEOUT_SECONDS` | No | `5` | Timeout for Mainlayer API calls |
| `LOG_LEVEL` | No | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `HOST` | No | `0.0.0.0` | Server bind address |
| `PORT` | No | `8000` | Server port |
| `CORS_ORIGINS` | No | `*` | Allowed CORS origins (comma-separated) |

*Required only if `MAINLAYER_ENABLED=true`

## Running Tests

```bash
pytest tests/ -v

# With coverage
pytest tests/ --cov=src/
```

## Deployment

### Railway

```bash
railway up
```

### Docker

```bash
docker build -t embeddings-api .
docker run -e MAINLAYER_API_KEY=sk_... -p 8000:8000 embeddings-api
```

### Fly.io

```bash
fly launch
fly deploy
```

## Internals

- **`src/main.py`**: FastAPI app with payment middleware
- **`src/embeddings.py`**: Deterministic vector generation (no external model required)
- **`src/mainlayer.py`**: Mainlayer payment integration (check_payment, record_charge)
- **`src/pricing.py`**: Token estimation and cost calculation
- **`src/models.py`**: Pydantic models (OpenAI-compatible schemas)

## Example: Batch Embeddings

```python
import httpx

client = httpx.Client()

response = client.post(
    "http://localhost:8000/embeddings/batch",
    json={
        "requests": [
            {
                "model": "text-embedding-3-small",
                "input": ["Text 1", "Text 2"],
            },
            {
                "model": "text-embedding-3-large",
                "input": "Text 3",
            },
        ]
    },
    headers={"X-Payer-Wallet": "user_123"},
)

data = response.json()
print(f"Processed {data['batch_size']} sub-requests")
print(f"Total tokens: {data['total_usage']['total_tokens']}")
print(f"Cost at batch rate: ${data['total_usage']['total_tokens'] * 0.0008 / 1000:.6f}")
```

## Support

- Docs: [docs.mainlayer.fr](https://docs.mainlayer.fr)
- Issues: GitHub issues on this repository
- Community: [mainlayer.fr/discord](https://mainlayer.fr/discord)
