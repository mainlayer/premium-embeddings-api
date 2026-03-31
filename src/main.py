"""
Premium Embeddings API — FastAPI application.

OpenAI-compatible embeddings with per-call billing via Mainlayer.
Pricing: $0.001 per 1,000 tokens (standard) / $0.0008 per 1,000 tokens (batch).
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.embeddings import generate_embedding_response
from src.mainlayer import (
    check_payment,
    close_client,
    payment_required_body,
    record_charge,
)
from src.models import (
    BatchEmbeddingRequest,
    BatchEmbeddingResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    UsageStats,
)
from src.pricing import calculate_cost, estimate_cost, estimate_tokens, pricing_info

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Premium Embeddings API starting up.")
    yield
    logger.info("Premium Embeddings API shutting down.")
    await close_client()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Premium Embeddings API",
    description=(
        "OpenAI-compatible text embeddings with per-call billing via Mainlayer. "
        "Drop-in replacement for the OpenAI embeddings endpoint. "
        "Pricing: $0.001/1K tokens (standard), $0.0008/1K tokens (batch)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _payer_from_header(x_payer_wallet: Optional[str]) -> Optional[str]:
    """Return the payer wallet string, stripping whitespace."""
    if x_payer_wallet:
        return x_payer_wallet.strip() or None
    return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "Premium Embeddings API",
        "version": "1.0.0",
        "docs": "/docs",
        "pricing": "/pricing",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "premium-embeddings-api"}


@app.get("/pricing")
async def get_pricing():
    """Return current pricing information."""
    return pricing_info()


@app.get("/models")
async def list_models():
    """List available embedding models."""
    return {
        "object": "list",
        "data": [
            {
                "id": "text-embedding-3-small",
                "object": "model",
                "dimensions": 1536,
                "description": "Efficient embedding model, 1536 dimensions.",
            },
            {
                "id": "text-embedding-3-large",
                "object": "model",
                "dimensions": 3072,
                "description": "High-quality embedding model, 3072 dimensions.",
            },
            {
                "id": "text-embedding-ada-002",
                "object": "model",
                "dimensions": 1536,
                "description": "Legacy ada-002 compatible model, 1536 dimensions.",
            },
        ],
    }


@app.post(
    "/embeddings",
    response_model=EmbeddingResponse,
    responses={
        200: {"description": "Embeddings generated successfully."},
        402: {"description": "Payment required — insufficient balance."},
        422: {"description": "Validation error in request body."},
    },
    summary="Create embeddings",
    tags=["Embeddings"],
)
async def create_embeddings(
    request: EmbeddingRequest,
    x_payer_wallet: Optional[str] = Header(None, description="Your Mainlayer wallet address."),
):
    """
    Generate embeddings for one or more texts.

    **Pricing**: $0.001 per 1,000 tokens.

    **Drop-in replacement** for `POST https://api.openai.com/v1/embeddings` —
    use the same request/response schema.

    Payment is verified before generation via Mainlayer. Include your wallet
    address in the `X-Payer-Wallet` header. The actual token count is charged
    after a successful response.
    """
    payer = _payer_from_header(x_payer_wallet)
    texts = request.get_texts()
    estimated_cost = estimate_cost(texts)

    # --- Payment check ---
    authorized = await check_payment(payer, estimated_cost)
    if not authorized:
        return JSONResponse(
            status_code=402,
            content=payment_required_body(estimated_cost),
        )

    # --- Generate embeddings ---
    response = await generate_embedding_response(request)

    # --- Record actual charge ---
    actual_cost = calculate_cost(response.usage.total_tokens)
    await record_charge(
        payer,
        actual_cost,
        metadata={
            "model": request.model,
            "token_count": response.usage.total_tokens,
            "input_count": len(texts),
            "endpoint": "/embeddings",
        },
    )

    logger.info(
        "Embeddings generated: model=%s inputs=%d tokens=%d cost=$%.6f payer=%s",
        request.model,
        len(texts),
        response.usage.total_tokens,
        actual_cost,
        payer or "anonymous",
    )

    return response


@app.post(
    "/embeddings/batch",
    response_model=BatchEmbeddingResponse,
    responses={
        200: {"description": "Batch embeddings generated successfully."},
        402: {"description": "Payment required — insufficient balance."},
        422: {"description": "Validation error in request body."},
    },
    summary="Create batch embeddings",
    tags=["Embeddings"],
)
async def batch_embeddings(
    batch_request: BatchEmbeddingRequest,
    x_payer_wallet: Optional[str] = Header(None, description="Your Mainlayer wallet address."),
):
    """
    Generate embeddings for multiple independent requests in a single call.

    **Pricing**: $0.0008 per 1,000 tokens (20% discount vs. standard rate).

    Ideal for bulk processing where you have many independent texts that
    don't need to be in the same request.
    """
    payer = _payer_from_header(x_payer_wallet)

    # Estimate total cost across all sub-requests at batch rate
    all_texts: List[str] = []
    for sub in batch_request.requests:
        all_texts.extend(sub.get_texts())

    estimated_cost = estimate_cost(all_texts, batch=True)

    # --- Payment check ---
    authorized = await check_payment(payer, estimated_cost)
    if not authorized:
        return JSONResponse(
            status_code=402,
            content=payment_required_body(estimated_cost),
        )

    # --- Generate all embeddings ---
    results: List[EmbeddingResponse] = []
    total_prompt_tokens = 0
    total_tokens = 0

    for sub_request in batch_request.requests:
        response = await generate_embedding_response(sub_request)
        results.append(response)
        total_prompt_tokens += response.usage.prompt_tokens
        total_tokens += response.usage.total_tokens

    # --- Record actual charge at batch rate ---
    actual_cost = calculate_cost(total_tokens, batch=True)
    await record_charge(
        payer,
        actual_cost,
        metadata={
            "batch_size": len(batch_request.requests),
            "total_token_count": total_tokens,
            "input_count": len(all_texts),
            "endpoint": "/embeddings/batch",
        },
    )

    logger.info(
        "Batch embeddings generated: requests=%d inputs=%d tokens=%d cost=$%.6f payer=%s",
        len(batch_request.requests),
        len(all_texts),
        total_tokens,
        actual_cost,
        payer or "anonymous",
    )

    return BatchEmbeddingResponse(
        results=results,
        total_usage=UsageStats(
            prompt_tokens=total_prompt_tokens,
            total_tokens=total_tokens,
        ),
        batch_size=len(batch_request.requests),
    )


# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "type": "api_error",
                "code": str(exc.status_code),
            }
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "An internal server error occurred.",
                "type": "internal_server_error",
                "code": "500",
            }
        },
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        reload=os.environ.get("RELOAD", "false").lower() == "true",
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
    )
