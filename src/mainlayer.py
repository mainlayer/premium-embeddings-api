"""
Mainlayer billing integration.

Mainlayer is the per-call payment infrastructure for AI APIs.
Base URL: https://api.mainlayer.xyz
Auth:     Authorization: Bearer <api_key>

This module handles:
- Payment verification before serving requests
- Charge recording after successful responses
- Payment-required response construction
"""

import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (loaded from environment)
# ---------------------------------------------------------------------------

MAINLAYER_API_KEY: str = os.environ.get("MAINLAYER_API_KEY", "")
MAINLAYER_BASE_URL: str = os.environ.get(
    "MAINLAYER_BASE_URL", "https://api.mainlayer.xyz"
)
MAINLAYER_TIMEOUT_SECONDS: float = float(
    os.environ.get("MAINLAYER_TIMEOUT_SECONDS", "5.0")
)

# Feature flag: set MAINLAYER_ENABLED=false to skip billing in local dev
_ENABLED: bool = os.environ.get("MAINLAYER_ENABLED", "true").lower() != "false"

# ---------------------------------------------------------------------------
# HTTP client (module-level singleton, reused across requests)
# ---------------------------------------------------------------------------

_http_client: Optional[httpx.AsyncClient] = None


def _get_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            base_url=MAINLAYER_BASE_URL,
            headers={
                "Authorization": f"Bearer {MAINLAYER_API_KEY}",
                "Content-Type": "application/json",
                "User-Agent": "premium-embeddings-api/1.0",
            },
            timeout=MAINLAYER_TIMEOUT_SECONDS,
        )
    return _http_client


async def close_client() -> None:
    """Close the shared HTTP client. Call on application shutdown."""
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None


# ---------------------------------------------------------------------------
# Payment verification
# ---------------------------------------------------------------------------


async def check_payment(payer_wallet: Optional[str], amount_usd: float) -> bool:
    """
    Verify that the payer has an active balance sufficient to cover `amount_usd`.

    When MAINLAYER_ENABLED is false (local dev), always returns True.

    Args:
        payer_wallet: The wallet/account identifier from the X-Payer-Wallet header.
        amount_usd:   The estimated cost in USD for this request.

    Returns:
        True if payment is confirmed (or billing is disabled), False otherwise.
    """
    if not _ENABLED:
        logger.debug("Mainlayer billing disabled — skipping payment check.")
        return True

    if not payer_wallet:
        logger.info("Payment check failed: no X-Payer-Wallet header provided.")
        return False

    if not MAINLAYER_API_KEY:
        logger.warning(
            "MAINLAYER_API_KEY not set. Denying request to avoid unbilled usage."
        )
        return False

    try:
        client = _get_client()
        response = await client.post(
            "/v1/verify",
            json={
                "payer": payer_wallet,
                "amount": round(amount_usd, 8),
                "currency": "USD",
            },
        )

        if response.status_code == 200:
            data = response.json()
            authorized: bool = data.get("authorized", False)
            if not authorized:
                logger.info(
                    "Payment not authorized for wallet %s (amount=%.6f USD)",
                    payer_wallet,
                    amount_usd,
                )
            return authorized

        if response.status_code == 402:
            logger.info(
                "Insufficient balance for wallet %s (amount=%.6f USD)",
                payer_wallet,
                amount_usd,
            )
            return False

        logger.warning(
            "Unexpected Mainlayer response: status=%d body=%s",
            response.status_code,
            response.text[:200],
        )
        return False

    except httpx.TimeoutException:
        logger.error(
            "Mainlayer payment check timed out after %.1fs. Denying request.",
            MAINLAYER_TIMEOUT_SECONDS,
        )
        return False

    except httpx.RequestError as exc:
        logger.error("Mainlayer network error during payment check: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Charge recording
# ---------------------------------------------------------------------------


async def record_charge(
    payer_wallet: Optional[str],
    amount_usd: float,
    metadata: Optional[dict] = None,
) -> bool:
    """
    Record a completed charge against the payer's account.

    This is a best-effort call; failures are logged but do not affect the
    already-served response.

    Args:
        payer_wallet: The wallet/account identifier.
        amount_usd:   The actual charge in USD.
        metadata:     Optional dict with request context (model, token_count, etc.).

    Returns:
        True if the charge was recorded successfully, False otherwise.
    """
    if not _ENABLED or not payer_wallet or not MAINLAYER_API_KEY:
        return True

    payload = {
        "payer": payer_wallet,
        "amount": round(amount_usd, 8),
        "currency": "USD",
        "metadata": metadata or {},
    }

    try:
        client = _get_client()
        response = await client.post("/v1/charge", json=payload)
        if response.status_code in (200, 201):
            return True
        logger.warning(
            "Charge recording failed: status=%d body=%s",
            response.status_code,
            response.text[:200],
        )
        return False

    except (httpx.TimeoutException, httpx.RequestError) as exc:
        logger.error("Mainlayer charge recording error: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


def payment_required_body(amount_usd: float) -> dict:
    """
    Build the JSON body for a 402 Payment Required response.

    Keeps messaging generic — no references to specific payment rails.
    """
    return {
        "error": "payment_required",
        "message": (
            "This endpoint requires pre-authorized payment. "
            "Fund your Mainlayer account and include your wallet address "
            "in the X-Payer-Wallet header."
        ),
        "amount_required": round(amount_usd, 6),
        "currency": "USD",
        "payment_endpoint": f"{MAINLAYER_BASE_URL}/v1/fund",
        "documentation_url": "https://docs.mainlayer.xyz/getting-started",
    }
