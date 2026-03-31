"""
Pricing logic for the premium embeddings API.

Standard rate:  $0.001 per 1,000 tokens
Batch rate:     $0.0008 per 1,000 tokens (20% discount)
"""

from typing import List

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STANDARD_RATE_PER_1K_TOKENS: float = 0.001   # USD
BATCH_RATE_PER_1K_TOKENS: float = 0.0008     # USD  (20% discount)
BATCH_DISCOUNT_PERCENT: float = 20.0

# Minimum billable charge per request (avoid dust charges)
MINIMUM_CHARGE: float = 0.000001  # $0.000001 USD

# Approximate characters-per-token ratio for cost estimation
# (OpenAI uses ~4 chars/token for English text)
CHARS_PER_TOKEN: float = 4.0

# Per-model token limits
MODEL_TOKEN_LIMITS: dict = {
    "text-embedding-3-small": 8191,
    "text-embedding-3-large": 8191,
    "text-embedding-ada-002": 8191,
}

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for a string.

    Uses character-based heuristic (4 chars ≈ 1 token) which is accurate
    within ~10% for typical English text. For production use, replace with
    a tiktoken call.
    """
    if not text:
        return 0
    # Simple word-boundary heuristic that performs well on varied text
    word_count = len(text.split())
    char_count = len(text)
    # Blend both estimates; character count typically wins for non-English
    char_estimate = max(1, round(char_count / CHARS_PER_TOKEN))
    word_estimate = max(1, round(word_count * 1.33))
    return max(char_estimate, word_estimate)


def estimate_tokens_batch(texts: List[str]) -> int:
    """Estimate total tokens for a list of texts."""
    return sum(estimate_tokens(t) for t in texts)


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------


def calculate_cost(token_count: int, *, batch: bool = False) -> float:
    """
    Calculate the charge in USD for a given token count.

    Args:
        token_count: Estimated number of tokens to embed.
        batch: Whether to apply the batch discount rate.

    Returns:
        Cost in USD, never below MINIMUM_CHARGE.
    """
    if token_count <= 0:
        return MINIMUM_CHARGE

    rate = BATCH_RATE_PER_1K_TOKENS if batch else STANDARD_RATE_PER_1K_TOKENS
    cost = (token_count / 1000.0) * rate
    return max(cost, MINIMUM_CHARGE)


def estimate_cost(input_: object, *, batch: bool = False) -> float:
    """
    Estimate the cost for an embedding input before generation.

    Accepts a single string, a list of strings, or a nested list (batch).
    """
    if isinstance(input_, str):
        tokens = estimate_tokens(input_)
    elif isinstance(input_, list):
        flat: List[str] = []
        for item in input_:
            if isinstance(item, str):
                flat.append(item)
            elif isinstance(item, list):
                flat.extend(item)
        tokens = estimate_tokens_batch(flat)
    else:
        tokens = 1

    return calculate_cost(tokens, batch=batch)


# ---------------------------------------------------------------------------
# Pricing info helpers
# ---------------------------------------------------------------------------


def pricing_info() -> dict:
    """Return a human-readable pricing summary."""
    return {
        "standard": {
            "rate": f"${STANDARD_RATE_PER_1K_TOKENS:.4f} per 1,000 tokens",
            "rate_numeric": STANDARD_RATE_PER_1K_TOKENS,
            "endpoint": "POST /embeddings",
        },
        "batch": {
            "rate": f"${BATCH_RATE_PER_1K_TOKENS:.4f} per 1,000 tokens",
            "rate_numeric": BATCH_RATE_PER_1K_TOKENS,
            "discount": f"{BATCH_DISCOUNT_PERCENT:.0f}% off standard rate",
            "endpoint": "POST /embeddings/batch",
        },
        "currency": "USD",
        "billing": "Per-call via Mainlayer",
    }
