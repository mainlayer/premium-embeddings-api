"""
OpenAI-compatible request/response models for the embeddings API.
"""

from typing import List, Optional, Union
from pydantic import BaseModel, Field, validator


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request."""

    input: Union[str, List[str]] = Field(
        ...,
        description="Text(s) to embed. Can be a single string or a list of strings.",
    )
    model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model to use.",
    )
    encoding_format: str = Field(
        default="float",
        description="Format for the returned embeddings. Only 'float' is supported.",
    )
    dimensions: Optional[int] = Field(
        default=None,
        description="Number of dimensions for the output embeddings. Defaults to model maximum.",
    )
    user: Optional[str] = Field(
        default=None,
        description="Optional unique identifier representing the end-user.",
    )

    @validator("input")
    def validate_input(cls, v):
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("Input string cannot be empty.")
            return v
        if isinstance(v, list):
            if len(v) == 0:
                raise ValueError("Input list cannot be empty.")
            if len(v) > 2048:
                raise ValueError("Input list cannot exceed 2048 items.")
            for item in v:
                if not isinstance(item, str):
                    raise ValueError("All items in input list must be strings.")
                if not item.strip():
                    raise ValueError("Input strings cannot be empty.")
            return v
        raise ValueError("Input must be a string or list of strings.")

    @validator("model")
    def validate_model(cls, v):
        allowed = {
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        }
        if v not in allowed:
            raise ValueError(
                f"Model '{v}' is not supported. Choose from: {', '.join(sorted(allowed))}"
            )
        return v

    @validator("encoding_format")
    def validate_encoding_format(cls, v):
        if v != "float":
            raise ValueError("Only 'float' encoding format is supported.")
        return v

    def get_texts(self) -> List[str]:
        """Normalize input to a list of strings."""
        if isinstance(self.input, str):
            return [self.input]
        return self.input


class EmbeddingObject(BaseModel):
    """A single embedding result, OpenAI-compatible."""

    object: str = Field(default="embedding")
    embedding: List[float] = Field(..., description="The embedding vector.")
    index: int = Field(..., description="Index of the input text.")


class UsageStats(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response."""

    object: str = Field(default="list")
    data: List[EmbeddingObject]
    model: str
    usage: UsageStats


class BatchEmbeddingRequest(BaseModel):
    """Batch embedding request for multiple independent inputs."""

    requests: List[EmbeddingRequest] = Field(
        ...,
        description="List of embedding requests to process in batch.",
        min_items=1,
        max_items=100,
    )


class BatchEmbeddingResponse(BaseModel):
    """Response for a batch embedding operation."""

    object: str = Field(default="batch_list")
    results: List[EmbeddingResponse]
    total_usage: UsageStats
    batch_size: int


class ErrorDetail(BaseModel):
    message: str
    type: str
    code: str
    param: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class PaymentRequiredResponse(BaseModel):
    """Payment required response with Mainlayer billing details."""

    error: str = "payment_required"
    message: str
    amount_required: float
    currency: str = "USD"
    payment_endpoint: str
    documentation_url: str
