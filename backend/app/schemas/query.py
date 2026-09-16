from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Text question. Image is handled as a separate multipart field on the route,
    not in this JSON body, since /query accepts multipart/form-data when an image
    is attached (see api/routes/query.py)."""

    question: str = Field(..., min_length=1, description="The user's question")


class DetectionInfo(BaseModel):
    num_products_detected: int
    avg_confidence: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    detection: Optional[DetectionInfo] = None
