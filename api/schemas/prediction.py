from typing import List

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    """Request schema for the prediction endpoint."""

    features: List[float]


class PredictionResponse(BaseModel):
    """Response schema for the prediction endpoint."""

    prediction: int
    label: str
    confidence: float
    status: str
