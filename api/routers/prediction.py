from fastapi import APIRouter

from api.schemas.prediction import PredictionRequest, PredictionResponse

router = APIRouter()

_LABEL_MAP = {1: "Normal", 2: "Suspect", 3: "Pathological"}


@router.post("/prediction", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Returns a mock fetal health prediction for the provided feature vector.

    Args:
        request: Input containing the feature values for prediction.

    Returns:
        A PredictionResponse with a mocked prediction class, label,
        confidence score, and a status flag indicating the mock nature.
    """
    return PredictionResponse(
        prediction=1,
        label=_LABEL_MAP[1],
        confidence=0.95,
        status="mock",
    )
