from fastapi import APIRouter

from api.schemas.health import HealthResponse

router = APIRouter()

API_VERSION = "1.0.0"


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Returns the current health status and version of the API."""
    return HealthResponse(status="ok", version=API_VERSION)
