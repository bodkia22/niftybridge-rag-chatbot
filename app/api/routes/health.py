from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health_check() -> dict[str, str]:
    """Simple liveness check — confirms the service is running."""
    return {"status": "ok"}