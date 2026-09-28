"""
Health Check Endpoint
"""

from fastapi import APIRouter
from datetime import datetime

from api.models import HealthResponse
from config.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns API status
    
    Example:
        GET /health
        
        Response:
        {
            "status": "healthy",
            "timestamp": "2026-01-18T03:30:00",
            "version": "1.0.0"
        }
    """
    logger.info("Health check requested")
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )