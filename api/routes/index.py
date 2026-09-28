"""
Indexing Endpoints - API routes for video indexing
"""

from fastapi import APIRouter, HTTPException, status
from datetime import datetime

from api.models import IndexRequest, IndexResponse, VideoStatusResponse
from chains.indexing_chain import index_video_to_pinecone, check_if_video_indexed
from config.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/index", tags=["Indexing"])


@router.post("", response_model=IndexResponse, status_code=status.HTTP_201_CREATED)
async def index_video(request: IndexRequest):
    """
    Index a YouTube video
    
    Executes complete indexing pipeline using LangChain.
    
    Example:
        POST /index
        {
            "video_id": "kY14KfZQ1TI",
            "chunk_size": 1000,
            "chunk_overlap": 200
        }
    """
    logger.info(f"Indexing request for video: {request.video_id}")
    
    try:
        # Check if already indexed
        if check_if_video_indexed(request.video_id, namespace=request.namespace):
            logger.info(f"Video {request.video_id} already indexed")
            return IndexResponse(
                video_id=request.video_id,
                status="already_indexed",
                num_chunks=0,
                transcript_chars=0,
                duration_seconds=0.0,
                timestamp=datetime.now().isoformat()
            )
        
        # Execute indexing pipeline
        result = index_video_to_pinecone(
            video_id=request.video_id,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            namespace=request.namespace
        )
        
        # Convert to response model
        response = IndexResponse(**result)
        
        logger.info(
            f"Successfully indexed video {request.video_id}: "
            f"{result['num_chunks']} chunks in {result['duration_seconds']:.2f}s"
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {str(e)}"
        )


@router.get("/status/{video_id}", response_model=VideoStatusResponse)
async def get_video_status(video_id: str, namespace: str = ""):
    """
    Check if a video is indexed
    
    Example:
        GET /index/status/kY14KfZQ1TI
        
        Response:
        {
            "video_id": "kY14KfZQ1TI",
            "is_indexed": true
        }
    """
    logger.info(f"Status check for video: {video_id}")
    
    try:
        is_indexed = check_if_video_indexed(video_id, namespace=namespace)
        
        response = VideoStatusResponse(
            video_id=video_id,
            is_indexed=is_indexed
        )
        
        logger.info(f"Video {video_id} indexed: {is_indexed}")
        
        return response
    
    except Exception as e:
        logger.error(f"Failed to check status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )