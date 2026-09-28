"""
API Models - Request and Response Schemas using Pydantic
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


# ============================================
# Request Models
# ============================================

class IndexRequest(BaseModel):
    """Request model for indexing a video"""
    
    video_id: str = Field(
        ...,
        description="YouTube video ID (11 characters)",
        min_length=11,
        max_length=11,
        example="kY14KfZQ1TI"
    )
    
    namespace: Optional[str] = Field(
        default="",
        description="Pinecone namespace",
        example=""
    )
    
    chunk_size: Optional[int] = Field(
        default=None,
        description="Override default chunk size",
        ge=100,
        le=5000,
        example=1000
    )
    
    chunk_overlap: Optional[int] = Field(
        default=None,
        description="Override default chunk overlap",
        ge=0,
        le=1000,
        example=200
    )
    
    @validator('video_id')
    def validate_video_id_format(cls, v):
        """Validate video ID contains only valid characters"""
        import re
        if not re.match(r'^[a-zA-Z0-9_-]{11}$', v):
            raise ValueError('Invalid video ID format')
        return v


class ChatMessage(BaseModel):
    """A single turn in the conversation history"""
    question: str
    answer: str


class QueryRequest(BaseModel):
    """Request model for querying the system"""

    question: str = Field(
        ...,
        description="User's question",
        min_length=3,
        max_length=500,
        example="What is deep learning?"
    )

    video_id: Optional[str] = Field(
        default=None,
        description="Filter to specific video (optional)",
        min_length=11,
        max_length=11,
        example="kY14KfZQ1TI"
    )

    retriever_type: str = Field(
        default="simple",
        description="Retrieval strategy to use",
        example="simple"
    )

    top_k: Optional[int] = Field(
        default=4,
        description="Number of chunks to retrieve",
        ge=1,
        le=10,
        example=4
    )

    include_citations: bool = Field(
        default=True,
        description="Include source citations in answer",
        example=True
    )

    chat_history: Optional[List[ChatMessage]] = Field(
        default_factory=list,
        description="Previous conversation turns for context"
    )

    @validator('retriever_type')
    def validate_retriever_type(cls, v):
        """Validate retriever type is supported"""
        valid_types = ['simple', 'rewriting', 'hybrid']
        if v not in valid_types:
            raise ValueError(f'retriever_type must be one of {valid_types}')
        return v


# ============================================
# Response Models
# ============================================

class IndexResponse(BaseModel):
    """Response model for indexing operation"""
    
    video_id: str = Field(..., example="kY14KfZQ1TI")
    status: str = Field(..., example="success")
    num_chunks: int = Field(..., example=69)
    transcript_chars: int = Field(..., example=54326)
    duration_seconds: float = Field(..., example=18.5)
    timestamp: str = Field(..., example="2026-01-18T03:30:00")


class QueryResponse(BaseModel):
    """Response model for query operation"""
    
    question: str = Field(..., example="What is deep learning?")
    answer: str = Field(..., example="Deep learning is...")
    citations: List[int] = Field(default_factory=list, example=[0, 1, 2])
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved_chunks: int = Field(..., example=4)
    retriever_type: str = Field(..., example="simple")
    duration_seconds: float = Field(..., example=2.5)
    timestamp: str = Field(..., example="2026-01-18T03:30:00")


class VideoStatusResponse(BaseModel):
    """Response model for video status check"""
    
    video_id: str = Field(..., example="kY14KfZQ1TI")
    is_indexed: bool = Field(..., example=True)


class HealthResponse(BaseModel):
    """Response model for health check"""
    
    status: str = Field(..., example="healthy")
    timestamp: str = Field(..., example="2026-01-18T03:30:00")
    version: str = Field(..., example="1.0.0")