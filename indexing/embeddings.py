"""
Embedding Generator using LangChain OpenAI Integration
Simple wrapper around LangChain's OpenAIEmbeddings
"""

from typing import List, Dict, Optional
from langchain_openai import OpenAIEmbeddings

from config.logging_config import get_logger
from config.settings import settings
from utils.exceptions import EmbeddingGenerationError

logger = get_logger(__name__)


def get_embeddings_model(
    model: Optional[str] = None,
    dimensions: Optional[int] = None
) -> OpenAIEmbeddings:
    """
    Get LangChain OpenAI embeddings instance
    
    Args:
        model: OpenAI model name (default: from settings)
        dimensions: Embedding dimensions (default: from settings)
    
    Returns:
        LangChain OpenAIEmbeddings instance
    
    Example:
        >>> embeddings = get_embeddings_model()
        >>> vector = embeddings.embed_query("Hello world")
        >>> len(vector)
        512
    """
    model = model or settings.OPENAI_EMBEDDING_MODEL
    dimensions = dimensions or settings.EMBEDDING_DIMENSIONS
    
    try:
        embeddings = OpenAIEmbeddings(
            model=model,
            dimensions=dimensions,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        logger.info(f"Created LangChain OpenAIEmbeddings: model={model}, dims={dimensions}")
        
        return embeddings
    
    except Exception as e:
        logger.error(f"Failed to create embeddings: {e}")
        raise EmbeddingGenerationError(f"Embedding initialization failed: {e}")