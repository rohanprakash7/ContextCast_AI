"""
Custom Exceptions for Conversational RAG Platform
Define domain-specific exceptions for better error handling
"""


class RAGException(Exception):
    """Base exception for all RAG-related errors"""
    pass


# ============================================
# Indexing Exceptions
# ============================================

class IndexingError(RAGException):
    """Base exception for indexing-related errors"""
    pass


class TranscriptFetchError(IndexingError):
    """Raised when YouTube transcript cannot be fetched"""
    pass


class TranscriptNotAvailableError(IndexingError):
    """Raised when transcript is disabled for a video"""
    pass


class EmbeddingGenerationError(IndexingError):
    """Raised when embedding generation fails"""
    pass


class VectorStoreError(IndexingError):
    """Raised when vector store operations fail"""
    pass


# ============================================
# Retrieval Exceptions
# ============================================

class RetrievalError(RAGException):
    """Base exception for retrieval-related errors"""
    pass


class QueryRewriteError(RetrievalError):
    """Raised when query rewriting fails"""
    pass


class SearchError(RetrievalError):
    """Raised when vector search fails"""
    pass


class RerankingError(RetrievalError):
    """Raised when reranking operation fails"""
    pass


# ============================================
# Generation Exceptions
# ============================================

class GenerationError(RAGException):
    """Base exception for generation-related errors"""
    pass


class LLMError(GenerationError):
    """Raised when LLM API call fails"""
    pass


class PromptTooLongError(GenerationError):
    """Raised when prompt exceeds token limit"""
    pass


class CitationError(GenerationError):
    """Raised when citation generation fails"""
    pass


# ============================================
# Validation Exceptions
# ============================================

class ValidationError(RAGException):
    """Base exception for validation errors"""
    pass


class InvalidVideoIDError(ValidationError):
    """Raised when YouTube video ID is invalid"""
    pass


class InvalidURLError(ValidationError):
    """Raised when URL format is invalid"""
    pass


class InvalidConfigError(ValidationError):
    """Raised when configuration is invalid"""
    pass


# ============================================
# API Exceptions
# ============================================

class APIError(RAGException):
    """Base exception for API-related errors"""
    pass


class RateLimitError(APIError):
    """Raised when API rate limit is exceeded"""
    pass


class AuthenticationError(APIError):
    """Raised when API authentication fails"""
    pass


class ResourceNotFoundError(APIError):
    """Raised when requested resource is not found"""
    pass