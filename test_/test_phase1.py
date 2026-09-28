"""
Test Phase 1: Configuration and Utilities
"""

import pytest
from config.settings import settings
from config.logging_config import get_logger
from utils.exceptions import (
    InvalidVideoIDError,
    InvalidURLError,
    TranscriptFetchError,
    LLMError
)
from utils.validators import (
    validate_youtube_video_id,
    extract_video_id_from_url,
    validate_query,
    is_valid_youtube_url,
    sanitize_text
)


# ============================================
# Test Settings
# ============================================

def test_settings_loaded():
    """Test that settings are loaded correctly"""
    assert settings.OPENAI_EMBEDDING_MODEL == "text-embedding-3-small"
    assert settings.OPENAI_CHAT_MODEL == "gpt-4o-mini"
    assert settings.CHUNK_SIZE == 1000
    assert settings.CHUNK_OVERLAP == 200
    assert settings.RETRIEVAL_TOP_K == 4


def test_allowed_origins_list():
    """Test CORS origins parsing"""
    origins = settings.allowed_origins_list
    assert isinstance(origins, list)
    assert len(origins) > 0


# ============================================
# Test Logging
# ============================================

def test_logger_creation():
    """Test logger can be created"""
    logger = get_logger(__name__)
    assert logger is not None
    logger.info("Test log message")


# ============================================
# Test Exceptions
# ============================================

def test_custom_exceptions():
    """Test custom exceptions can be raised"""
    with pytest.raises(InvalidVideoIDError):
        raise InvalidVideoIDError("Test error")
    
    with pytest.raises(TranscriptFetchError):
        raise TranscriptFetchError("Test error")
    
    with pytest.raises(LLMError):
        raise LLMError("Test error")


# ============================================
# Test Validators
# ============================================

def test_validate_youtube_video_id_valid():
    """Test valid YouTube video ID"""
    video_id = "dQw4w9WgXcQ"
    result = validate_youtube_video_id(video_id)
    assert result == video_id


def test_validate_youtube_video_id_invalid():
    """Test invalid YouTube video ID"""
    with pytest.raises(InvalidVideoIDError):
        validate_youtube_video_id("invalid")
    
    with pytest.raises(InvalidVideoIDError):
        validate_youtube_video_id("")
    
    with pytest.raises(InvalidVideoIDError):
        validate_youtube_video_id("toolongvideoidshouldnotwork")


def test_extract_video_id_from_url_standard():
    """Test extracting video ID from standard YouTube URL"""
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    video_id = extract_video_id_from_url(url)
    assert video_id == "dQw4w9WgXcQ"


def test_extract_video_id_from_url_short():
    """Test extracting video ID from short YouTube URL"""
    url = "https://youtu.be/dQw4w9WgXcQ"
    video_id = extract_video_id_from_url(url)
    assert video_id == "dQw4w9WgXcQ"


def test_extract_video_id_from_url_embed():
    """Test extracting video ID from embed URL"""
    url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
    video_id = extract_video_id_from_url(url)
    assert video_id == "dQw4w9WgXcQ"


def test_extract_video_id_from_url_invalid():
    """Test invalid URL"""
    with pytest.raises(InvalidURLError):
        extract_video_id_from_url("https://example.com")


def test_validate_query_valid():
    """Test valid query"""
    query = "What is RAG?"
    result = validate_query(query)
    assert result == "What is RAG?"


def test_validate_query_too_short():
    """Test query too short"""
    with pytest.raises(ValueError):
        validate_query("Hi")


def test_validate_query_whitespace_cleanup():
    """Test query whitespace cleanup"""
    query = "  What   is   RAG?  "
    result = validate_query(query)
    assert result == "What is RAG?"


def test_is_valid_youtube_url():
    """Test YouTube URL validation"""
    assert is_valid_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ") is True
    assert is_valid_youtube_url("https://youtu.be/dQw4w9WgXcQ") is True
    assert is_valid_youtube_url("https://example.com") is False


def test_sanitize_text():
    """Test text sanitization"""
    text = "  Hello   World  "
    result = sanitize_text(text)
    assert result == "Hello World"
    
    # Test with max length
    long_text = "a" * 100
    result = sanitize_text(long_text, max_length=50)
    assert len(result) == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])