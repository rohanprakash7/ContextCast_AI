"""
Simple tests for Phase 2: Document Ingestion
"""

import pytest
from indexing.document_loader import YouTubeTranscriptLoader, load_youtube_transcript
from utils.exceptions import InvalidVideoIDError, TranscriptFetchError
from utils.validators import extract_video_id_from_url


def test_loader_initialization():
    """Test basic initialization"""
    loader = YouTubeTranscriptLoader()
    assert loader.preferred_languages == ["en"]


def test_extract_video_id_from_url():
    """Test extracting ID from full YouTube URL"""
    url = "https://www.youtube.com/watch?v=kY14KfZQ1TI"
    video_id = extract_video_id_from_url(url)
    assert video_id == "kY14KfZQ1TI"


def test_load_transcript_simple():
    """Test loading a real transcript"""
    loader = YouTubeTranscriptLoader()
    result = loader.load("kY14KfZQ1TI")
    
    # Check structure
    assert "text" in result
    assert "video_id" in result
    assert "num_segments" in result
    
    # Check values
    assert len(result["text"]) > 0
    assert result["video_id"] == "kY14KfZQ1TI"
    
    print(f"\nLoaded {result['num_segments']} segments, {result['total_chars']} chars")


def test_invalid_video_id():
    """Test with invalid video ID"""
    loader = YouTubeTranscriptLoader()
    
    with pytest.raises(InvalidVideoIDError):
        loader.load("invalid")


def test_convenience_function():
    """Test the convenience function"""
    result = load_youtube_transcript("kY14KfZQ1TI")
    assert len(result["text"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])