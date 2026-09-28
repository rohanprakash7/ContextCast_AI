"""
Input Validators for Conversational RAG Platform
Validate user inputs before processing
"""

import re
from typing import Optional
from urllib.parse import urlparse, parse_qs

from utils.exceptions import InvalidVideoIDError, InvalidURLError


def validate_youtube_video_id(video_id: str) -> str:
    """
    Validate YouTube video ID format
    
    Args:
        video_id: YouTube video ID to validate
    
    Returns:
        Validated video ID (cleaned)
    
    Raises:
        InvalidVideoIDError: If video ID format is invalid
    
    Example:
        >>> validate_youtube_video_id("dQw4w9WgXcQ")
        'dQw4w9WgXcQ'
        >>> validate_youtube_video_id("invalid!")
        InvalidVideoIDError
    """
    if not video_id:
        raise InvalidVideoIDError("Video ID cannot be empty")
    
    # Remove whitespace
    video_id = video_id.strip()
    
    # YouTube video IDs are always 11 characters
    # Contains: letters (a-z, A-Z), numbers (0-9), hyphens (-), underscores (_)
    video_id_pattern = r'^[a-zA-Z0-9_-]{11}$'
    
    if not re.match(video_id_pattern, video_id):
        raise InvalidVideoIDError(
            f"Invalid YouTube video ID format: '{video_id}'. "
            "Must be 11 characters (letters, numbers, -, _)"
        )
    
    return video_id


def extract_video_id_from_url(url: str) -> str:
    """
    Extract video ID from YouTube URL
    
    Supports multiple URL formats:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - https://m.youtube.com/watch?v=VIDEO_ID
    
    Args:
        url: YouTube URL
    
    Returns:
        Extracted and validated video ID
    
    Raises:
        InvalidURLError: If URL format is invalid
        InvalidVideoIDError: If extracted video ID is invalid
    
    Example:
        >>> extract_video_id_from_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        'dQw4w9WgXcQ'
    """
    if not url:
        raise InvalidURLError("URL cannot be empty")
    
    url = url.strip()
    
    try:
        parsed = urlparse(url)
        
        # Check if it's a YouTube domain
        valid_domains = ['youtube.com', 'www.youtube.com', 'm.youtube.com', 'youtu.be']
        if parsed.netloc not in valid_domains:
            raise InvalidURLError(
                f"Invalid YouTube domain: {parsed.netloc}. "
                f"Must be one of: {', '.join(valid_domains)}"
            )
        
        # Extract video ID based on URL format
        video_id = None
        
        # Format 1: youtube.com/watch?v=VIDEO_ID
        if 'watch' in parsed.path:
            query_params = parse_qs(parsed.query)
            video_id = query_params.get('v', [None])[0]
        
        # Format 2: youtu.be/VIDEO_ID
        elif parsed.netloc == 'youtu.be':
            video_id = parsed.path.lstrip('/')
        
        # Format 3: youtube.com/embed/VIDEO_ID
        elif 'embed' in parsed.path:
            video_id = parsed.path.split('/embed/')[-1]
        
        if not video_id:
            raise InvalidURLError(
                f"Could not extract video ID from URL: {url}"
            )
        
        # Validate extracted video ID
        return validate_youtube_video_id(video_id)
    
    except Exception as e:
        if isinstance(e, (InvalidURLError, InvalidVideoIDError)):
            raise
        raise InvalidURLError(f"Failed to parse URL: {e}")


def validate_query(query: str, min_length: int = 3, max_length: int = 500) -> str:
    """
    Validate user query
    
    Args:
        query: User's search query
        min_length: Minimum query length (default: 3)
        max_length: Maximum query length (default: 500)
    
    Returns:
        Validated query (cleaned)
    
    Raises:
        ValueError: If query is invalid
    
    Example:
        >>> validate_query("What is RAG?")
        'What is RAG?'
    """
    if not query:
        raise ValueError("Query cannot be empty")
    
    # Remove extra whitespace
    query = " ".join(query.split())
    
    if len(query) < min_length:
        raise ValueError(
            f"Query too short. Minimum {min_length} characters required."
        )
    
    if len(query) > max_length:
        raise ValueError(
            f"Query too long. Maximum {max_length} characters allowed."
        )
    
    return query


def is_valid_youtube_url(url: str) -> bool:
    """
    Check if URL is a valid YouTube URL (non-raising version)
    
    Args:
        url: URL to check
    
    Returns:
        True if valid YouTube URL, False otherwise
    
    Example:
        >>> is_valid_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        True
        >>> is_valid_youtube_url("https://example.com")
        False
    """
    try:
        extract_video_id_from_url(url)
        return True
    except (InvalidURLError, InvalidVideoIDError):
        return False


def sanitize_text(text: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize user input text
    
    Args:
        text: Text to sanitize
        max_length: Maximum length (truncate if longer)
    
    Returns:
        Sanitized text
    
    Example:
        >>> sanitize_text("  Hello   World  ")
        'Hello World'
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Truncate if needed
    if max_length and len(text) > max_length:
        text = text[:max_length].rstrip()
    
    return text