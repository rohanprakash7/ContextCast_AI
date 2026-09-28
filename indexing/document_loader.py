"""
Document Loader for YouTube Transcripts
Works with youtube-transcript-api 1.2.3+
"""

from typing import Dict, List, Optional
from youtube_transcript_api import YouTubeTranscriptApi

from config.logging_config import get_logger
from utils.exceptions import TranscriptFetchError, TranscriptNotAvailableError, InvalidVideoIDError
from utils.validators import validate_youtube_video_id

logger = get_logger(__name__)


class YouTubeTranscriptLoader:
    """Load YouTube video transcripts using youtube-transcript-api 1.2.3+"""
    
    def __init__(self, preferred_languages: Optional[List[str]] = None):
        """Initialize transcript loader"""
        self.preferred_languages = preferred_languages or ["en"]
        logger.info(f"Initialized YouTubeTranscriptLoader with languages: {self.preferred_languages}")
    
    def load(self, video_id: str) -> Dict[str, any]:
        """
        Load transcript for a YouTube video
        
        Args:
            video_id: YouTube video ID (11 characters)
        
        Returns:
            Dictionary with transcript text and metadata
        """
        video_id = validate_youtube_video_id(video_id)
        logger.info(f"Fetching transcript for video: {video_id}")
        
        try:
            # Create API instance
            api = YouTubeTranscriptApi()
            
            # Fetch transcript (returns FetchedTranscriptSnippet objects)
            transcript_list = api.fetch(video_id, languages=self.preferred_languages)
            
            # Extract text from FetchedTranscriptSnippet objects
            transcript_text = " ".join(segment.text for segment in transcript_list)
            
            result = {
                "text": transcript_text,
                "video_id": video_id,
                "language": self.preferred_languages[0],
                "num_segments": len(transcript_list),
                "total_chars": len(transcript_text)
            }
            
            logger.info(
                f"Successfully loaded transcript: "
                f"{result['num_segments']} segments, {result['total_chars']} chars"
            )
            
            return result
        
        except Exception as e:
            error_msg = f"Failed to fetch transcript for {video_id}: {str(e)}"
            logger.error(error_msg)
            raise TranscriptFetchError(error_msg)


def load_youtube_transcript(
    video_id: str,
    languages: Optional[List[str]] = None
) -> Dict[str, any]:
    """
    Load a YouTube transcript (convenience function)
    
    Args:
        video_id: YouTube video ID
        languages: Preferred languages (default: ["en"])
    
    Returns:
        Dictionary with transcript and metadata
    """
    loader = YouTubeTranscriptLoader(preferred_languages=languages)
    return loader.load(video_id)