"""
Text Splitter for Document Chunking
Splits long texts into smaller chunks for optimal retrieval
"""

from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class TranscriptTextSplitter:
    """
    Split YouTube transcripts into chunks for embedding
    
    Uses LangChain's RecursiveCharacterTextSplitter to intelligently
    split text while preserving semantic coherence.
    
    Features:
    - Configurable chunk size and overlap
    - Preserves semantic coherence (tries to split at sentence boundaries)
    - Metadata preservation (video_id, language, etc.)
    - Character-based splitting (not token-based)
    """
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize text splitter
        
        Args:
            chunk_size: Maximum characters per chunk (default: from settings)
            chunk_overlap: Characters overlap between chunks (default: from settings)
            separators: List of separators for splitting (default: ["\n\n", "\n", ". ", " ", ""])
        
        Example:
            >>> splitter = TranscriptTextSplitter(chunk_size=1000, chunk_overlap=200)
            >>> # Creates splitter with 1000 char chunks, 200 char overlap
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        
        # Initialize LangChain's RecursiveCharacterTextSplitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False
        )
        
        logger.info(
            f"Initialized TranscriptTextSplitter with "
            f"chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}"
        )
    
    def split_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, any]]:
        """
        Split text into chunks with metadata
        
        Args:
            text: Text to split
            metadata: Optional metadata to attach to each chunk
        
        Returns:
            List of chunk dictionaries, each containing:
                - text: Chunk text
                - chunk_id: Sequential ID (0, 1, 2, ...)
                - chunk_length: Character count
                - metadata: Copy of provided metadata
        
        Example:
            >>> splitter = TranscriptTextSplitter()
            >>> chunks = splitter.split_text("Long text here...", {"video_id": "abc123"})
            >>> print(len(chunks))  # Number of chunks created
            >>> print(chunks[0]['text'])  # First chunk text
        """
        if not text:
            logger.warning("Empty text provided to splitter")
            return []
        
        # Split text using LangChain splitter
        text_chunks = self.splitter.split_text(text)
        
        logger.info(
            f"Split text into {len(text_chunks)} chunks "
            f"(original length: {len(text)} chars)"
        )
        
        # Create chunk objects with metadata
        chunks = []
        for idx, chunk_text in enumerate(text_chunks):
            chunk = {
                "text": chunk_text,
                "chunk_id": idx,
                "chunk_length": len(chunk_text),
                "metadata": metadata or {}
            }
            chunks.append(chunk)
        
        return chunks
    
    def split_transcript(
        self,
        transcript_data: Dict[str, any]
    ) -> List[Dict[str, any]]:
        """
        Split transcript data from document loader
        
        This is the main method to use - takes output from Phase 2
        and produces chunks ready for Phase 4.
        
        Args:
            transcript_data: Dictionary from YouTubeTranscriptLoader.load()
                Expected keys: text, video_id, language, num_segments, total_chars
        
        Returns:
            List of chunks with full metadata
        
        Example:
            >>> from indexing.document_loader import load_youtube_transcript
            >>> from indexing.text_splitter import TranscriptTextSplitter
            >>> 
            >>> # Phase 2: Load transcript
            >>> transcript = load_youtube_transcript("kY14KfZQ1TI")
            >>> 
            >>> # Phase 3: Split into chunks
            >>> splitter = TranscriptTextSplitter()
            >>> chunks = splitter.split_transcript(transcript)
            >>> 
            >>> print(f"Created {len(chunks)} chunks")
            >>> print(f"First chunk: {chunks[0]['text'][:100]}...")
        """
        # Extract data from Phase 2 output
        text = transcript_data.get("text", "")
        video_id = transcript_data.get("video_id", "unknown")
        language = transcript_data.get("language", "unknown")
        
        # Create metadata for each chunk
        metadata = {
            "video_id": video_id,
            "language": language,
            "source": "youtube",
            "total_chars": len(text)
        }
        
        logger.info(f"Splitting transcript for video {video_id}")
        
        # Split text
        chunks = self.split_text(text, metadata)
        
        logger.info(
            f"Created {len(chunks)} chunks for video {video_id} "
            f"(original: {len(text)} chars)"
        )
        
        return chunks
    
    def get_chunk_stats(self, chunks: List[Dict[str, any]]) -> Dict[str, any]:
        """
        Calculate statistics about chunks
        
        Useful for debugging and monitoring.
        
        Args:
            chunks: List of chunk dictionaries
        
        Returns:
            Dictionary with statistics:
                - total_chunks: Number of chunks
                - total_chars: Total characters across all chunks
                - avg_chunk_length: Average chunk size
                - min_chunk_length: Smallest chunk
                - max_chunk_length: Largest chunk
        
        Example:
            >>> stats = splitter.get_chunk_stats(chunks)
            >>> print(f"Average chunk size: {stats['avg_chunk_length']:.0f} chars")
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_chars": 0,
                "avg_chunk_length": 0,
                "min_chunk_length": 0,
                "max_chunk_length": 0
            }
        
        chunk_lengths = [c["chunk_length"] for c in chunks]
        
        stats = {
            "total_chunks": len(chunks),
            "total_chars": sum(chunk_lengths),
            "avg_chunk_length": sum(chunk_lengths) / len(chunks),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths)
        }
        
        return stats


def split_transcript_into_chunks(
    transcript_data: Dict[str, any],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> List[Dict[str, any]]:
    """
    Convenience function to split transcript
    
    Args:
        transcript_data: Dictionary from Phase 2 (document loader)
        chunk_size: Optional chunk size override
        chunk_overlap: Optional overlap override
    
    Returns:
        List of chunks ready for Phase 4 (embedding)
    
    Example:
        >>> from indexing.document_loader import load_youtube_transcript
        >>> from indexing.text_splitter import split_transcript_into_chunks
        >>> 
        >>> # Phase 2: Load
        >>> transcript = load_youtube_transcript("kY14KfZQ1TI")
        >>> 
        >>> # Phase 3: Split
        >>> chunks = split_transcript_into_chunks(transcript)
        >>> 
        >>> print(f"Created {len(chunks)} chunks")
    """
    splitter = TranscriptTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_transcript(transcript_data)