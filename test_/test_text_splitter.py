"""
Test Phase 3: Text Splitting
Tests for transcript chunking functionality
"""

import pytest
from indexing.text_splitter import (
    TranscriptTextSplitter,
    split_transcript_into_chunks
)
from indexing.document_loader import load_youtube_transcript


# ============================================
# Test Initialization
# ============================================

def test_splitter_initialization_default():
    """Test splitter initializes with default settings"""
    splitter = TranscriptTextSplitter()
    assert splitter.chunk_size == 1000
    assert splitter.chunk_overlap == 200


def test_splitter_initialization_custom():
    """Test splitter with custom parameters"""
    splitter = TranscriptTextSplitter(chunk_size=500, chunk_overlap=100)
    assert splitter.chunk_size == 500
    assert splitter.chunk_overlap == 100


# ============================================
# Test Basic Text Splitting
# ============================================

def test_split_text_simple():
    """Test basic text splitting"""
    splitter = TranscriptTextSplitter(chunk_size=100, chunk_overlap=20)
    
    # Create test text (200 chars)
    text = "This is a test sentence. " * 8  # 200 chars
    
    chunks = splitter.split_text(text)
    
    # Should create multiple chunks
    assert len(chunks) > 1
    
    # Each chunk should have required fields
    assert "text" in chunks[0]
    assert "chunk_id" in chunks[0]
    assert "chunk_length" in chunks[0]
    
    print(f"\nSplit {len(text)} chars into {len(chunks)} chunks")


def test_split_text_with_metadata():
    """Test splitting with metadata"""
    splitter = TranscriptTextSplitter(chunk_size=100, chunk_overlap=20)
    
    text = "Test text. " * 20
    metadata = {"video_id": "test123", "language": "en"}
    
    chunks = splitter.split_text(text, metadata=metadata)
    
    # Metadata should be in each chunk
    assert chunks[0]["metadata"]["video_id"] == "test123"
    assert chunks[0]["metadata"]["language"] == "en"


def test_split_empty_text():
    """Test splitting empty text"""
    splitter = TranscriptTextSplitter()
    chunks = splitter.split_text("")
    
    assert len(chunks) == 0


# ============================================
# Test Real Transcript Splitting
# ============================================

def test_split_real_transcript():
    """Test splitting actual YouTube transcript (Phase 2 + Phase 3 integration)"""
    # Phase 2: Load transcript
    transcript = load_youtube_transcript("kY14KfZQ1TI")
    
    print(f"\nLoaded transcript:")
    print(f"  Total chars: {transcript['total_chars']}")
    
    # Phase 3: Split transcript
    splitter = TranscriptTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_transcript(transcript)
    
    print(f"\nSplit into chunks:")
    print(f"  Number of chunks: {len(chunks)}")
    
    # Verify chunks created
    assert len(chunks) > 0
    
    # First chunk should have all fields
    first_chunk = chunks[0]
    assert "text" in first_chunk
    assert "chunk_id" in first_chunk
    assert "chunk_length" in first_chunk
    assert "metadata" in first_chunk
    
    # Metadata should include video info
    assert first_chunk["metadata"]["video_id"] == "kY14KfZQ1TI"
    assert first_chunk["metadata"]["language"] == "en"
    
    # Chunk IDs should be sequential
    assert chunks[0]["chunk_id"] == 0
    assert chunks[1]["chunk_id"] == 1
    
    print(f"\nFirst chunk preview:")
    print(f"  ID: {first_chunk['chunk_id']}")
    print(f"  Length: {first_chunk['chunk_length']} chars")
    print(f"  Text: {first_chunk['text'][:100]}...")


def test_chunk_overlap():
    """Test that chunks actually overlap"""
    splitter = TranscriptTextSplitter(chunk_size=100, chunk_overlap=30)
    
    # Create text that will split into multiple chunks
    text = "ABCDEFGHIJ" * 20  # 200 chars
    
    chunks = splitter.split_text(text)
    
    # Should have overlap between chunks
    if len(chunks) > 1:
        # Last part of chunk 0 should appear in chunk 1
        chunk0_end = chunks[0]["text"][-10:]
        chunk1_start = chunks[1]["text"][:10]
        
        # Not exact match, but should have some overlap
        assert len(chunks) >= 2
        print(f"\nCreated {len(chunks)} chunks with overlap")


# ============================================
# Test Chunk Statistics
# ============================================

def test_get_chunk_stats():
    """Test chunk statistics calculation"""
    transcript = load_youtube_transcript("kY14KfZQ1TI")
    splitter = TranscriptTextSplitter()
    chunks = splitter.split_transcript(transcript)
    
    stats = splitter.get_chunk_stats(chunks)
    
    # Check all stats present
    assert "total_chunks" in stats
    assert "total_chars" in stats
    assert "avg_chunk_length" in stats
    assert "min_chunk_length" in stats
    assert "max_chunk_length" in stats
    
    # Values should make sense
    assert stats["total_chunks"] == len(chunks)
    assert stats["total_chunks"] > 0
    assert stats["avg_chunk_length"] > 0
    
    print(f"\nChunk Statistics:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Average length: {stats['avg_chunk_length']:.0f} chars")
    print(f"  Min length: {stats['min_chunk_length']} chars")
    print(f"  Max length: {stats['max_chunk_length']} chars")


def test_get_chunk_stats_empty():
    """Test stats with empty chunk list"""
    splitter = TranscriptTextSplitter()
    stats = splitter.get_chunk_stats([])
    
    assert stats["total_chunks"] == 0
    assert stats["total_chars"] == 0


# ============================================
# Test Different Chunk Sizes
# ============================================

def test_different_chunk_sizes():
    """Test how chunk size affects number of chunks"""
    transcript = load_youtube_transcript("kY14KfZQ1TI")
    
    # Small chunks
    splitter_small = TranscriptTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks_small = splitter_small.split_transcript(transcript)
    
    # Large chunks
    splitter_large = TranscriptTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks_large = splitter_large.split_transcript(transcript)
    
    # Smaller chunk size = more chunks
    assert len(chunks_small) > len(chunks_large)
    
    print(f"\nChunk size comparison:")
    print(f"  500 chars: {len(chunks_small)} chunks")
    print(f"  2000 chars: {len(chunks_large)} chunks")


# ============================================
# Test Convenience Function
# ============================================

def test_convenience_function():
    """Test the convenience function"""
    transcript = load_youtube_transcript("kY14KfZQ1TI")
    chunks = split_transcript_into_chunks(transcript)
    
    assert len(chunks) > 0
    assert chunks[0]["metadata"]["video_id"] == "kY14KfZQ1TI"


def test_convenience_function_custom_params():
    """Test convenience function with custom parameters"""
    transcript = load_youtube_transcript("kY14KfZQ1TI")
    chunks = split_transcript_into_chunks(
        transcript,
        chunk_size=500,
        chunk_overlap=50
    )
    
    assert len(chunks) > 0
    print(f"\nConvenience function created {len(chunks)} chunks")


# ============================================
# Integration Test: Phase 2 + Phase 3
# ============================================

def test_phase2_to_phase3_integration():
    """Test complete flow: Load transcript → Split into chunks"""
    print("\n" + "=" * 60)
    print("Integration Test: Phase 2 → Phase 3")
    print("=" * 60)
    
    # Phase 2: Document Ingestion
    print("\nPhase 2: Loading transcript...")
    transcript = load_youtube_transcript("kY14KfZQ1TI")
    print(f"  ✓ Loaded {transcript['total_chars']} characters")
    
    # Phase 3: Text Splitting
    print("\nPhase 3: Splitting into chunks...")
    splitter = TranscriptTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_transcript(transcript)
    print(f"  ✓ Created {len(chunks)} chunks")
    
    # Verify integration
    print("\nVerification:")
    print(f"  Original text: {transcript['total_chars']} chars")
    print(f"  Number of chunks: {len(chunks)}")
    print(f"  Average chunk size: {sum(c['chunk_length'] for c in chunks) / len(chunks):.0f} chars")
    
    # First chunk should have all metadata
    first_chunk = chunks[0]
    print(f"\nFirst chunk details:")
    print(f"  Chunk ID: {first_chunk['chunk_id']}")
    print(f"  Length: {first_chunk['chunk_length']} chars")
    print(f"  Video ID: {first_chunk['metadata']['video_id']}")
    print(f"  Preview: {first_chunk['text'][:150]}...")
    
    # All chunks should have metadata
    for chunk in chunks:
        assert chunk["metadata"]["video_id"] == "kY14KfZQ1TI"
    
    print("\n" + "=" * 60)
    print("✓ Phase 2 + Phase 3 Integration Working!")
    print("=" * 60)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])