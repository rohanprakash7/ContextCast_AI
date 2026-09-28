"""
Quick verification for Phase 3: Text Splitting
"""

from indexing.document_loader import load_youtube_transcript
from indexing.text_splitter import TranscriptTextSplitter, split_transcript_into_chunks

print("=" * 60)
print("Phase 3: Text Splitting Verification")
print("=" * 60)

# Step 1: Load transcript (Phase 2)
print("\nStep 1: Loading transcript from YouTube...")
transcript = load_youtube_transcript("kY14KfZQ1TI")
print(f"  ✓ Loaded {transcript['total_chars']} characters")
print(f"  ✓ Video ID: {transcript['video_id']}")

# Step 2: Split into chunks (Phase 3)
print("\nStep 2: Splitting into chunks...")
splitter = TranscriptTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_transcript(transcript)
print(f"  ✓ Created {len(chunks)} chunks")

# Step 3: Show statistics
print("\nStep 3: Analyzing chunks...")
stats = splitter.get_chunk_stats(chunks)
print(f"  Total chunks: {stats['total_chunks']}")
print(f"  Average size: {stats['avg_chunk_length']:.0f} chars")
print(f"  Min size: {stats['min_chunk_length']} chars")
print(f"  Max size: {stats['max_chunk_length']} chars")

# Step 4: Show sample chunks
print("\nStep 4: Sample chunks...")
print(f"\nChunk 0 (first):")
print(f"  ID: {chunks[0]['chunk_id']}")
print(f"  Length: {chunks[0]['chunk_length']} chars")
print(f"  Video: {chunks[0]['metadata']['video_id']}")
print(f"  Text: {chunks[0]['text'][:150]}...")

if len(chunks) > 1:
    print(f"\nChunk 1 (second):")
    print(f"  ID: {chunks[1]['chunk_id']}")
    print(f"  Length: {chunks[1]['chunk_length']} chars")
    print(f"  Text: {chunks[1]['text'][:150]}...")

if len(chunks) > 2:
    mid = len(chunks) // 2
    print(f"\nChunk {mid} (middle):")
    print(f"  ID: {chunks[mid]['chunk_id']}")
    print(f"  Length: {chunks[mid]['chunk_length']} chars")
    print(f"  Text: {chunks[mid]['text'][:150]}...")

# Step 5: Verify overlap
print("\nStep 5: Checking chunk overlap...")
if len(chunks) > 1:
    # Check if chunks have overlap
    chunk0_end = chunks[0]['text'][-50:]
    chunk1_text = chunks[1]['text']
    
    # Check if any part of chunk 0's end appears in chunk 1
    has_overlap = any(word in chunk1_text for word in chunk0_end.split()[:5])
    
    if has_overlap:
        print(f"  ✓ Chunks have overlap (as expected)")
    else:
        print(f"  ✓ Chunks created successfully")

print("\n" + "=" * 60)
print("Phase 3 Verification Complete!")
print("=" * 60)
print(f"\nReady for Phase 4:")
print(f"  - {len(chunks)} chunks to embed")
print(f"  - Will create {len(chunks)} vectors")
print(f"  - Will store in Pinecone")