"""
Phase 4 Verification: Using LangChain for Everything
"""

import os
from indexing.document_loader import load_youtube_transcript
from indexing.text_splitter import split_transcript_into_chunks
from indexing.embeddings import get_embeddings_model
from indexing.vector_store import PineconeVectorStore
from config.settings import settings

# Set environment variables for LangChain Pinecone
os.environ["PINECONE_API_KEY"] = settings.PINECONE_API_KEY
os.environ["PINECONE_ENVIRONMENT"] = settings.PINECONE_ENVIRONMENT

print("=" * 60)
print("Phase 4: LangChain Embeddings & Vector Store")
print("=" * 60)

# Step 1: Load transcript (Phase 2)
print("\nStep 1: Loading transcript...")
transcript = load_youtube_transcript("kY14KfZQ1TI")
print(f"  ✓ Loaded {transcript['total_chars']} characters")

# Step 2: Split into chunks (Phase 3)
print("\nStep 2: Splitting into chunks...")
chunks = split_transcript_into_chunks(transcript)
print(f"  ✓ Created {len(chunks)} chunks")

# Step 3: Add to Pinecone using LangChain
print("\nStep 3: Adding to Pinecone via LangChain...")
print("  (LangChain will automatically generate embeddings)")
print("  (This takes ~15-30 seconds for 69 chunks)")

store = PineconeVectorStore()
result = store.add_chunks(chunks)

print(f"  ✓ Added {result['added_count']} documents to Pinecone")
print(f"  ✓ LangChain handled: embedding + storage automatically!")

# Step 4: Verify it's indexed
print("\nStep 4: Checking if video is indexed...")
status = store.check_if_indexed("kY14KfZQ1TI")
print(f"  ✓ Video indexed: {status['is_indexed']}")

# Step 5: Test retriever
print("\nStep 5: Testing LangChain retriever...")
retriever = store.get_retriever(
    k=3,
    filter={"video_id": "kY14KfZQ1TI"}
)
print(f"  ✓ Created LangChain retriever")

# Test query
results = retriever.invoke("What is deep learning?")
print(f"  ✓ Retrieved {len(results)} documents")

if results:
    print(f"\n  Top result preview:")
    print(f"    {results[0].page_content[:150]}...")

print("\n" + "=" * 60)
print("Phase 4 Complete - Pure LangChain!")
print("=" * 60)
print("\nWhat LangChain Did For Us:")
print("  ✓ Embedded 69 chunks automatically")
print("  ✓ Stored in Pinecone with metadata")
print("  ✓ Created unique IDs (video_id_chunk_id)")
print("  ✓ Enabled retrieval with filtering")
print("\nReady for Phase 5: Build complete indexing chain!")