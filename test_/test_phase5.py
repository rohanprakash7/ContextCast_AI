"""
Phase 5 Verification: Pure LangChain Indexing Pipeline
"""

from chains.indexing_chain import index_video_to_pinecone, check_if_video_indexed

print("=" * 60)
print("Phase 5: Pure LangChain Indexing Pipeline")
print("=" * 60)

video_id = "kY14KfZQ1TI"

# Check if already indexed
print(f"\nChecking if {video_id} is indexed...")
is_indexed = check_if_video_indexed(video_id)
print(f"  Is indexed: {is_indexed}")

if is_indexed:
    print("\n✓ Video already indexed!")
    print("  Using LangChain's existing index connection")
    print("  No re-indexing needed (saves time and cost)")
else:
    print("\n  Video not indexed. Starting LangChain pipeline...")
    print("\n  LangChain will:")
    print("    1. Split text with RecursiveCharacterTextSplitter")
    print("    2. Create LangChain Documents")
    print("    3. Generate embeddings with OpenAIEmbeddings")
    print("    4. Store via PineconeVectorStore.from_documents()")
    print("\n  Expected time: ~20-30 seconds")
    
    # One function call does everything!
    result = index_video_to_pinecone(video_id)
    
    print("\n" + "=" * 60)
    print("✓ LangChain Pipeline Complete!")
    print("=" * 60)
    print(f"\n  Video ID: {result['video_id']}")
    print(f"  Status: {result['status']}")
    print(f"  Chunks: {result['num_chunks']}")
    print(f"  Duration: {result['duration_seconds']:.2f} seconds")

print("\n" + "=" * 60)
print("Phase 5 Complete - Pure LangChain!")
print("=" * 60)
print("\nLangChain Components Used:")
print("  ✓ RecursiveCharacterTextSplitter (chunking)")
print("  ✓ OpenAIEmbeddings (embedding generation)")
print("  ✓ PineconeVectorStore (vector storage)")
print("  ✓ Document schema (data format)")
print("\nReady for Phase 6: Build retrieval chains!")