"""
Phase 6 Verification: All Three Retrieval Strategies
"""

from retrieval.simple_retriever import create_simple_retriever
from retrieval.query_rewriter import create_rewriting_retriever
from retrieval.hybrid_retriever import create_hybrid_retriever

print("=" * 60)
print("Phase 6: Three Retrieval Strategies")
print("=" * 60)

video_id = "kY14KfZQ1TI"
question = "What is deep learning?"

# Strategy 1: Simple Similarity Search
print("\n" + "=" * 60)
print("Strategy 1: Simple Similarity Search (Baseline)")
print("=" * 60)
print(f"Question: '{question}'")

retriever_simple = create_simple_retriever(video_id=video_id, top_k=3)
results_simple = retriever_simple.invoke(question)

print(f"✓ Retrieved {len(results_simple)} chunks")
print(f"\nTop result:")
print(f"  {results_simple[0].page_content[:150]}...")

# Strategy 2: Query Rewriting with MultiQueryRetriever
print("\n" + "=" * 60)
print("Strategy 2: Query Rewriting (LangChain MultiQueryRetriever)")
print("=" * 60)
vague_question = "what's that ml thing"
print(f"Vague question: '{vague_question}'")
print("  (LangChain will rewrite this to better variations)")

retriever_rewriting = create_rewriting_retriever(video_id=video_id, top_k=3)
results_rewriting = retriever_rewriting.invoke(vague_question)

print(f"✓ Retrieved {len(results_rewriting)} chunks (after rewriting)")
print(f"\nTop result:")
print(f"  {results_rewriting[0].page_content[:150]}...")

# Strategy 3: Hybrid (Semantic + Keyword)
print("\n" + "=" * 60)
print("Strategy 3: Hybrid Retrieval (Semantic + BM25)")
print("=" * 60)
keyword_question = "GPT-3 parameters training"
print(f"Question: '{keyword_question}'")
print("  (Will get semantic matches AND exact keyword matches)")

retriever_hybrid = create_hybrid_retriever(video_id=video_id, top_k=3)
results_hybrid = retriever_hybrid.invoke(keyword_question)

print(f"✓ Retrieved {len(results_hybrid)} chunks (hybrid search)")
print(f"\nTop result:")
print(f"  {results_hybrid[0].page_content[:150]}...")

# Summary
print("\n" + "=" * 60)
print("Phase 6 Complete - Three Retrieval Strategies!")
print("=" * 60)
print("\nStrategies Implemented:")
print("  1. Simple (Baseline): Pure semantic similarity")
print("  2. Query Rewriting: LLM improves vague queries")
print("  3. Hybrid: Semantic + keyword (BM25)")
print("\nAll using LangChain:")
print("  ✓ VectorStoreRetriever (simple)")
print("  ✓ MultiQueryRetriever (query rewriting)")
print("  ✓ EnsembleRetriever (hybrid)")
print("\nReady for Phase 7: Prompt engineering!")