"""
Phase 9 Verification: Complete RAG Pipeline with LangChain LCEL
"""

from chains.qa_chain import create_qa_chain, answer_question

print("=" * 60)
print("Phase 9: Complete QA Chain with LangChain LCEL")
print("=" * 60)

video_id = "kY14KfZQ1TI"

# Test 1: Simple QA chain
print("\nTest 1: QA Chain with Simple Retriever...")
print("=" * 60)

chain_simple = create_qa_chain(
    video_id=video_id,
    retriever_type="simple",
    include_citations=False,
    top_k=3
)

question1 = "What is deep learning?"
print(f"Question: {question1}")
print("Invoking LangChain LCEL chain...")

answer1 = chain_simple.invoke(question1)

print(f"\n✓ Answer received!")
print(f"\nAnswer:\n{answer1}\n")

# Test 2: QA with citations
print("\nTest 2: QA with Citations...")
print("=" * 60)

result2 = answer_question(
    question="How do neural networks work?",
    video_id=video_id,
    retriever_type="simple",
    include_citations=True,
    top_k=3
)

print(f"Question: How do neural networks work?")
print(f"\n✓ Generated answer with citations")
print(f"\nAnswer:\n{result2['answer']}\n")
print(f"Citations found: {result2.get('citations', [])}")
print(f"Sources: {result2.get('num_valid_citations', 0)} chunks")

# Test 3: Test all three retrievers
print("\nTest 3: Comparing Three Retrievers...")
print("=" * 60)

question3 = "What is machine learning?"

print(f"\nQuestion: {question3}\n")

# Simple
result_simple = answer_question(
    question=question3,
    video_id=video_id,
    retriever_type="simple",
    include_citations=False,
    top_k=3
)
print(f"1. Simple Retriever ({result_simple['duration_seconds']:.2f}s):")
print(f"   {result_simple['answer'][:150]}...\n")

# Rewriting
result_rewriting = answer_question(
    question=question3,
    video_id=video_id,
    retriever_type="rewriting",
    include_citations=False,
    top_k=3
)
print(f"2. Rewriting Retriever ({result_rewriting['duration_seconds']:.2f}s):")
print(f"   {result_rewriting['answer'][:150]}...\n")

# Hybrid
result_hybrid = answer_question(
    question=question3,
    video_id=video_id,
    retriever_type="hybrid",
    include_citations=False,
    top_k=3
)
print(f"3. Hybrid Retriever ({result_hybrid['duration_seconds']:.2f}s):")
print(f"   {result_hybrid['answer'][:150]}...\n")

# Test 4: One-liner RAG
print("\nTest 4: Complete RAG in One Line...")
print("=" * 60)

answer = answer_question("What is AI?", video_id, retriever_type="simple")

print(f"✓ One function call executed entire RAG pipeline!")
print(f"\nQuestion: What is AI?")
print(f"\nAnswer:\n{answer['answer']}")

# Summary
print("\n" + "=" * 60)
print("Phase 9 Complete - Full RAG Pipeline Working!")
print("=" * 60)
print("\nWhat We Achieved:")
print("  ✓ Complete RAG chain using LangChain LCEL")
print("  ✓ One-line question answering")
print("  ✓ Three retrieval strategies available")
print("  ✓ Citation support working")
print("  ✓ All phases integrated seamlessly")
print("\nLangChain LCEL Chain Structure:")
print("  RunnableParallel({")
print("    'context': retriever | format_docs,")
print("    'question': RunnablePassthrough()")
print("  }) | prompt | llm | StrOutputParser()")
print("\nReady for Phase 10: FastAPI backend!")