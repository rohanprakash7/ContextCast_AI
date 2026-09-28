"""
Test Phase 9: Complete QA Chain
Tests for end-to-end RAG pipeline
"""

import pytest
from chains.qa_chain import create_qa_chain, answer_question


def test_create_qa_chain_simple():
    """Test creating QA chain with simple retriever"""
    chain = create_qa_chain(
        video_id="kY14KfZQ1TI",
        retriever_type="simple",
        include_citations=False,
        top_k=3
    )
    
    # Chain should be a Runnable
    assert chain is not None
    assert hasattr(chain, 'invoke')
    
    print("\n✓ Created QA chain (simple retriever)")


def test_qa_chain_invoke():
    """Test invoking QA chain end-to-end"""
    chain = create_qa_chain(
        video_id="kY14KfZQ1TI",
        retriever_type="simple",
        include_citations=False,
        top_k=3
    )
    
    # Invoke chain (one call does everything!)
    answer = chain.invoke("What is deep learning?")
    
    # Should return string
    assert isinstance(answer, str)
    assert len(answer) > 0
    
    print("\n✓ QA chain invoke working!")
    print(f"\nQuestion: What is deep learning?")
    print(f"\nAnswer:\n{answer}")


def test_answer_question_simple():
    """Test answer_question with simple retriever"""
    result = answer_question(
        question="What is deep learning?",
        video_id="kY14KfZQ1TI",
        retriever_type="simple",
        include_citations=False,
        top_k=3
    )
    
    # Check result structure
    assert "answer" in result
    assert "retrieved_chunks" in result
    assert "retriever_type" in result
    assert "duration_seconds" in result
    
    print("\n✓ answer_question (simple) working")
    print(f"  Answer length: {len(result['answer'])} chars")
    print(f"  Chunks used: {result['retrieved_chunks']}")
    print(f"  Duration: {result['duration_seconds']:.2f}s")


def test_answer_question_with_citations():
    """Test answer_question with citations enabled"""
    result = answer_question(
        question="What is deep learning?",
        video_id="kY14KfZQ1TI",
        retriever_type="simple",
        include_citations=True,
        top_k=3
    )
    
    # Should have citation info
    assert "citations" in result
    assert "sources" in result
    assert "num_citations" in result
    
    print("\n✓ answer_question (with citations) working")
    print(f"\n  Answer:\n  {result['answer']}")
    print(f"\n  Citations found: {result['citations']}")
    print(f"  Valid sources: {result['num_valid_citations']}")


def test_different_retrievers():
    """Test answer_question with different retriever types"""
    question = "How do neural networks learn?"
    
    # Simple
    result_simple = answer_question(
        question=question,
        video_id="kY14KfZQ1TI",
        retriever_type="simple",
        include_citations=False,
        top_k=2
    )
    
    print(f"\n✓ Simple retriever: {len(result_simple['answer'])} chars")
    
    # Rewriting
    result_rewriting = answer_question(
        question="what's ml",  # Vague query
        video_id="kY14KfZQ1TI",
        retriever_type="rewriting",
        include_citations=False,
        top_k=2
    )
    
    print(f"✓ Rewriting retriever: {len(result_rewriting['answer'])} chars")
    
    # Hybrid
    result_hybrid = answer_question(
        question=question,
        video_id="kY14KfZQ1TI",
        retriever_type="hybrid",
        include_citations=False,
        top_k=2
    )
    
    print(f"✓ Hybrid retriever: {len(result_hybrid['answer'])} chars")


def test_complete_rag_pipeline():
    """Test complete RAG pipeline end-to-end"""
    print("\n" + "=" * 60)
    print("Complete RAG Pipeline Test")
    print("=" * 60)
    
    question = "What is deep learning and how does it work?"
    
    result = answer_question(
        question=question,
        video_id="kY14KfZQ1TI",
        retriever_type="hybrid",
        include_citations=True,
        top_k=4
    )
    
    print(f"\nQuestion: {question}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nMetadata:")
    print(f"  Retrieved chunks: {result['retrieved_chunks']}")
    print(f"  Retriever type: {result['retriever_type']}")
    print(f"  Citations: {result.get('citations', [])}")
    print(f"  Duration: {result['duration_seconds']:.2f}s")
    
    print("\n" + "=" * 60)
    print("✓ Complete RAG Pipeline Working!")
    print("=" * 60)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])