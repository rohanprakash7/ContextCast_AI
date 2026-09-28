"""
Test Phase 7: Prompt Templates
Tests for LangChain prompt template functionality
"""

import pytest
from langchain.schema import Document
from augmentation.prompt_templates import (
    QA_PROMPT,
    QA_PROMPT_WITH_CITATIONS,
    format_docs_for_prompt,
    create_qa_chain_prompt
)


def test_qa_prompt_template():
    """Test basic QA prompt template"""
    # Create test context and question
    context = "Deep learning is a subset of machine learning."
    question = "What is deep learning?"
    
    # Format prompt using LangChain
    formatted_prompt = QA_PROMPT.format(context=context, question=question)
    
    # Check prompt contains our inputs
    assert "Deep learning" in formatted_prompt
    assert "What is deep learning?" in formatted_prompt
    assert "Answer ONLY using information" in formatted_prompt
    
    print("\n✓ QA prompt template works")
    print(f"\nFormatted prompt preview:")
    print(formatted_prompt[:200])


def test_qa_prompt_with_citations():
    """Test QA prompt with citations"""
    context = "[Chunk 0]\nDeep learning uses neural networks.\n\n[Chunk 1]\nIt learns from data."
    question = "How does deep learning work?"
    
    formatted_prompt = QA_PROMPT_WITH_CITATIONS.format(context=context, question=question)
    
    assert "[Chunk 0]" in formatted_prompt
    assert "citations" in formatted_prompt.lower()
    
    print("\n✓ QA prompt with citations works")


def test_format_docs_without_ids():
    """Test formatting LangChain Documents without chunk IDs"""
    docs = [
        Document(page_content="First chunk about AI", metadata={"chunk_id": 0}),
        Document(page_content="Second chunk about ML", metadata={"chunk_id": 1})
    ]
    
    context = format_docs_for_prompt(docs, include_chunk_ids=False)
    
    assert "First chunk about AI" in context
    assert "Second chunk about ML" in context
    assert "[Chunk" not in context  # No IDs
    
    print("\n✓ Document formatting (no IDs) works")


def test_format_docs_with_ids():
    """Test formatting with chunk IDs for citations"""
    docs = [
        Document(page_content="AI is powerful", metadata={"chunk_id": 0}),
        Document(page_content="ML learns patterns", metadata={"chunk_id": 1})
    ]
    
    context = format_docs_for_prompt(docs, include_chunk_ids=True)
    
    assert "[Chunk 0]" in context
    assert "[Chunk 1]" in context
    assert "AI is powerful" in context
    
    print("\n✓ Document formatting (with IDs) works")
    print(f"\nContext with IDs:\n{context}")


def test_create_qa_chain_prompt_standard():
    """Test getting standard prompt"""
    prompt = create_qa_chain_prompt(include_citations=False)
    
    assert prompt == QA_PROMPT
    print("\n✓ Standard QA prompt selected")


def test_create_qa_chain_prompt_citations():
    """Test getting citations prompt"""
    prompt = create_qa_chain_prompt(include_citations=True)
    
    assert prompt == QA_PROMPT_WITH_CITATIONS
    print("\n✓ Citations prompt selected")


def test_prompt_with_real_retrieval():
    """Test prompt with actual retrieved documents"""
    from retrieval.simple_retriever import create_simple_retriever
    
    # Retrieve real chunks
    retriever = create_simple_retriever(video_id="kY14KfZQ1TI", top_k=3)
    docs = retriever.invoke("What is deep learning?")
    
    # Format for prompt
    context = format_docs_for_prompt(docs, include_chunk_ids=False)
    question = "What is deep learning?"
    
    # Create final prompt
    final_prompt = QA_PROMPT.format(context=context, question=question)
    
    print("\n✓ Prompt with real retrieval works")
    print(f"\nPrompt length: {len(final_prompt)} characters")
    print(f"Context from {len(docs)} retrieved chunks")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])