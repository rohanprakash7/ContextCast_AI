"""
Test Phase 8: LLM Generation
Tests for answer generation and citation handling
"""

import pytest
from langchain.schema import Document
from generation.llm_client import create_llm, create_llm_chain
from generation.citation_handler import extract_citations, add_source_info, remove_citations


def test_create_llm():
    """Test creating LangChain LLM"""
    llm = create_llm()
    
    assert llm is not None
    assert hasattr(llm, 'invoke')
    
    print("\n✓ Created LangChain ChatOpenAI")


def test_llm_simple_generation():
    """Test basic LLM generation"""
    llm = create_llm(temperature=0.0, max_tokens=50)
    
    response = llm.invoke("What is 2+2? Answer in one word.")
    
    # Response should be an AIMessage
    assert hasattr(response, 'content')
    
    print(f"\n✓ LLM generated response")
    print(f"  Response: {response.content}")


def test_llm_chain_with_parser():
    """Test LLM chain with output parser"""
    chain = create_llm_chain(temperature=0.0, max_tokens=50)
    
    response = chain.invoke("What is AI? Answer in one sentence.")
    
    # Should return string (not AIMessage) due to parser
    assert isinstance(response, str)
    
    print(f"\n✓ LLM chain with parser works")
    print(f"  Response: {response}")


def test_extract_citations():
    """Test extracting citations from text"""
    text = "RAG is useful [Chunk 0]. It combines retrieval [Chunk 1] and generation [Chunk 2]."
    citations = extract_citations(text)
    
    assert citations == [0, 1, 2]
    
    print(f"\n✓ Extracted citations: {citations}")


def test_extract_citations_no_citations():
    """Test text without citations"""
    text = "This has no citations."
    citations = extract_citations(text)
    
    assert citations == []


def test_add_source_info():
    """Test adding source information to answer"""
    answer = "Deep learning is powerful [Chunk 0]. Neural networks learn [Chunk 1]."
    
    docs = [
        Document(
            page_content="Deep learning uses neural networks...",
            metadata={"chunk_id": 0, "video_id": "kY14KfZQ1TI"}
        ),
        Document(
            page_content="Neural networks learn patterns...",
            metadata={"chunk_id": 1, "video_id": "kY14KfZQ1TI"}
        )
    ]
    
    result = add_source_info(answer, docs)
    
    assert result['num_citations'] == 2
    assert result['num_valid_citations'] == 2
    assert len(result['sources']) == 2
    
    print(f"\n✓ Source info added")
    print(f"  Citations: {result['citations']}")
    print(f"  Valid sources: {result['num_valid_citations']}")


def test_remove_citations():
    """Test removing citation markers"""
    text = "RAG is useful [Chunk 0]. It helps [Chunk 2]."
    clean = remove_citations(text)
    
    assert "[Chunk" not in clean
    assert "RAG is useful" in clean
    
    print(f"\n✓ Citations removed")
    print(f"  Original: {text}")
    print(f"  Clean: {clean}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])