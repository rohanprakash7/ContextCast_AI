"""
Citation Handler for Answer Generation
Extracts and validates citations from LLM responses
"""

import re
from typing import List, Dict
from langchain.schema import Document

from config.logging_config import get_logger

logger = get_logger(__name__)


def extract_citations(text: str) -> List[int]:
    """
    Extract citation chunk IDs from text
    
    Looks for patterns like [Chunk 0], [Chunk 1], etc.
    
    Args:
        text: Text containing citations
    
    Returns:
        List of cited chunk IDs
    
    Example:
        >>> answer = "RAG is useful [Chunk 0]. It helps [Chunk 2]."
        >>> citations = extract_citations(answer)
        >>> print(citations)
        [0, 2]
    """
    # Pattern: [Chunk N] where N is a number
    pattern = r'\[Chunk\s+(\d+)\]'
    matches = re.findall(pattern, text)
    
    # Convert to integers
    chunk_ids = [int(match) for match in matches]
    
    logger.debug(f"Extracted {len(chunk_ids)} citations: {chunk_ids}")
    
    return chunk_ids


def add_source_info(
    answer: str,
    retrieved_docs: List[Document]
) -> Dict[str, any]:
    """
    Add source information to answer
    
    Extracts citations and maps them to source documents.
    
    Args:
        answer: Generated answer (may contain citations like [Chunk 0])
        retrieved_docs: Original LangChain Documents from retrieval
    
    Returns:
        Dictionary with answer and source information
    
    Example:
        >>> answer = "RAG is useful [Chunk 0]"
        >>> docs = retriever.invoke("What is RAG?")
        >>> result = add_source_info(answer, docs)
        >>> print(result['sources'])
    """
    # Extract cited chunk IDs
    cited_ids = extract_citations(answer)
    
    # Create chunk ID to document mapping
    doc_map = {doc.metadata.get("chunk_id", i): doc for i, doc in enumerate(retrieved_docs)}
    
    # Get source information for cited chunks
    sources = []
    for chunk_id in cited_ids:
        if chunk_id in doc_map:
            doc = doc_map[chunk_id]
            sources.append({
                "chunk_id": chunk_id,
                "text": doc.page_content,
                "video_id": doc.metadata.get("video_id", "unknown"),
                "metadata": doc.metadata
            })
        else:
            logger.warning(f"Citation refers to non-existent chunk: {chunk_id}")
    
    result = {
        "answer": answer,
        "citations": cited_ids,
        "sources": sources,
        "num_citations": len(cited_ids),
        "num_valid_citations": len(sources)
    }
    
    logger.info(
        f"Added source info: {len(cited_ids)} citations, "
        f"{len(sources)} valid sources"
    )
    
    return result


def remove_citations(text: str) -> str:
    """
    Remove citation markers from text
    
    Useful if you want clean text without [Chunk N] markers.
    
    Args:
        text: Text with citations
    
    Returns:
        Text without citations
    
    Example:
        >>> text = "RAG is useful [Chunk 0]."
        >>> clean = remove_citations(text)
        >>> print(clean)
        "RAG is useful."
    """
    # Remove [Chunk N] patterns
    pattern = r'\s*\[Chunk\s+\d+\]\s*'
    clean_text = re.sub(pattern, ' ', text)
    
    # Clean up extra spaces
    clean_text = ' '.join(clean_text.split())
    
    return clean_text