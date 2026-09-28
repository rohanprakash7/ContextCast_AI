"""
Retrieval package — public factory for all retriever types.
"""

from typing import Optional

from retrieval.simple_retriever import create_simple_retriever
from retrieval.hybrid_retriever import create_hybrid_retriever
from retrieval.query_rewriter import create_rewriting_retriever


def create_retriever(
    video_id: Optional[str] = None,
    retriever_type: str = "simple",
    top_k: int = 4,
    namespace: str = "",
):
    """
    Factory that returns the requested retriever.

    Args:
        video_id: Filter results to a specific YouTube video.
        retriever_type: One of ``"simple"``, ``"hybrid"``, ``"rewriting"``.
        top_k: Number of documents to return.
        namespace: Pinecone namespace.

    Returns:
        A LangChain ``BaseRetriever`` instance.

    Raises:
        ValueError: If ``retriever_type`` is not recognised.
    """
    if retriever_type == "simple":
        return create_simple_retriever(video_id=video_id, top_k=top_k, namespace=namespace)
    elif retriever_type == "hybrid":
        return create_hybrid_retriever(video_id=video_id, top_k=top_k, namespace=namespace)
    elif retriever_type == "rewriting":
        return create_rewriting_retriever(video_id=video_id, top_k=top_k, namespace=namespace)
    else:
        raise ValueError(
            f"Invalid retriever_type {retriever_type!r}. "
            "Choose from: 'simple', 'hybrid', 'rewriting'."
        )


__all__ = [
    "create_retriever",
    "create_simple_retriever",
    "create_hybrid_retriever",
    "create_rewriting_retriever",
]
