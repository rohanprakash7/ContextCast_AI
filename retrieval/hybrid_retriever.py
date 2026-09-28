"""
Hybrid Retriever combining dense (semantic) and sparse (BM25) retrieval.

BM25 is built at query-time from a broader semantic search pool rather than
from a static dummy-query corpus. This ensures the keyword index covers
documents that are actually relevant to the question being asked.
"""

import os
from itertools import zip_longest
from typing import Any, List, Optional

from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pydantic import Field

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)

os.environ["PINECONE_API_KEY"] = settings.PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY


class HybridRetriever(BaseRetriever):
    """
    Dense + sparse hybrid retriever.

    At query time:
    1. Fetches top-k semantically similar docs (dense).
    2. Fetches a broader corpus (top_k * corpus_multiplier) using the same
       query so BM25 has genuinely relevant documents to re-rank over.
    3. Re-ranks the corpus with BM25 (keyword match).
    4. Interleaves dense and BM25 results (rank fusion) and returns top-k.
    """

    vector_store: Any = Field(exclude=True)
    video_id: Optional[str] = None
    top_k: int = 4
    corpus_multiplier: int = 5
    dense_weight: float = 0.7
    sparse_weight: float = 0.3

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        filter_kwargs = {"filter": {"video_id": self.video_id}} if self.video_id else {}

        # --- Dense retrieval ---
        dense_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k, **filter_kwargs},
        )
        dense_docs = dense_retriever.invoke(query)

        # --- BM25 corpus: broader semantic pool using the real query ---
        corpus_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.top_k * self.corpus_multiplier, **filter_kwargs},
        )
        corpus_docs = corpus_retriever.invoke(query)

        if not corpus_docs:
            logger.warning("BM25 corpus is empty; falling back to dense results")
            return dense_docs

        bm25_retriever = BM25Retriever.from_documents(corpus_docs)
        bm25_retriever.k = self.top_k
        bm25_docs = bm25_retriever.invoke(query)

        # --- Rank fusion: interleave dense and BM25, deduplicate ---
        seen: set = set()
        merged: List[Document] = []
        for dense_doc, bm25_doc in zip_longest(dense_docs, bm25_docs):
            for doc in (dense_doc, bm25_doc):
                if doc is None:
                    continue
                key = doc.page_content[:120]
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)
            if len(merged) >= self.top_k:
                break

        return merged[: self.top_k]


def create_hybrid_retriever(
    video_id: Optional[str] = None,
    top_k: int = 4,
    namespace: str = "",
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
) -> HybridRetriever:
    """
    Create a hybrid (dense + BM25) retriever.

    Args:
        video_id: Filter results to a specific YouTube video.
        top_k: Number of documents to return.
        namespace: Pinecone namespace.
        dense_weight: Relative weight for semantic results in rank fusion.
        sparse_weight: Relative weight for BM25 results in rank fusion.

    Returns:
        HybridRetriever instance (a LangChain BaseRetriever).

    Example:
        >>> retriever = create_hybrid_retriever(video_id="kY14KfZQ1TI")
        >>> results = retriever.invoke("What is GPT-3?")
    """
    logger.info(
        f"Creating hybrid retriever: video_id={video_id}, top_k={top_k}, "
        f"weights=({dense_weight}/{sparse_weight})"
    )

    embeddings = OpenAIEmbeddings(
        model=settings.OPENAI_EMBEDDING_MODEL,
        dimensions=settings.EMBEDDING_DIMENSIONS,
        openai_api_key=settings.OPENAI_API_KEY,
    )

    vector_store = PineconeVectorStore.from_existing_index(
        index_name=settings.PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=namespace,
    )

    retriever = HybridRetriever(
        vector_store=vector_store,
        video_id=video_id,
        top_k=top_k,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
    )

    logger.info("Created HybridRetriever (query-time BM25 corpus)")
    return retriever
