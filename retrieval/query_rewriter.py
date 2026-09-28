"""
Query Rewriting Retriever using LangChain's MultiQueryRetriever
Improves vague queries using LLM before retrieval
"""

import os
from typing import Optional
from langchain.retrievers import MultiQueryRetriever
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)

# Set environment variables for LangChain
os.environ["PINECONE_API_KEY"] = settings.PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY


def create_rewriting_retriever(
    video_id: Optional[str] = None,
    top_k: int = 4,
    namespace: str = ""
):
    """
    Create query rewriting retriever using LangChain's MultiQueryRetriever
    
    How it works:
    1. User asks vague question: "what's rag"
    2. LLM rewrites to better questions:
       - "What is Retrieval-Augmented Generation?"
       - "How does RAG work?"
       - "What are RAG benefits?"
    3. Retrieves for each variation
    4. Combines and deduplicates results
    
    From tutorial: Improves retrieval for vague queries by 5-10%
    
    Args:
        video_id: Optional video filter
        top_k: Results per query variation
        namespace: Pinecone namespace
    
    Returns:
        LangChain MultiQueryRetriever
    
    Example:
        >>> retriever = create_rewriting_retriever(video_id="kY14KfZQ1TI")
        >>> results = retriever.invoke("what's a transformer")
        >>> # LLM expands query, retrieves better results!
    """
    logger.info(f"Creating rewriting retriever: video_id={video_id}, top_k={top_k}")
    
    # Create embeddings
    embeddings = OpenAIEmbeddings(
        model=settings.OPENAI_EMBEDDING_MODEL,
        dimensions=settings.EMBEDDING_DIMENSIONS,
        openai_api_key=settings.OPENAI_API_KEY
    )
    
    # Create base retriever
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=settings.PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=namespace
    )
    
    search_kwargs = {"k": top_k}
    if video_id:
        search_kwargs["filter"] = {"video_id": video_id}
    
    base_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )
    
    # Create LLM for query generation
    llm = ChatOpenAI(
        model=settings.OPENAI_CHAT_MODEL,
        temperature=0.0,
        openai_api_key=settings.OPENAI_API_KEY
    )
    
    # Create MultiQueryRetriever (LangChain does query rewriting!)
    rewriting_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )
    
    logger.info("Created MultiQueryRetriever with automatic query rewriting")

    return rewriting_retriever