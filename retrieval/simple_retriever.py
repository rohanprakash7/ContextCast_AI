"""
Simple Retriever using LangChain's Pinecone Integration
Basic semantic similarity search - the baseline retriever
"""

import os
from typing import List, Optional

from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)

# Make keys available to downstream libs
os.environ["PINECONE_API_KEY"] = settings.PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY


def create_simple_retriever(
    video_id: Optional[str] = None,
    top_k: int = 4,
    namespace: str = ""
):
    """
    Create a simple similarity search retriever (baseline)
    """
    logger.info(f"Creating simple retriever: video_id={video_id}, top_k={top_k}")

    embeddings = OpenAIEmbeddings(
    model=settings.OPENAI_EMBEDDING_MODEL,
    openai_api_key=settings.OPENAI_API_KEY,
)

    vector_store = PineconeVectorStore.from_existing_index(
        index_name=settings.PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=namespace,
    )

    search_kwargs = {"k": top_k}
    if video_id:
        search_kwargs["filter"] = {"video_id": video_id}

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )

    logger.info("Created simple similarity retriever")
    return retriever


def retrieve_chunks(
    question: str,
    video_id: Optional[str] = None,
    top_k: int = 4
) -> List[Document]:
    """
    Retrieve relevant chunks for a question
    """
    logger.info(f"Retrieving for question: '{question[:50]}...'")

    retriever = create_simple_retriever(video_id=video_id, top_k=top_k)
    results = retriever.invoke(question)

    logger.info(f"Retrieved {len(results)} chunks")
    return results