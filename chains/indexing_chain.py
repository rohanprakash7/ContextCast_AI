"""
Indexing Chain - YouTube transcript processing and Pinecone indexing
"""

import os
import time
from typing import Dict, Any, List
from datetime import datetime

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from youtube_transcript_api import YouTubeTranscriptApi

from config.settings import settings
from config.logging_config import get_logger

logger = get_logger(__name__)


def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.OPENAI_EMBEDDING_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
    )


def _set_pinecone_env() -> None:
    api_key = settings.PINECONE_API_KEY
    if not api_key:
        raise ValueError("Missing Pinecone API key")
    os.environ["PINECONE_API_KEY"] = api_key


def _load_youtube_transcript(video_id: str) -> str:
    ytt_api = YouTubeTranscriptApi()
    transcript = ytt_api.fetch(video_id)
    return " ".join(chunk.text for chunk in transcript)


def _create_documents(transcript_text: str, video_id: str) -> List[Document]:
    return [
        Document(
            page_content=transcript_text,
            metadata={"video_id": video_id},
        )
    ]


def index_video_to_pinecone(
    video_id: str,
    namespace: str = "",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> Dict[str, Any]:
    start_time = time.time()
    logger.info(f"Starting indexing for video {video_id}")

    _set_pinecone_env()

    transcript_text = _load_youtube_transcript(video_id)
    docs = _create_documents(transcript_text, video_id)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_docs = splitter.split_documents(docs)

    embeddings = _get_embeddings()

    PineconeVectorStore.from_documents(
        documents=split_docs,
        embedding=embeddings,
        index_name=settings.PINECONE_INDEX_NAME,
        namespace=namespace,
    )

    duration = time.time() - start_time
    logger.info(f"Indexed video {video_id} successfully in {duration:.2f}s")

    return {
        "video_id": video_id,
        "status": "success",
        "num_chunks": len(split_docs),
        "transcript_chars": len(transcript_text),
        "duration_seconds": round(duration, 2),
        "timestamp": datetime.utcnow().isoformat()
    }


def check_if_video_indexed(video_id: str, namespace: str = "") -> bool:
    try:
        _set_pinecone_env()
        embeddings = _get_embeddings()

        vectorstore = PineconeVectorStore(
            index_name=settings.PINECONE_INDEX_NAME,
            embedding=embeddings,
            namespace=namespace,
        )

        results = vectorstore.similarity_search(
            query=video_id,
            k=1,
            filter={"video_id": video_id},
        )

        return len(results) > 0

    except Exception as e:
        logger.error(f"Error checking video index status for {video_id}: {e}")
        return False