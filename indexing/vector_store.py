"""
Vector Store Operations using LangChain Pinecone Integration
Handles Pinecone v3+ API with LangChain compatibility
"""

import os
from typing import List, Dict, Optional
from langchain_pinecone import PineconeVectorStore as LangChainPinecone
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from pinecone import Pinecone

from config.logging_config import get_logger
from config.settings import settings
from utils.exceptions import VectorStoreError

logger = get_logger(__name__)


class PineconeVectorStore:
    """
    LangChain Pinecone vector store wrapper
    
    Uses langchain-pinecone for proper integration with Pinecone v3+
    """
    
    def __init__(self):
        """
        Initialize Pinecone with LangChain
        """
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index_name = settings.PINECONE_INDEX_NAME
        self.index = self.pc.Index(self.index_name)
        
        # Create LangChain embeddings
        self.embeddings = OpenAIEmbeddings(
            model=settings.OPENAI_EMBEDDING_MODEL,
            dimensions=settings.EMBEDDING_DIMENSIONS,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        logger.info(
            f"Initialized PineconeVectorStore with LangChain "
            f"(index={self.index_name}, dims={settings.EMBEDDING_DIMENSIONS})"
        )
    
    def add_chunks(
        self,
        chunks: List[Dict[str, any]],
        namespace: str = ""
    ) -> Dict[str, any]:
        """
        Add chunks using LangChain Pinecone integration
        """
        if not chunks:
            return {"added_count": 0}
        
        try:
            # Prepare texts and metadatas
            texts = []
            metadatas = []
            ids = []
            
            for chunk in chunks:
                texts.append(chunk["text"])
                
                video_id = chunk["metadata"]["video_id"]
                metadatas.append({
                    "video_id": video_id,
                    "chunk_id": chunk["chunk_id"],
                    "language": chunk["metadata"].get("language", "en"),
                    "source": chunk["metadata"].get("source", "youtube")
                })
                
                ids.append(f"{video_id}_{chunk['chunk_id']}")
            
            logger.info(f"Adding {len(texts)} documents via LangChain")
            
            # Use langchain-pinecone's from_texts
            vector_store = LangChainPinecone.from_texts(
                texts=texts,
                embedding=self.embeddings,
                index_name=self.index_name,
                namespace=namespace,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully added {len(texts)} documents")
            
            return {
                "added_count": len(texts),
                "vector_store": vector_store
            }
        
        except Exception as e:
            error_msg = f"Failed to add chunks: {e}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg)
    
    def check_if_indexed(self, video_id: str, namespace: str = "") -> Dict[str, any]:
        """Check if video is indexed"""
        try:
            dummy_vector = [0.0] * settings.EMBEDDING_DIMENSIONS
            
            results = self.index.query(
                vector=dummy_vector,
                filter={"video_id": video_id},
                top_k=1,
                namespace=namespace
            )
            
            is_indexed = len(results.matches) > 0
            
            return {
                "video_id": video_id,
                "is_indexed": is_indexed
            }
        except Exception as e:
            logger.error(f"Check failed: {e}")
            return {"video_id": video_id, "is_indexed": False}
    
    def get_retriever(self, k: int = 4, filter: Optional[Dict] = None, namespace: str = ""):
        """Get LangChain retriever"""
        vector_store = LangChainPinecone.from_existing_index(
            index_name=self.index_name,
            embedding=self.embeddings,
            namespace=namespace
        )
        
        search_kwargs = {"k": k}
        if filter:
            search_kwargs["filter"] = filter
        
        return vector_store.as_retriever(search_kwargs=search_kwargs)


def add_chunks_to_pinecone(chunks: List[Dict[str, any]]) -> Dict[str, any]:
    """Convenience function"""
    store = PineconeVectorStore()
    return store.add_chunks(chunks)