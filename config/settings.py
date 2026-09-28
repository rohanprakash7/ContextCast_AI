"""
Configuration Management for Conversational RAG Platform
Handles all environment variables and application settings
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables
    Uses Pydantic for validation and type safety
    """
    
    # ============================================
    # OpenAI Configuration
    # ============================================
    OPENAI_API_KEY: str
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"
    OPENAI_TEMPERATURE: float = 0.2
    
    # ============================================
    # Vector Store Configuration
    # ============================================
    VECTOR_STORE_TYPE: str = "pinecone"  # Options: faiss, pinecone
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_INDEX_NAME: str = "youtube-rag-index"
    
    # Embedding dimensions (must match Pinecone index!)
    EMBEDDING_DIMENSIONS: int = 1536
    
    # ============================================
    # Retrieval Configuration
    # ============================================
    RETRIEVAL_TOP_K: int = 4
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # ============================================
    # API Configuration
    # ============================================
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    
    # ============================================
    # CORS Configuration
    # ============================================
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:5500,http://localhost:5501,http://127.0.0.1:5500,http://127.0.0.1:5501,chrome-extension://*"
    
    @property
    def allowed_origins_list(self) -> List[str]:
        """Convert comma-separated origins to list"""
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]
    
    # ============================================
    # Logging Configuration
    # ============================================
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"
    
    # ============================================
    # LangSmith Configuration (Optional)
    # ============================================
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_PROJECT: str = "conversational-rag-youtube"
    
    # ============================================
    # Evaluation Configuration (Optional)
    # ============================================
    ENABLE_EVALUATION: bool = False
    
    # ============================================
    # Pydantic Settings Configuration
    # ============================================
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"  # Ignore extra fields in .env
    )
    
    def validate_vector_store_config(self) -> None:
        """
        Validate vector store configuration
        Raises ValueError if configuration is invalid
        """
        if self.VECTOR_STORE_TYPE == "pinecone":
            if not self.PINECONE_API_KEY:
                raise ValueError(
                    "PINECONE_API_KEY is required when VECTOR_STORE_TYPE is 'pinecone'"
                )


# ============================================
# Global Settings Instance
# ============================================
settings = Settings()

# Validate configuration on import
settings.validate_vector_store_config()