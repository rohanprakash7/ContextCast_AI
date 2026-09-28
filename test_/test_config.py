from config.settings import settings
from config.logging_config import get_logger

logger = get_logger(__name__)

logger.info("Testing Phase 1 configuration")
logger.info(f"OpenAI Model: {settings.OPENAI_CHAT_MODEL}")
logger.info(f"Embedding Model: {settings.OPENAI_EMBEDDING_MODEL}")
logger.info(f"Vector Store: {settings.VECTOR_STORE_TYPE}")
logger.info(f"Chunk Size: {settings.CHUNK_SIZE}")
logger.info("Phase 1 configuration loaded successfully!")