# test_env.py
from config.settings import settings

print("Checking environment variables...")
print(f"OPENAI_API_KEY: {settings.OPENAI_API_KEY[:10]}..." if settings.OPENAI_API_KEY else "NOT SET")
print(f"PINECONE_API_KEY: {settings.PINECONE_API_KEY[:10]}..." if settings.PINECONE_API_KEY else "NOT SET")
print(f"PINECONE_INDEX_NAME: {settings.PINECONE_INDEX_NAME}")
print(f"EMBEDDING_DIMENSIONS: {settings.EMBEDDING_DIMENSIONS}")