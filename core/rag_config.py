# core/rag_config.py
from pydantic_settings import BaseSettings

class RAGSettings(BaseSettings):
    QDRANT_COLLECTION_NAME: str = "aegis_knowledge"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    VECTOR_DIMENSION: int = 384  # Dimension for bge-small-en-v1.5

rag_settings = RAGSettings()