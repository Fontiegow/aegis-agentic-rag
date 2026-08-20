# database/qdrant_client.py
from qdrant_client import QdrantClient
from qdrant_client.http import models
from core.config import settings
from core.rag_config import rag_settings

qdrant = QdrantClient(
    host=getattr(settings, "QDRANT_HOST", "localhost"),
    port=getattr(settings, "QDRANT_PORT", 6333),
    check_compatibility=False,
)

def init_qdrant_collection():
    if not qdrant.collection_exists(rag_settings.QDRANT_COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=rag_settings.QDRANT_COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=rag_settings.VECTOR_DIMENSION, 
                distance=models.Distance.COSINE
            ),
        )