import uuid
from typing import List, Dict, Any
from database.qdrant_client import qdrant
from services.rag.processor import RAGProcessor
from core.rag_config import rag_settings

class RAGRetriever:
    """Handles semantic search and context retrieval from Qdrant."""

    def __init__(self):
        self.processor = RAGProcessor()
        self.collection_name = rag_settings.QDRANT_COLLECTION_NAME

    def search(
        self, 
        query: str, 
        limit: int = 5, 
        score_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Embeds the query and retrieves the most relevant chunks.
        """
        # 1. Embed the user query (FastEmbed returns a list)
        query_vector = self.processor.generate_embeddings([query])[0]

        # 2. Perform vector similarity search via query_points
        response = qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )

        # 3. Format and return the retrieved context
        retrieved_chunks = []
        for hit in response.points:
            retrieved_chunks.append({
                "id": str(hit.id),
                "score": hit.score,
                "text": hit.payload.get("text", "") if hit.payload else "",
                "metadata": hit.payload.get("metadata", {}) if hit.payload else {}
            })

        return retrieved_chunks