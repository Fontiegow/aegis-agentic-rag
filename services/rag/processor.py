# services/rag/processor.py
import uuid
from typing import List, Dict, Any, Optional
from qdrant_client.http import models
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastembed import TextEmbedding
from core.rag_config import rag_settings
from database.qdrant_client import qdrant

class RAGProcessor:
    def __init__(self):
        self.embedding_model = TextEmbedding(model_name=rag_settings.EMBEDDING_MODEL)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=rag_settings.CHUNK_SIZE,
            chunk_overlap=rag_settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def chunk_text(self, text: str) -> List[str]:
        return self.splitter.split_text(text)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [list(e) for e in self.embedding_model.embed(texts)]

    def ingest_text(
        self, 
        text: str, 
        source: str = "manual_input", 
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Chunks, embeds, and upserts text into Qdrant."""
        chunks = self.chunk_text(text)
        if not chunks:
            return {"status": "empty", "chunks_ingested": 0, "doc_id": None}

        embeddings = self.generate_embeddings(chunks)
        doc_id = str(uuid.uuid4())

        points = []
        for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            payload = {
                "text": chunk,
                "source": source,
                "chunk_index": idx,
                "doc_id": doc_id,
            }
            if extra_metadata:
                payload.update(extra_metadata)

            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=payload
                )
            )

        qdrant.upsert(
            collection_name=rag_settings.QDRANT_COLLECTION_NAME,
            points=points
        )

        return {
            "doc_id": doc_id,
            "chunks_ingested": len(points),
            "status": "success"
        }