# test_rag.py
import uuid
from database.qdrant_client import qdrant, init_qdrant_collection
from core.rag_config import rag_settings
from services.rag.processor import RAGProcessor
from services.rag.retriever import RAGRetriever
from qdrant_client.http import models

def test_pipeline():
    print("1. Initializing Qdrant Collection...")
    init_qdrant_collection()

    print("2. Testing Chunking & Embeddings...")
    processor = RAGProcessor()
    sample_text = (
        "The Emperor of Mankind is the immortal ruler of the Imperium of Man. "
        "For over ten thousand years, he has sat enthroned upon the Golden Throne on Terra. "
        "The Space Marines, or Adeptus Astartes, are genetically engineered transhuman warriors created by the Emperor."
    )
    
    chunks = processor.chunk_text(sample_text)
    print(f"   -> Generated {len(chunks)} chunk(s).")
    
    embeddings = processor.generate_embeddings(chunks)
    print(f"   -> Generated embeddings with vector dimension: {len(embeddings[0])}")

    print("3. Upserting Sample Vector into Qdrant...")
    points = [
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[i],
            payload={
                "text": chunks[i],
                "metadata": {"source": "warhammer_lore_test.txt", "chunk_index": i}
            }
        )
        for i in range(len(chunks))
    ]
    qdrant.upsert(
        collection_name=rag_settings.QDRANT_COLLECTION_NAME,
        points=points
    )
    print("   -> Point successfully inserted into Qdrant!")

    print("4. Testing RAGRetriever Search...")
    retriever = RAGRetriever()
    results = retriever.search(query="Who are the Space Marines?", limit=2, score_threshold=0.3)
    
    print("\nSearch Results:")
    for res in results:
        print(f" - [Score: {res['score']:.4f}] {res['text']} (Source: {res['metadata'].get('source')})")

if __name__ == "__main__":
    test_pipeline()