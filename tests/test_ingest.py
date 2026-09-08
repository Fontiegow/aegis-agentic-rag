from services.rag.processor import RAGProcessor

def test_ingestion():
    processor = RAGProcessor()
    
    sample_text = (
        "Roboute Guilliman is the Primarch of the Ultramarines and "
        "Lord Commander of the Imperium."
    )
    
    result = processor.ingest_text(
        text=sample_text,
        source="test_guilliman.txt",
        extra_metadata={"faction": "Ultramarines"}
    )
    
    print("Ingestion Result:", result)

if __name__ == "__main__":
    test_ingestion()