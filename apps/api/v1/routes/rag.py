from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from services.rag.retriever import RAGRetriever
from services.rag.processor import RAGProcessor

router = APIRouter()
retriever = RAGRetriever()
processor = RAGProcessor()

class SearchRequest(BaseModel):
    query: str
    limit: int = Field(default=5, ge=1, le=20)
    score_threshold: float = Field(default=0.2, ge=0.0, le=1.0)

class IngestRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source: str = Field(default="api_ingest")
    metadata: Optional[Dict[str, Any]] = None

@router.post("/search")
async def search_rag(request: SearchRequest):
    try:
        results = retriever.search(
            query=request.query,
            limit=request.limit,
            score_threshold=request.score_threshold
        )
        return {"value": results, "Count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest")
async def ingest_rag(request: IngestRequest):
    try:
        res = processor.ingest_text(
            text=request.text,
            source=request.source,
            extra_metadata=request.metadata
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))