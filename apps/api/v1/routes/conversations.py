# apps/api/v1/routes/conversations.py
import time
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from database.connection import get_db_session
from services.agent.core import AegisAgent
from services.agent.memory import ChatMemoryManager
from services.rag.evaluator import RAGEvaluator
from core.metrics import (
    CHAT_REQUESTS_TOTAL,
    RAG_CONTEXT_PRECISION,
    RAG_FAITHFULNESS,
    CHAT_LATENCY_SECONDS,
)

# Instantiate the FastAPI APIRouter
router = APIRouter()

# Instantiate service dependencies
memory_manager = ChatMemoryManager()
evaluator = RAGEvaluator()
agent = AegisAgent()


class ChatRequest(BaseModel):
    session_id: str
    message: str


@router.post("/chat")
async def chat_endpoint(
    request: ChatRequest, db: AsyncSession = Depends(get_db_session)
):
    start_time = time.perf_counter()
    try:
        # 1. Fetch recent conversation history from Redis
        history = memory_manager.get_recent_history(request.session_id)

        # 2. Run agent with query and history
        agent_response = await agent.run(
            query=request.message, history=history
        )

        # 3. Extract retrieved contexts and generated answer
        retrieved_docs = agent_response.get("retrieved_contexts", [])
        answer_text = agent_response.get("output", "")

        # 4. Evaluate response quality
        eval_metrics = evaluator.evaluate(
            query=request.message,
            response=answer_text,
            retrieved_contexts=retrieved_docs,
        )

        # 5. Record Prometheus Metrics
        RAG_CONTEXT_PRECISION.observe(
            eval_metrics.get("context_precision", 0.0)
        )
        RAG_FAITHFULNESS.observe(eval_metrics.get("faithfulness", 0.0))
        CHAT_REQUESTS_TOTAL.labels(status="success").inc()
        CHAT_LATENCY_SECONDS.observe(time.perf_counter() - start_time)

        # 6. Persist turn to Redis and PostgreSQL
        memory_manager.add_turn(
            db=db,
            session_id=request.session_id,
            user_msg=request.message,
            assistant_msg=answer_text,
        )

        return {
            "session_id": request.session_id,
            "response": answer_text,
            "evaluation": eval_metrics,
        }
    except Exception as e:
        CHAT_REQUESTS_TOTAL.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))