from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession

from database.connection import get_db_session
from services.agent.core import AegisAgent
from services.agent.memory import ChatMemoryManager
from services.rag.evaluator import RAGEvaluator

router = APIRouter()
memory_manager = ChatMemoryManager()
evaluator = RAGEvaluator()
agent = AegisAgent()

class ChatRequest(BaseModel):
    session_id: str
    message: str

@router.post("/chat")
async def chat_endpoint(request: ChatRequest, db: AsyncSession = Depends(get_db_session)):
    try:
        # 1. Fetch recent conversation history from Redis
        history = memory_manager.get_recent_history(request.session_id)
        
        # 2. Run agent with matching parameters (query and history)
        agent_response = await agent.run(
            query=request.message,
            history=history
        )
        
        # 3. Extract retrieved contexts and generated answer
        retrieved_docs = agent_response.get("retrieved_contexts", [])
        answer_text = agent_response.get("output", "")
        
        # 4. Evaluate response quality
        eval_metrics = evaluator.evaluate(
            query=request.message,
            response=answer_text,
            retrieved_contexts=retrieved_docs
        )
        
        # 5. Persist turn to Redis and PostgreSQL
        memory_manager.add_turn(
            db=db,
            session_id=request.session_id,
            user_msg=request.message,
            assistant_msg=answer_text
        )
        
        return {
            "session_id": request.session_id,
            "response": answer_text,
            "evaluation": eval_metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))