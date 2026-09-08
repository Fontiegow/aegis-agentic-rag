from services.agent.memory import ChatMemoryManager
from services.rag.evaluator import RAGEvaluator

def test_memory_and_eval():
    session_id = "test_session_123"
    
    # 1. Test Redis Memory
    memory = ChatMemoryManager()
    memory.redis_client.rpush(
        f"chat:history:{session_id}", 
        '{"role": "user", "content": "Who leads the Ultramarines?"}'
    )
    history = memory.get_recent_history(session_id)
    print("Redis Memory Output:", history)

    # 2. Test Evaluator
    evaluator = RAGEvaluator()
    metrics = evaluator.evaluate(
        query="Who leads the Ultramarines?",
        response="Roboute Guilliman leads the Ultramarines.",
        retrieved_contexts=["Roboute Guilliman is the Primarch and Lord Commander."]
    )
    print("Evaluator Output:", metrics)

if __name__ == "__main__":
    test_memory_and_eval()