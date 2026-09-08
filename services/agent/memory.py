import json
from typing import List, Dict, Any
import redis
from sqlalchemy.orm import Session
# Adjust imports below based on your actual DB models location
# from db.models import Message, Conversation 

class ChatMemoryManager:
    def __init__(self, redis_url: str = "redis://localhost:6379/0", max_history: int = 10):
        self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.max_history = max_history

    def _get_redis_key(self, session_id: str) -> str:
        return f"chat:history:{session_id}"

    def get_recent_history(self, session_id: str) -> List[Dict[str, str]]:
        key = self._get_redis_key(session_id)
        raw_msgs = self.redis_client.lrange(key, 0, -1)
        return [json.loads(m) for m in raw_msgs]

    def add_turn(self, db: Session, session_id: str, user_msg: str, assistant_msg: str):
        key = self._get_redis_key(session_id)
        
        # 1. Push to Redis sliding window
        self.redis_client.rpush(key, json.dumps({"role": "user", "content": user_msg}))
        self.redis_client.rpush(key, json.dumps({"role": "assistant", "content": assistant_msg}))
        self.redis_client.ltrim(key, -self.max_history * 2, -1)
        
        # 2. Persist to Postgres (if DB models are integrated)
        # db.add(Message(session_id=session_id, role="user", content=user_msg))
        # db.add(Message(session_id=session_id, role="assistant", content=assistant_msg))
        # db.commit()