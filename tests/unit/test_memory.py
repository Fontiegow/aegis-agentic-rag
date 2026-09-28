# tests/unit/test_memory.py
import sys
from pathlib import Path

# Ensure root workspace directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import uuid
import pytest
from core.config import settings
from services.agent.memory import ChatMemoryManager


def test_add_turn_and_get_recent_history():
    """Verify adding a turn pushes messages to Redis and retrieves them in chronological order."""
    session_id = f"test-mem-{uuid.uuid4().hex[:8]}"
    memory = ChatMemoryManager(redis_url=settings.REDIS_URL, max_history=5)

    try:
        # Add a turn
        memory.add_turn(
            db=None,
            session_id=session_id,
            user_msg="Who is Rogal Dorn?",
            assistant_msg="Rogal Dorn is the Primarch of the Imperial Fists.",
        )

        # Retrieve recent history
        history = memory.get_recent_history(session_id=session_id)

        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "Who is Rogal Dorn?"}
        assert history[1] == {
            "role": "assistant",
            "content": "Rogal Dorn is the Primarch of the Imperial Fists.",
        }
    finally:
        # Cleanup test key from Redis
        memory.redis_client.delete(memory._get_redis_key(session_id))


def test_sliding_window_ltrim():
    """Verify max_history limits history buffer size using Redis ltrim."""
    session_id = f"test-trim-{uuid.uuid4().hex[:8]}"
    memory = ChatMemoryManager(redis_url=settings.REDIS_URL, max_history=2)

    try:
        # Push 3 turns (6 total messages) into a buffer configured for max_history=2 (4 max messages)
        for i in range(1, 4):
            memory.add_turn(
                db=None,
                session_id=session_id,
                user_msg=f"User Q{i}",
                assistant_msg=f"Agent A{i}",
            )

        history = memory.get_recent_history(session_id=session_id)

        # Should only retain the last 2 turns (4 messages total)
        assert len(history) == 4
        assert history[0]["content"] == "User Q2"
        assert history[-1]["content"] == "Agent A3"
    finally:
        memory.redis_client.delete(memory._get_redis_key(session_id))


def test_get_history_empty_session():
    """Verify querying an uninitialized session ID returns an empty list cleanly."""
    session_id = f"nonexistent-{uuid.uuid4().hex[:8]}"
    memory = ChatMemoryManager(redis_url=settings.REDIS_URL)

    history = memory.get_recent_history(session_id=session_id)
    assert history == []