# tests/unit/test_concurrency_faults.py
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import asyncio
import uuid
import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport

from core.config import settings
from core.metrics import CHAT_REQUESTS_TOTAL
from services.agent.memory import ChatMemoryManager
from services.agent.tools import search_knowledge_base
from apps.api.main import app


@pytest.mark.anyio
async def test_concurrent_chat_memory_writes():
    """Verify concurrent writes to ChatMemoryManager across parallel sessions do not corrupt Redis buffers."""
    memory = ChatMemoryManager(redis_url=settings.REDIS_URL, max_history=5)
    session_prefix = f"test-concurrent-{uuid.uuid4().hex[:6]}"

    async def simulate_user_turn(index: int):
        session_id = f"{session_prefix}-{index % 3}"  # Shared across 3 sessions
        await asyncio.to_thread(
            memory.add_turn,
            db=None,
            session_id=session_id,
            user_msg=f"User prompt {index}",
            assistant_msg=f"Assistant response {index}",
        )

    # Fire 15 concurrent turns across 3 sessions simultaneously
    await asyncio.gather(*[simulate_user_turn(i) for i in range(15)])

    # Verify each session's sliding window was truncated to max_history (10 messages max)
    for i in range(3):
        session_id = f"{session_prefix}-{i}"
        history = memory.get_recent_history(session_id)
        assert len(history) <= 10
        memory.redis_client.delete(memory._get_redis_key(session_id))


def test_search_tool_resilience_on_retriever_exception():
    """Verify search_knowledge_base catches vector DB exceptions cleanly without unhandled crashes."""
    with patch("services.agent.tools.get_retriever") as mock_get_retriever:
        mock_retriever = mock_get_retriever.return_value
        mock_retriever.search.side_effect = ConnectionError("Qdrant service unreachable")

        # Invoke tool directly
        result = search_knowledge_base.invoke({"query": "Space Marines"})

        assert "Knowledge base search failed" in result
        assert "ConnectionError" in result


@pytest.mark.anyio
async def test_chat_endpoint_agent_failure_fallback():
    """Verify endpoint catches agent execution failures, records error metrics, and returns 500."""
    initial_error_count = (
        CHAT_REQUESTS_TOTAL.labels(status="error")._value.get()
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        payload = {
            "session_id": "test-fault-session",
            "message": "Trigger failure",
        }

        with patch(
            "apps.api.v1.routes.conversations.agent.run",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Ollama local connection refused"),
        ):
            response = await ac.post("/api/v1/conversations/chat", json=payload)

            assert response.status_code == 500
            assert "Ollama local connection refused" in response.json()["detail"]

            # Verify Prometheus error counter incremented
            new_error_count = CHAT_REQUESTS_TOTAL.labels(
                status="error"
            )._value.get()
            assert new_error_count == initial_error_count + 1.0