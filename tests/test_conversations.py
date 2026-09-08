import pytest
from httpx import AsyncClient
from apps.api.main import app

@pytest.mark.asyncio
async def test_multi_turn_conversation_memory():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        session_id = "test-session-12345"
        
        # Turn 1
        res1 = await ac.post("/api/v1/conversations/chat", json={
            "session_id": session_id,
            "message": "Who is Roboute Guilliman?"
        })
        assert res1.status_code == 200
        assert "Guilliman" in res1.json()["response"]

        # Turn 2 (Dependent on Turn 1)
        res2 = await ac.post("/api/v1/conversations/chat", json={
            "session_id": session_id,
            "message": "Which Chapter does he lead?"
        })
        assert res2.status_code == 200
        assert "Ultramarines" in res2.json()["response"]