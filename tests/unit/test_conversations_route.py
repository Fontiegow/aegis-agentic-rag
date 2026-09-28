# tests/unit/test_conversations_route.py
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport
from apps.api.main import app


@pytest.mark.anyio
async def test_chat_endpoint_success():
    """Verify POST /api/v1/conversations/chat returns 200 OK and expected JSON payload structure."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        payload = {
            "session_id": "test-route-123",
            "message": "Who is Roboute Guilliman?",
        }

        mock_agent_result = {
            "output": "Roboute Guilliman is the Primarch of the Ultramarines.",
            "retrieved_contexts": [
                "Roboute Guilliman is the Primarch of the Ultramarines."
            ],
        }

        with patch(
            "apps.api.v1.routes.conversations.agent.run",
            new_callable=AsyncMock,
            return_value=mock_agent_result,
        ):
            response = await ac.post("/api/v1/conversations/chat", json=payload)

            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == "test-route-123"
            assert "Ultramarines" in data["response"]
            assert "evaluation" in data
            assert "context_precision" in data["evaluation"]
            assert "faithfulness" in data["evaluation"]


@pytest.mark.anyio
async def test_chat_endpoint_validation_error():
    """Verify missing required key 'message' returns 422 Unprocessable Entity."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        payload = {"session_id": "test-invalid-123"}  # Missing 'message'

        response = await ac.post("/api/v1/conversations/chat", json=payload)
        assert response.status_code == 422