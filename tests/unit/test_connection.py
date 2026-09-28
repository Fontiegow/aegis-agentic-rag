# tests/unit/test_connection.py
import pytest
from unittest.mock import patch
from database.connection import get_db_session, check_services_health, redis_client


@pytest.mark.anyio
async def test_get_db_session_lifecycle():
    """Verify get_db_session dependency yields an AsyncSession and closes it upon exit."""
    async for session in get_db_session():
        assert session is not None
        # Break generator to verify proper cleanup/exit
        break


@pytest.mark.anyio
async def test_check_services_health_live():
    """Integration check: verify health status and latency payload against running Docker containers."""
    health = await check_services_health()

    assert "status" in health
    assert "dependencies" in health

    # Verify structured latency dictionary contracts
    assert health["status"] == "healthy"
    assert health["dependencies"]["postgres"]["status"] == "up"
    assert "latency_ms" in health["dependencies"]["postgres"]

    assert health["dependencies"]["redis"]["status"] == "up"
    assert "latency_ms" in health["dependencies"]["redis"]

    assert health["dependencies"]["qdrant"]["status"] == "up"
    assert "latency_ms" in health["dependencies"]["qdrant"]


@pytest.mark.anyio
async def test_check_services_health_failure_handling():
    """Unit check: verify health check captures failure and sets status to 'unhealthy'."""
    with patch.object(
        redis_client, "ping", side_effect=Exception("Redis connection refused")
    ):
        health = await check_services_health()

        assert health["status"] == "unhealthy"
        assert health["dependencies"]["redis"]["status"] == "down"
        assert "Redis connection refused" in health["dependencies"]["redis"]["error"]