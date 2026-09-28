# tests/unit/test_config.py
from core.config import Settings


def test_default_settings():
    """Verify default setting values match core/config.py specifications."""
    test_settings = Settings(_env_file=None)

    assert test_settings.PROJECT_NAME == "Aegis"
    assert test_settings.API_V1_STR == "/api/v1"
    assert test_settings.ENVIRONMENT == "development"
    assert test_settings.POSTGRES_PORT == 5432
    assert test_settings.REDIS_PORT == 6379
    assert test_settings.QDRANT_PORT == 6333
    assert test_settings.OLLAMA_BASE_URL == "http://localhost:11434"


def test_computed_properties():
    """Verify property getters dynamically construct correct connection URIs."""
    test_settings = Settings(_env_file=None)

    assert (
        test_settings.ASYNC_DATABASE_URI
        == "postgresql+asyncpg://aegis_user:aegis_password@localhost:5432/aegis_db"
    )
    assert test_settings.REDIS_URL == "redis://localhost:6379/0"


def test_environment_override(monkeypatch):
    """Verify environment variables override default configuration values."""
    monkeypatch.setenv("PROJECT_NAME", "Aegis Test Cluster")
    monkeypatch.setenv("POSTGRES_PORT", "5433")
    monkeypatch.setenv("REDIS_HOST", "cache.internal")

    test_settings = Settings(_env_file=None)

    assert test_settings.PROJECT_NAME == "Aegis Test Cluster"
    assert test_settings.POSTGRES_PORT == 5433
    assert test_settings.REDIS_HOST == "cache.internal"
    assert (
        test_settings.ASYNC_DATABASE_URI
        == "postgresql+asyncpg://aegis_user:aegis_password@localhost:5433/aegis_db"
    )
    assert test_settings.REDIS_URL == "redis://cache.internal:6379/0"


def test_invalid_type_validation(monkeypatch):
    """Verify Pydantic enforces type validation on configuration variables."""
    monkeypatch.setenv("POSTGRES_PORT", "not_a_number")

    try:
        Settings(_env_file=None)
        assert False, "Expected ValidationError for invalid integer type"
    except Exception:
        pass