import time
from typing import AsyncGenerator, Dict, Any
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from redis.asyncio import Redis
from qdrant_client import AsyncQdrantClient
from core.config import settings

# SQLAlchemy Async Engine
engine = create_async_engine(
    settings.ASYNC_DATABASE_URI,
    echo=False,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)

# Shared Redis Client with explicit socket timeouts to prevent hanging
redis_client: Redis = Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=0,
    decode_responses=True,
    socket_timeout=3.0,
    socket_connect_timeout=3.0,
)

# Shared Async Qdrant Client
qdrant_client: AsyncQdrantClient = AsyncQdrantClient(
    host=settings.QDRANT_HOST,
    port=settings.QDRANT_PORT,
    check_compatibility=False  # Disable version check warning
)

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

async def check_services_health() -> Dict[str, Any]:
    """Pings all external services and measures latency in milliseconds."""
    health_status: Dict[str, Any] = {"status": "healthy", "dependencies": {}}

    # Check PostgreSQL
    pg_start = time.perf_counter()
    try:
        async with engine.connect() as conn:
            from sqlalchemy import text
            await conn.execute(text("SELECT 1"))
        pg_latency = round((time.perf_counter() - pg_start) * 1000, 2)
        health_status["dependencies"]["postgres"] = {"status": "up", "latency_ms": pg_latency}
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["dependencies"]["postgres"] = {"status": "down", "error": str(e)}

    # Check Redis
    redis_start = time.perf_counter()
    try:
        ping_res = await redis_client.ping()
        redis_latency = round((time.perf_counter() - redis_start) * 1000, 2)
        if ping_res:
            health_status["dependencies"]["redis"] = {"status": "up", "latency_ms": redis_latency}
        else:
            raise Exception("Redis PING failed")
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["dependencies"]["redis"] = {"status": "down", "error": str(e)}

    # Check Qdrant
    qdrant_start = time.perf_counter()
    try:
        await qdrant_client.get_collections()
        qdrant_latency = round((time.perf_counter() - qdrant_start) * 1000, 2)
        health_status["dependencies"]["qdrant"] = {"status": "up", "latency_ms": qdrant_latency}
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["dependencies"]["qdrant"] = {"status": "down", "error": str(e)}

    return health_status