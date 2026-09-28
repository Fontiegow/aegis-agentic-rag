import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

# Ensure root workspace directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.config import settings
from database.connection import check_services_health, engine, qdrant_client, redis_client
from apps.api.v1.router import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await redis_client.aclose()
    await qdrant_client.close()
    await engine.dispose()


# 1. Instantiate FastAPI once
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# 2. Attach Prometheus Instrumentator
instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
)
instrumentator.instrument(app).expose(app, endpoint="/metrics")

# 3. Mount Routers
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/health", tags=["Infrastructure Check"])
@app.get(f"{settings.API_V1_STR}/health", tags=["Infrastructure Check"])
async def health_check():
    health = await check_services_health()
    status_code = (
        status.HTTP_200_OK
        if health["status"] == "healthy"
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    return JSONResponse(status_code=status_code, content=health)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("apps.api.main:app", host="0.0.0.0", port=8000, reload=True)