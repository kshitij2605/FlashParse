import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from ..config.settings import Settings
from ..pipeline.orchestrator import AsyncPDFPipeline
from .routes import pdf as pdf_routes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.load()
    logger.info("Initializing pipeline...")
    app.state.pipeline = AsyncPDFPipeline(settings)
    pdf_routes.pipeline = app.state.pipeline
    logger.info("Pipeline ready")
    yield
    logger.info("Shutting down pipeline...")
    await app.state.pipeline.close()


app = FastAPI(
    title="GLM Hybrid OCR API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(pdf_routes.router)


@app.get("/health")
async def health_check():
    pipeline = getattr(app.state, "pipeline", None)
    vlm_ok = False
    if pipeline:
        vlm_ok = await pipeline.vlm_client.health_check()
    return {
        "status": "healthy",
        "version": "0.1.0",
        "vlm_available": vlm_ok,
    }


@app.get("/api/v1/health")
async def api_health_check():
    return await health_check()


def main():
    settings = Settings.load()
    uvicorn.run(
        "glm_hybrid_ocr.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
