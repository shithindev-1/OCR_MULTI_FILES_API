"""Application entry point.

    python -m app                        # production / Windows service (host and port from .env)
    uvicorn app.main:app --reload        # local development
"""

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.router import api_router
from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging
from app.core.middleware import OcrSecurityMiddleware
from app.services.blob_storage import get_blob_storage
from app.services.gemini import get_gemini_service
from app.services.retention import run_retention_sweeps

logger = logging.getLogger("uae_ocr")


async def _warm_up_connections() -> None:
    """Pays the one-off client creation and DNS/TLS setup at startup instead of on the first
    request. Best effort: a failure is logged and the first request simply connects itself."""
    started_at = time.perf_counter()
    targets = [("gemini", get_gemini_service().warm_up())]
    if storage := get_blob_storage():
        targets.append(("blob", storage.warm_up()))
    results = await asyncio.gather(*(job for _, job in targets), return_exceptions=True)
    for (name, _), result in zip(targets, results):
        if isinstance(result, Exception):
            logger.warning("Warm-up failed target=%s error=%s", name, type(result).__name__)
    logger.info(
        "Warm-up finished targets=%s duration_sec=%.2f",
        [name for name, _ in targets], time.perf_counter() - started_at,
    )


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # In the background, so the service accepts requests immediately rather than after the warm-up.
    warm_up = asyncio.create_task(_warm_up_connections())
    sweeper = asyncio.create_task(run_retention_sweeps(settings)) if settings.output_retention_hours > 0 else None
    try:
        yield
    finally:
        for task in (warm_up, sweeper):
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task


def create_app() -> FastAPI:
    configure_logging(settings)

    docs = settings.docs_enabled
    application = FastAPI(
        title="Document OCR API",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if docs else None,
        redoc_url="/redoc" if docs else None,
        openapi_url="/openapi.json" if docs else None,
    )
    application.add_middleware(OcrSecurityMiddleware)
    if settings.cors_origin_list:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origin_list,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )
    register_exception_handlers(application)
    application.include_router(api_router)
    return application


app = create_app()
