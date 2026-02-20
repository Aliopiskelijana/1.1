"""
FastAPI application factory with rate limiting and startup model loading.
"""

import logging
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.api.model_store import ModelStore
from src.api.routes import router

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Rate limiter: 60 requests / minute per IP
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])


def create_app() -> FastAPI:
    app = FastAPI(
        title="Predictive Maintenance API",
        description=(
            "Machine failure prediction using the AI4I 2020 dataset.\n\n"
            "Includes SHAP-based explainability for EU AI Act compliance."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # CORS — restrict in production via env var
    origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Routes
    app.include_router(router, prefix="/api/v1")

    # Preload model at startup
    @app.on_event("startup")
    async def startup():
        logger.info("Loading model artifacts...")
        ModelStore.get()
        logger.info("API ready.")

    return app


app = create_app()
