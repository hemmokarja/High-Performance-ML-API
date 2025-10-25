import os
from contextlib import asynccontextmanager
from typing import List, Optional

import structlog
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from inference.models.hugginface import HugginFaceEmbeddingModel

from inference.api import exception_handlers, routes, lifespan as lifespan_module

load_dotenv()

logger = structlog.get_logger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

MAX_BATCH_SIZE = 32
BATCH_INTERVAL = 0.01  # 10ms
NUM_WORKERS = 2

HOST = "0.0.0.0"
PORT = 8001


def create_app() -> FastAPI:
    lifespan = lifespan_module.create_lifespan(
        model_factory=lambda: HugginFaceEmbeddingModel(MODEL_NAME, HF_TOKEN),
        max_batch_size=MAX_BATCH_SIZE,
        batch_interval=BATCH_INTERVAL,
        num_workers=NUM_WORKERS,
    )
    app = FastAPI(
        title="Sentence Embedding Model API",
        description="Internal API for sentence embedding inference",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",  # Swagger UI
        redoc_url="/redoc",  # ReDoc
    )
    exception_handlers.register_exception_handlers(app)
    routes.register_routes(app)
    logger.info("FastAPI application created")
    return app


app = create_app()


def main():
    uvicorn.run(
        "inference.api.app:app",
        host=HOST,
        port=PORT,
        log_level="info",
        access_log=True,
        workers=1,  # only 1 worker for model serving
        limit_concurrency=1000,
        timeout_keep_alive=30,
    )


if __name__ == "__main__":
    main()
