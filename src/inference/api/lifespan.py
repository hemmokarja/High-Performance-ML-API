from contextlib import asynccontextmanager
from typing import Callable

import structlog
from fastapi import FastAPI
from pydantic import BaseModel

from inference.api.batcher import DynamicBatcher, NoBatchingWrapper

logger = structlog.get_logger(__name__)


def create_lifespan(
    model_factory: Callable[[], BaseModel],
    max_batch_size: int,
    batch_timeout: float,
    num_workers: int,
    batcher_cls: DynamicBatcher | NoBatchingWrapper = DynamicBatcher,
):
    """
    Create a lifespan context for the FastAPI app.

    Usage:

        lifespan_fn = create_lifespan(
            model_factory=lambda: MyModel(),
            max_batch_size=32,
            batch_timeout=0.01,
            num_workers=4,
        )
        app = FastAPI(lifespan=lifespan_fn)

    """
    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        """Manages startup and shutdown."""
        try:
            model = model_factory()
            batcher = batcher_cls(
                model=model,
                max_batch_size=max_batch_size,
                batch_timeout=batch_timeout,
                num_workers=num_workers,
            )
            await batcher.start()

            app.state.model = model
            app.state.batcher = batcher

            logger.info("Model API started successfully")

        except Exception as e:
            logger.error("Failed to start model API", error=str(e))
            raise

        # hand over control to the running app
        yield

        logger.info("Shutting down model API")
        if hasattr(app.state, "batcher") and app.state.batcher:
            await app.state.batcher.shutdown()

        logger.info("Model API shutdown complete")
    
    return _lifespan
