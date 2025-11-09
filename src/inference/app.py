import argparse
import os

import torch
import structlog
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from inference.api import exception_handlers, routes, lifespan as lifespan_module
from inference.models import huggingface
from shared import logging_config
from shared.middleware import CorrelationIdMiddleware

load_dotenv()
logger = structlog.get_logger(__name__)
logging_config.configure_structlog()

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

DEFAULT_MAX_BATCH_SIZE = 32
DEFAULT_BATCH_TIMEOUT = 0.01  # 10ms
DEFAULT_NUM_WORKERS = 2
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8001


def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("yes", "true", "t", "1", "y")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Text Embedding Model API Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help="Host to bind the server to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Port to bind the server to",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=DEFAULT_MAX_BATCH_SIZE,
        help="Maximum batch size for dynamic batching",
    )
    parser.add_argument(
        "--batch-timeout",
        type=float,
        default=DEFAULT_BATCH_TIMEOUT,
        help=(
            "Maximum wait time (in seconds) for dynamic batching before processing a "
            "partial batch"
        )
    )
    parser.add_argument(
        "--num-batching-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of worker threads for batch processing",
    )
    parser.add_argument(
        "--use-onnx",
        type=_str_to_bool,
        default=False,
        help="Use ONNX instead of PyTorch model"
    )
    return parser.parse_args()


def _create_app(
    max_batch_size: int, batch_timeout: float, num_workers: int, use_onnx: bool = False
) -> FastAPI:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lifespan = lifespan_module.create_lifespan(
        model_factory=lambda: huggingface.model_factory(
            MODEL_NAME, HF_TOKEN, device, use_onnx
        ),
        max_batch_size=max_batch_size,
        batch_timeout=batch_timeout,
        num_workers=num_workers,
    )
    app = FastAPI(
        title="Text Embedding Model API",
        description="Internal API for text embedding inference",
        lifespan=lifespan,
        docs_url="/docs",  # Swagger UI
        redoc_url="/redoc",  # ReDoc
    )

    app.add_middleware(CorrelationIdMiddleware, prefix="inf")

    exception_handlers.register_exception_handlers(app)
    routes.register_routes(app)

    logger.info(
        "FastAPI application created",
        max_batch_size=max_batch_size,
        batch_timeout=batch_timeout,
        num_workers=num_workers,
    )
    return app


def main():
    args = _parse_args()
    app = _create_app(
        max_batch_size=args.max_batch_size,
        batch_timeout=args.batch_timeout,
        num_workers=args.num_batching_workers,
        use_onnx=args.use_onnx,
    )

    Instrumentator().instrument(app).expose(app)

    logger.info("Starting server", host=args.host, port=args.port)
    uvicorn.run(
        app,  # recommended to pass as string in prod
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=True,
        workers=1,  # only 1 worker for model serving
    )


if __name__ == "__main__":
    main()
