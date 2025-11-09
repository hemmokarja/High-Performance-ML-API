import argparse
import os

import structlog
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

from gateway.api import exception_handlers, routes, lifespan as lifespan_module
from gateway.auth import rate_limiter as rate_limiter_module
from gateway.auth.api_key_db import ApiKeyDB
from gateway.auth.auth import AuthService
from shared import logging_config
from shared.middleware import CorrelationIdASGIMiddleware

load_dotenv()
logger = structlog.get_logger(__name__)
logging_config.configure_structlog()

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_INFERENCE_URL = "http://localhost:8001"
DEFAULT_RATE_LIMIT_MINUTE = 60
DEFAULT_RATE_LIMIT_HOUR = 1000
DEFAULT_REDIS_URL = "redis://localhost:6379/0"


def parse_args():
    parser = argparse.ArgumentParser(
        description="ML Inference API Gateway",
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
        "--inference-url",
        type=str,
        default=DEFAULT_INFERENCE_URL,
        help="URL of the inference service",
    )
    parser.add_argument(
        "--rate-limit-minute",
        type=int,
        default=DEFAULT_RATE_LIMIT_MINUTE,
        help="Default requests per minute rate limit",
    )
    parser.add_argument(
        "--rate-limit-hour",
        type=int,
        default=DEFAULT_RATE_LIMIT_HOUR,
        help="Default requests per hour rate limit",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of Uvicorn workers to use",
    )
    parser.add_argument(
        "--redis-url",
        type=str,
        default=DEFAULT_REDIS_URL,
        help="Redis connection URL for distributed rate limiting",
    )
    parser.add_argument(
        "--bypass-rate-limits",
        action="store_true",
        help="Disable rate limiting entirely",
    )
    return parser.parse_args()


def _initialize_dev_api_key(
    api_key_db: ApiKeyDB,
    default_minute_limit: int,
    default_hour_limit: int
) -> None:
    """
    Initialize the API key database with default key.
    First checks for API_KEY in environment variables. If not found, generates a new
    key and prints it.

    NOTE: this is implemented for convenience, not somethign would do in production.
    """
    dev_key = os.environ.get("API_KEY")
    if dev_key:
        logger.info("Using API key from environment variable API_KEY")
    else:
        dev_key = ApiKeyDB.generate_key(prefix="sk_dev")
        logger.warning(
            "Generated development API key - store this securely!",
            api_key=dev_key
        )
        print(f"\n{'='*60}")
        print(f"Development API Key: {dev_key}")
        print(f"{'='*60}\n")

    api_key_db.add_key(
        key=dev_key,
        user_id="dev_user",
        name="Development API Key",
        rate_limit_per_minute=default_minute_limit,
        rate_limit_per_hour=default_hour_limit,
    )


def _create_app(
    inference_url: str,
    rate_limit_minute: int,
    rate_limit_hour: int,
    redis_url: str,
    bypass_rate_limits: bool,
) -> FastAPI:
    api_key_db = ApiKeyDB()
    rate_limiter = rate_limiter_module.create_rate_limiter(
        redis_url, bypass_rate_limits
    )
    auth_service = AuthService(api_key_db, rate_limiter)

    _initialize_dev_api_key(api_key_db, rate_limit_minute, rate_limit_hour)

    lifespan = lifespan_module.create_lifespan(
        inference_service_url=inference_url,
        api_key_db=api_key_db,
        rate_limiter=rate_limiter
    )

    app = FastAPI(
        title="ML Inference API Gateway",
        description="Public-facing API gateway for text embedding inference",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(CorrelationIdASGIMiddleware, prefix="gw")

    exception_handlers.register_exception_handlers(app)
    routes.register_routes(app, auth_service.verify_api_key)

    logger.info(
        "FastAPI application created",
        inference_url=inference_url,
        rate_limit_minute=rate_limit_minute,
        rate_limit_hour=rate_limit_hour,
        rate_limiter_type=type(rate_limiter).__name__,
    )

    return app


# need to init app globally to allow multiple workers, which requires passing app path 
# string to uvicorn.run()
args = parse_args()

bypass_rate_limits_env = os.environ.get("BYPASS_RATE_LIMITS", "false").lower() == "true"
bypass_rate_limits = args.bypass_rate_limits or bypass_rate_limits_env

app = _create_app(
    inference_url=args.inference_url,
    rate_limit_minute=args.rate_limit_minute,
    rate_limit_hour=args.rate_limit_hour,
    redis_url=args.redis_url,
    bypass_rate_limits=bypass_rate_limits_env,
)

if __name__ == "__main__":
    logger.info(
        "Starting API Gateway",
        host=args.host,
        port=args.port,
        workers=args.workers,
    )

    uvicorn.run(
        "gateway.app:app",
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=True,
        workers=args.workers,
    )
