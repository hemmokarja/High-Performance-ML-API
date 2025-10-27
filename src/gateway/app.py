import argparse
import os

import structlog
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gateway.api import exception_handlers, routes, lifespan as lifespan_module
from gateway.auth.api_key_db import ApiKeyDB
from gateway.auth.rate_limiter import SlidingWindowRateLimiter
from gateway.auth.middleware import AuthMiddleware

load_dotenv()
logger = structlog.get_logger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_INFERENCE_URL = "http://localhost:8001"
DEFAULT_RATE_LIMIT_MINUTE = 60
DEFAULT_RATE_LIMIT_HOUR = 1000


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
        "--cors-origins",
        type=str,
        default="*",
        help="CORS allowed origins (comma-separated, or * for all)",
    )
    return parser.parse_args()


def _initialize_dev_api_key(
    api_key_db: ApiKeyDB,
    default_minute_limit: int,
    default_hour_limit: int
) -> None:
    """
    Initialize the API key database with default key.

    In production, load it from environment variable or a secure vault.
    """
    dev_key = ApiKeyDB.generate_key(prefix="sk_dev")
    api_key_db.add_key(
        key=dev_key,
        user_id="dev_user",
        name="Development API Key",
        rate_limit_per_minute=default_minute_limit,
        rate_limit_per_hour=default_hour_limit,
    )
    logger.warning(
        "Generated development API key - store this securely!",
        api_key=dev_key
    )
    print(f"\n{'='*60}")
    print(f"Development API Key: {dev_key}")
    print(f"{'='*60}\n")


def _create_app(
    inference_url: str,
    rate_limit_minute: int,
    rate_limit_hour: int,
    cors_origins: str
) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        inference_url: URL of the inference service
        rate_limit_minute: Default rate limit per minute
        rate_limit_hour: Default rate limit per hour
        cors_origins: CORS allowed origins
        
    Returns:
        Configured FastAPI application
    """
    api_key_db = ApiKeyDB()
    rate_limiter = SlidingWindowRateLimiter()
    auth_middleware = AuthMiddleware(api_key_db, rate_limiter)

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

    origins = cors_origins.split(",") if cors_origins != "*" else ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    exception_handlers.register_exception_handlers(app)
    routes.register_routes(app, auth_middleware.verify_api_key)

    logger.info(
        "FastAPI application created",
        inference_url=inference_url,
        rate_limit_minute=rate_limit_minute,
        rate_limit_hour=rate_limit_hour,
        cors_origins=origins
    )

    return app


def main():
    args = parse_args()

    app = _create_app(
        inference_url=args.inference_url,
        rate_limit_minute=args.rate_limit_minute,
        rate_limit_hour=args.rate_limit_hour,
        cors_origins=args.cors_origins
    )

    logger.info(
        "Starting API Gateway",
        host=args.host,
        port=args.port
    )
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=True,
        workers=1,  # TODO change this later
    )


if __name__ == "__main__":
    main()
