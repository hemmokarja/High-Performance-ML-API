# API Gateway

Professional API Gateway with authentication and rate limiting for the ML inference service.

## Features

- **API Key Authentication**: Secure bearer token authentication with SHA-256 hashed keys
- **Rate Limiting**: Sliding window rate limiter with per-minute and per-hour limits
- **Professional Error Handling**: Comprehensive error responses with proper HTTP status codes
- **Usage Tracking**: Monitor your API usage in real-time
- **Health Checks**: Kubernetes-ready health and readiness endpoints

## Architecture

```
Client → API Gateway (port 8000) → Inference Service (port 8001)
         ↓
    Auth + Rate Limiting
```

## Quick Start

### Start the Inference Service

In terminal

```bash
make start-inference
make start-gateway
```

or with `docker-compose`

```bash
make up
```

### Use the API

```bash
# The gateway will print your API key on startup
export API_KEY="sk_dev_XXXXXXXXX"

# generate embeddings
curl -X POST http://localhost:8000/v1/embed \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input_text": "Hello, world!"}'

# check usage
curl http://localhost:8000/v1/usage \
  -H "Authorization: Bearer $API_KEY"

# health check
curl http://localhost:8000/health
```

## API Reference

### Authentication

All protected endpoints require a bearer token:

```
Authorization: Bearer sk_live_XXXXXXXXX
```

### Endpoints

#### `POST /v1/embed`

Generate embeddings for input text.

**Request:**
```json
{
  "input_text": "Your text here"
}
```

**Response:**
```json
{
  "embedding": [0.123, -0.456, ...],
  "model": "sentence-transformers/all-mpnet-base-v2",
}
```

**Rate Limits:**
- Default: 60 requests/minute, 1000 requests/hour
- Returns `429 Too Many Requests` with `Retry-After` header when exceeded

#### `GET /v1/usage`

Get current rate limit usage.

**Response:**
```json
{
  "user_id": "dev_user",
  "usage": {
    "requests_last_minute": 5,
    "requests_last_hour": 42,
    "timestamp": "2025-01-27T10:30:00Z"
  },
  "limits": {
    "per_minute": 60,
    "per_hour": 1000
  }
}
```

#### `GET /health`

Health check endpoint (no auth required).

**Response:**
```json
{
  "status": "healthy",
  "gateway_version": "1.0.0",
  "inference_service": {
    "status": "healthy",
    "model": "sentence-transformers/all-mpnet-base-v2",
    "device": "cuda:0",
    "queue_size": 0,
    "inflight_batches": 0
  }
}
```

#### `GET /ready`

Readiness check for Kubernetes (no auth required).

## Configuration

Command Line Arguments:

```
--host                  Host to bind to (default: 0.0.0.0)
--port                  Port to bind to (default: 8000)
--inference-url         Inference service URL (default: http://localhost:8001)
--rate-limit-minute     Requests per minute limit (default: 60)
--rate-limit-hour       Requests per hour limit (default: 1000)
```

## Rate Limiting

The gateway uses a **sliding window** rate limiter with two independent limits:

- **Per-minute limit**: Short-term burst protection
- **Per-hour limit**: Long-term usage control

When a limit is exceeded:
- Returns `429 Too Many Requests`
- Includes `Retry-After` header with seconds until reset
- Provides `X-RateLimit-Limit` and `X-RateLimit-Reset` headers

## Error Handling

All errors return structured JSON responses:

```json
{
  "error": "Rate limit exceeded",
  "detail": "Rate limit exceeded: 60 requests per minute. Retry after 42 seconds.",
  "code": "RATE_LIMIT_EXCEEDED",
  "retry_after": 42,
  "limit": 60,
  "limit_type": "minute"
}
```

**Error Codes:**
- `VALIDATION_ERROR`: Invalid request data
- `VALUE_ERROR`: Business logic validation failed
- `RATE_LIMIT_EXCEEDED`: Rate limit hit
- `INTERNAL_ERROR`: Unexpected server error

## Production Considerations

### Database-Backed Storage

Replace in-memory storage with persistent databases:

```python
# Use PostgreSQL for API keys
from sqlalchemy import create_engine

# Use Redis for rate limiting
import redis
rate_limiter = RedisRateLimiter(redis_client)
```

### Security

- Store API keys in environment variables or secret management systems (AWS Secrets Manager, HashiCorp Vault)
- Use HTTPS in production
- Implement key rotation policies
- Add request signing for extra security
- Consider adding IP whitelisting

### Monitoring

Integrate with monitoring tools:

```python
from prometheus_client import Counter, Histogram

requests_total = Counter('api_requests_total', 'Total requests')
request_duration = Histogram('request_duration_seconds', 'Request duration')
```

### Scaling

- Run multiple gateway instances behind a load balancer
- Use Redis for distributed rate limiting
- Implement circuit breakers for inference service calls
- Add request queuing for burst handling

## API Documentation

Interactive API documentation is available at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
