# API Gateway

Professional API Gateway with authentication and rate limiting for the ML inference service.

## Features

- ✅ **API Key Authentication**: Secure bearer token authentication with SHA-256 hashed keys
- ✅ **Rate Limiting**: Sliding window rate limiter with per-minute and per-hour limits
- ✅ **Professional Error Handling**: Comprehensive error responses with proper HTTP status codes
- ✅ **Usage Tracking**: Monitor your API usage in real-time
- ✅ **Health Checks**: Kubernetes-ready health and readiness endpoints
- ✅ **CORS Support**: Configurable cross-origin resource sharing

## Architecture

```
Client → API Gateway (port 8000) → Inference Service (port 8001)
         ↓
    Auth + Rate Limiting
```

## Quick Start

### 1. Start the Inference Service

```bash
python -m inference.app --port 8001
```

### 2. Start the Gateway

```bash
# Basic start (generates a dev API key)
python -m gateway.app

# With custom configuration
python -m gateway.app \
  --host 0.0.0.0 \
  --port 8000 \
  --inference-url http://localhost:8001 \
  --rate-limit-minute 60 \
  --rate-limit-hour 1000
```

### 3. Use the API

```bash
# The gateway will print your API key on startup
export API_KEY="sk_dev_XXXXXXXXX"

# Generate embeddings
curl -X POST http://localhost:8000/v1/embed \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input_text": "Hello, world!"}'

# Check usage
curl http://localhost:8000/v1/usage \
  -H "Authorization: Bearer $API_KEY"

# Health check
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
  "usage": {
    "prompt_tokens": 4,
    "total_tokens": 4
  }
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

### Environment Variables

```bash
# Persist your API key across restarts
export TEST_API_KEY="sk_live_XXXXXXXXX"

# Configure inference service URL
export INFERENCE_SERVICE_URL="http://inference:8001"
```

### Command Line Arguments

```
--host                  Host to bind to (default: 0.0.0.0)
--port                  Port to bind to (default: 8000)
--inference-url         Inference service URL (default: http://localhost:8001)
--rate-limit-minute     Requests per minute limit (default: 60)
--rate-limit-hour       Requests per hour limit (default: 1000)
--cors-origins          CORS allowed origins (default: *)
```

## Key Management

### Generate API Keys

```bash
# Generate a new key
python -m gateway.utils.key_management generate

# Generate with custom prefix
python -m gateway.utils.key_management generate --prefix sk_test
```

### Key Format

- Production keys: `sk_live_XXXXXXXXX`
- Test keys: `sk_test_XXXXXXXXX`
- Development keys: `sk_dev_XXXXXXXXX`

Keys are stored as SHA-256 hashes for security.

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

## Testing

```bash
# Run with test configuration
python -m gateway.app \
  --rate-limit-minute 10 \
  --rate-limit-hour 100

# Test rate limiting
for i in {1..15}; do
  curl -X POST http://localhost:8000/v1/embed \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"input_text": "test"}'
  echo ""
done
```

## License

This is a personal project for learning and experimentation.