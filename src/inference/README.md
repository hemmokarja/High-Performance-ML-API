# Inference API

High-performance ML inference service with dynamic request batching for efficient model serving.

## Features

- **Dynamic Batching**: Automatically groups requests into optimal batches for GPU efficiency
- **Async Request Handling**: Non-blocking request processing with asyncio
- **Model Agnostic**: Supports any model that benefits from batch inference
- **Observability**: Prometheus metrics and Grafana dashboards for monitoring batch sizes, latencies, and throughput
- **Health Checks**: Production-ready health and readiness endpoints
- **Professional Error Handling**: Comprehensive error responses with proper status codes

## Architecture

The service uses a **worker pool pattern** with dynamic batching:

```
Request → Queue → Batch Collector Workers → Thread Pool → Model Inference
                        ↓
                  Prometheus Metrics
```

**Key Components:**
- **Request Queue**: Async queue collecting incoming prediction requests
- **Batch Collectors**: Background workers that group requests
- **Thread Pool**: Executes blocking model inference without blocking the event loop
- **Metrics System**: Tracks queue sizes, batch statistics, and latency

## How Dynamic Batching Works

The batching system optimizes GPU utilization by processing multiple requests together:

1. **Request arrives** → Added to async queue
2. **Worker collects requests** until either:
   - Batch reaches `max_batch_size` (e.g., 32), OR
   - `batch_timeout` expires (e.g., 10ms)
3. **Batch processed** in thread pool to avoid blocking new requests
4. **Results returned** to original callers via futures

This approach significantly improves throughput compared to processing requests individually.

## Quick Start

### Start the Service

In terminal

```bash
make start-inference
```

or with `docker-compose` (starts also the `gateway-api`)

```bash
make build
make up
```

The service will start on `http://localhost:8001` by default.

### Generate Embeddings

```bash
curl -X POST http://localhost:8001/embed \
  -H "Content-Type: application/json" \
  -d '{"input_text": "Hello, world!"}'
```

**Response:**
```json
{
  "embedding": [0.123, -0.456, ...],
  "model": "sentence-transformers/all-mpnet-base-v2"
}
```

### Check Service Health

```bash
curl http://localhost:8001/health
```

**Response:**
```json
{
  "status": "healthy",
  "model": "sentence-transformers/all-mpnet-base-v2",
  "device": "cuda:0",
  "queue_size": 0,
  "inflight_batches": 0
}
```

## API Reference

### Endpoints

#### `POST /embed`

Generate embeddings for input text. Requests are automatically batched for optimal performance.

**Request:**
```json
{
  "input_text": "Your text here"
}
```

**Validation:**
- Text must be 1-1024 characters
- Text cannot be empty or whitespace-only

**Response:**
```json
{
  "embedding": [0.123, -0.456, ...],
  "model": "sentence-transformers/all-mpnet-base-v2"
}
```

**Errors:**
- `422 Unprocessable Entity`: Invalid input (empty text, too long, etc.)
- `500 Internal Server Error`: Model inference failed
- `503 Service Unavailable`: Service not ready
- `504 Gateway Timeout`: Request timed out

#### `GET /health`

Health check endpoint with service metrics.

**Response:**
```json
{
  "status": "healthy",
  "model": "sentence-transformers/all-mpnet-base-v2",
  "device": "cuda:0",
  "queue_size": 0,
  "inflight_batches": 0
}
```

Returns `503 Service Unavailable` if model or batcher are not ready.

#### `GET /ready`

Kubernetes-style readiness probe.

**Response:**
```json
{
  "status": "ready"
}
```

Returns `503` if service cannot accept requests.

#### `GET /metrics`

Prometheus metrics endpoint for monitoring.

**Metrics Exposed:**
- `batcher_requests_total`: Total prediction requests (labeled by status)
- `batcher_request_latency_seconds`: End-to-end request latency
- `batcher_batch_size`: Distribution of batch sizes
- `batcher_batch_wait_time_seconds`: Time spent forming batches
- `batcher_inference_time_seconds`: Model inference time per batch
- `batcher_queue_size`: Current requests in queue (gauge)
- `batcher_inflight_batches`: Batches currently processing (gauge)

## Configuration

### Command Line Arguments

```bash
python -m inference.app \
  --host 0.0.0.0 \
  --port 8001 \
  --max-batch-size 32 \
  --batch-timeout 0.01 \
  --num-batching-workers 2
```

### Environment Variables

```env
# Hugging Face authentication (if using private models)
HF_TOKEN=your_hugging_face_token_here

# Model configuration
MODEL_NAME=sentence-transformers/all-mpnet-base-v2
```

### Tuning Batch Parameters

**`max_batch_size`**: Higher values improve GPU utilization but increase latency
- **Small models/fast inference**: 8-16
- **Medium models**: 32-64
- **Large models**: 128+

**`batch_timeout`**: Balance between throughput and latency
- **Low latency priority**: 5-10ms
- **Balanced**: 10-50ms
- **High throughput priority**: 50-100ms

**`num_batching_workers`**: Number of concurrent batch collectors
- **Single GPU**: 1-2 workers
- **Multiple GPUs**: Scale workers with GPU count
- More workers allow batches to form while previous batches are processing

## Error Handling

All errors return structured JSON responses:

```json
{
  "error": "Validation error",
  "detail": "Input text cannot be empty"
}
```

**Common Error Scenarios:**
- **Empty input**: Returns `422` with validation error
- **Text too long**: Returns `422` with max length message
- **Model failure**: Returns `500` with generic error message (details in logs)
- **Service not ready**: Returns `503` during startup/shutdown

## Monitoring with Prometheus + Grafana

Prometheus and Grafana are readily configured when running the application with `make up`.

See `MONITORING.md` (in project root) for a more detailed breakdown of observability.

## API Documentation

Interactive API documentation available at:

- **Swagger UI**: `http://localhost:8001/docs`
- **ReDoc**: `http://localhost:8001/redoc`

## Troubleshooting

**Service returns 503 on startup:**
- Model is still loading - wait a few seconds and retry
- Check logs for model loading errors

**High latency despite batching:**
- Reduce `batch_timeout` for lower latency
- Check if `max_batch_size` is too large
- Monitor GPU utilization - may need smaller batches

**Queue size growing:**
- Increase `max_batch_size` for higher throughput
- Add more batching workers
- Consider scaling horizontally with multiple instances
