# High-Performance Inference API

A production-ready machine learning inference service demonstrating advanced ML Engineering and MLOps practices, with an emphasis on latency optimization and throughput enhancement.

## 🎯 Project Overview

This project serves as a comprehensive demonstration of building scalable, high-performance ML inference systems. While I use a pre-trained BERT embedding model from Hugging Face as the example workload, the architecture is intentionally model-agnostic - you can adapt these components to serve virtually any model that benefits from batching with minimal code changes.

Think of this as a real-time embedding microservice capable of vectorizing user queries for downstream tasks like semantic search, but the patterns and optimizations apply broadly across ML serving scenarios.

### Key Learning Goals

- Implementing performance-critical optimizations for production ML systems
- Building scalable API architectures with proper separation of concerns
- Applying MLOps best practices including observability and load testing

## 🏗️ Architecture

The system implements a two-tier architecture that separates public-facing concerns from inference workloads:

### Gateway API (Public-Facing)

A lightweight FastAPI service handling:
- **API Key Authentication**: Secure access control for client applications
- **Distributed Rate Limiting**: Redis-backed sliding window implementation supporting horizontal scaling across multiple workers/nodes
- **Request Validation**: Pydantic schemas ensure data integrity before forwarding to inference
- **Health Checks**: `/health` and `/readiness` endpoints for orchestration platforms

### Inference API (Internal)

The core ML serving engine featuring:
- **Asynchronous Dynamic Batching**: Intelligently groups concurrent requests based on configurable batch size and wait time thresholds, processing model inference in dedicated threads while workers handle request queuing
- **Model Serving**: Flexible model loading supporting both standard PyTorch and optimized ONNX runtime
- **Prometheus Metrics**: Comprehensive instrumentation for latency, batch sizes, and throughput
- **Grafana Dashboards**: Pre-configured visualizations for real-time performance monitoring

### Shared Infrastructure

Both services benefit from:
- **Correlation ID Tracking**: Request tracing across service boundaries for distributed debugging and log aggregation
- **Standardized Error Handling**: Consistent exception patterns across the API surface
- **Structured Logging**: JSON-formatted logs with automatic correlation ID injection
- **Docker Compose Orchestration**: Simple multi-container deployment with Redis and monitoring stack

## ⚡ Performance Optimizations

### Performance Benchmarks

The following optimizations progressively improve inference performance:

| Configuration | Latency (median) | Latency (p95) | Throughput (RPS) |
|--------------|------------------|---------------|------------------|
| CPU baseline | 1600 | 1700 | 30 |
| GPU | 490 | 520 | 101 |
| GPU + Dynamic Batching | 110 | 130 | 438 |
| GPU + Batching + ONNX | 26 | 52 | 497 |
| GPU + Batching + ONNX + FP16 | 26 `*` | 42 | 497 |

*All tests measured against the Gateway API at constant throughput of 500 RPS*

`*` See "FP16 Conversion" section below for explanation why FP16 didn't improve latency or throughput here.

### Asynchronous Dynamic Batching

The batching system uses a worker pool architecture that maintains high throughput while minimizing latency:

**How it works:**
- Multiple async workers continuously collect incoming requests into batches
- Each batch is processed when it reaches either the **max batch size** or **max wait time** threshold
- Model inference runs in a dedicated thread pool, allowing workers to keep collecting new batches in the background while the GPU processes the current batch
- Results are returned to callers via async futures, maintaining request-response mapping

**Configuration parameters:**
- **Max Batch Size**: Upper limit on requests grouped together (larger batches = better GPU utilization)
- **Max Wait Time**: Maximum time to wait for additional requests (shorter wait = lower latency)

This dual-threshold approach with concurrent collection and inference allows the system to maintain sub-second latencies even under high load, while achieving latency and throughput improvements of nearly 5x compared to serial processing.

### ONNX Runtime with FP16 Precision

Post-training optimizations include:
- **ONNX Export**: Cross-platform inference runtime with graph optimizations, like kernel fusion, constant folding, and memory layout improvements
- **FP16 Conversion**: Half-precision floating point can reduce memory usage and increase throughput on modern GPUs, but in this toy project GPU utilization remains very low, so we don’t observe any speedup

These post-training optimizations further reduce latency by almost 80% and improve throughput by 15%.

## 🚀 Getting Started

### Prerequisites

- Python 3.11+ (managed via `uv`)
- Docker and Docker Compose (for containerized deployment)
- CUDA-capable GPU (optional, but required for GPU benchmarks)
- Hugging Face account token (for model downloads)

### Installation

1. **Install Python dependencies**:
   ```bash
   uv sync --extra cu128  # or --extra cpu for running on CPU
   ```

2. **Configure environment variables**:
   
   Create a `.env` file in the project root:
   ```bash
   HF_TOKEN=your_huggingface_token_here
   API_KEY=your_secret_api_key_here
   ```
   
   - `HF_TOKEN`: Required for downloading models from Hugging Face Hub
   - `API_KEY`: Used for gateway authentication. If omitted, a random key is generated and printed in the startup logs

3. **Export model to ONNX** (optional, for ONNX runtime):
   ```bash
   make onnx-export
   ```

   If you don't want to use ONNX runtime, set `USE_ONNX=false` in `Makefile` before running the application.

### Running the Services

#### Docker Deployment (Recommended)

```bash
# build images
make build

# start all services (Gateway, Inference, Redis, Prometheus, Grafana)
make up

# stop all services
make down
```

#### Local Development

Run each service in a separate terminal:

```bash
# terminal 1: start inference API
make start-inference

# terminal 2: start gateway API
make start-gateway
```

### Load Testing

Benchmark the services using Locust:

```bash
# test inference API directly
make load-test-inference

# test gateway API (disable rate limits for pure throughput testing)
make up BYPASS_RATE_LIMITS=true
make load-test-gateway
```

### Monitoring Stack

- **Prometheus**: Metrics collection with custom batching and rate limit metrics
- **Grafana**: Pre-configured dashboards for latency, throughput, batch sizes, and more
- **Structured Logs**: JSON format ready for ELK/Loki aggregation

## 📖 Documentation

- **Gateway API Details**: See `src/gateway/README.md` for endpoint specifications, authentication flow, and rate limiting configuration
- **Inference API Details**: See `src/inference/README.md` for model loading, batching parameters, and metrics exposed
- **Monitoring Setup**: See `MONITORING.md` for Prometheus configuration and Grafana dashboard usage

## 📁 Project Structure

```
.
├── src/
│   ├── gateway/          # Public API: auth, rate limiting, routing
│   ├── inference/        # ML service: batching, model serving, metrics
│   ├── shared/           # Common utilities: correlation IDs, middleware, logging
│   ├── benchmarks/       # Locust load testing scenarios
│   └── onnx_util/        # Model export utilities
├── monitoring/
│   ├── prometheus.yml    # Metrics collection config
│   └── grafana/          # Dashboard definitions and provisioning
├── scripts/              # Helper scripts for setup and testing
├── docker-compose.yaml   # Multi-service orchestration
└── Makefile              # Common development commands
```

## 🔮 Roadmap

Planned enhancements:

- **Circuit Breaker**: Fault isolation to prevent cascade failures when inference service degrades
- **gRPC instead of HTTP/JSON**: Replace REST endpoints with gRPC for lower latency, type-safe interfaces, and efficient streaming over HTTP/2, enabling better performance under load
- **Response Caching**: Redis-backed cache for duplicate requests

## 📄 License

This project is licensed under the MIT License.
