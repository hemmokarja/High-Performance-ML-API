.PHONY: start-inference start-gateway load-test build up down

INFERENCE_PORT?=8001
GATEWAY_PORT?=8000
MAX_BATCH_SIZE?=32
BATCH_TIMEOUT?=0.01
NUM_BATCHING_WORKERS?=2
NUM_UVICORN_WORKERS?=4
BYPASS_RATE_LIMITS?=false

start-inference:
	@uv run python src/inference/app.py \
		--host 0.0.0.0 \
		--port $(INFERENCE_PORT) \
		--max-batch-size $(MAX_BATCH_SIZE) \
		--batch-timeout $(BATCH_TIMEOUT) \
		--num-batching-workers $(NUM_BATCHING_WORKERS)

start-gateway:
	@BYPASS_RATE_LIMITS=$(BYPASS_RATE_LIMITS) uv run python src/gateway/app.py \
		--host 0.0.0.0 \
		--port $(GATEWAY_PORT) \
		--inference-url "http://localhost:$(INFERENCE_PORT)" \
		--rate-limit-minute 60 \
		--rate-limit-hour 1000 \
		--workers $(NUM_UVICORN_WORKERS)

load-test-inference:
	@./scripts/load-test.sh \
		-H localhost \
		-P $(INFERENCE_PORT) \
		-u 50 \
		-r 10 \
		-d "30s" \
		-f src/benchmarks/locustfile_inference.py

load-test-gateway:
	@./scripts/load-test.sh \
		-H localhost \
		-P $(GATEWAY_PORT) \
		-u 50 \
		-r 10 \
		-d "30s" \
		-f src/benchmarks/locustfile_gateway.py

build:
	@docker compose build

up:
	@INFERENCE_PORT=$(INFERENCE_PORT) \
		MAX_BATCH_SIZE=$(MAX_BATCH_SIZE) \
		BATCH_TIMEOUT=$(BATCH_TIMEOUT) \
		NUM_BATCHING_WORKERS=$(NUM_BATCHING_WORKERS) \
		GATEWAY_PORT=$(GATEWAY_PORT) \
		INFERENCE_URL=http://inference-api:$(INFERENCE_PORT) \
		NUM_UVICORN_WORKERS=$(NUM_UVICORN_WORKERS) \
		BYPASS_RATE_LIMITS=$(BYPASS_RATE_LIMITS) \
		docker compose up -d

down:
	@docker compose down

# get ip for port-forwarding
get-ip:
	@echo "$(curl -s http://169.254.169.254/latest/meta-data/local-ipv4)"