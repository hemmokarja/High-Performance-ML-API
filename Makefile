.PHONY: start-inference start-gateway load-test docker-build docker-up docker-down

INFERENCE_PORT?=8001
GATEWAY_PORT?=8000
MAX_BATCH_SIZE?=32
BATCH_TIMEOUT?=0.01
NUM_BATCHING_WORKERS?=2

start-inference:
	@uv run python src/inference/app.py \
		--host 0.0.0.0 \
		--port $(INFERENCE_PORT) \
		--max-batch-size $(MAX_BATCH_SIZE) \
		--batch-timeout $(BATCH_TIMEOUT) \
		--num-batching-workers $(NUM_BATCHING_WORKERS)

start-gateway:
	@uv run python src/gateway/app.py \
		--host 0.0.0.0 \
		--port $(GATEWAY_PORT) \
		--inference-url "http://localhost:$(INFERENCE_PORT)"
		--rate-limit-minute 60
		--rate-limit-hour 1000

load-test:
	@./scripts/load-test.sh -H localhost -P $(GATEWAY_PORT) -u 50 -r 10 -d "30s"
	@open reports/latest.html

docker-build:
	@docker-compose build

docker-up:
	@INFERENCE_PORT=$(INFERENCE_PORT) \
	MAX_BATCH_SIZE=$(MAX_BATCH_SIZE) \
	BATCH_TIMEOUT=$(BATCH_TIMEOUT) \
	NUM_BATCHING_WORKERS=$(NUM_BATCHING_WORKERS) \
	docker-compose up -d

docker-down:
	@docker-compose down