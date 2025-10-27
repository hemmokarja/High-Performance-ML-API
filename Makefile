.PHONY: start-api-local load-test docker-build docker-up docker-down

PORT?=8001
MAX_BATCH_SIZE?=32
BATCH_TIMEOUT?=0.01
NUM_BATCHING_WORKERS?=2

start-api-local:
	@uv run python src/inference/app.py \
		--host 0.0.0.0 \
		--port $(PORT) \
		--max-batch-size $(MAX_BATCH_SIZE) \
		--batch-timeout $(BATCH_TIMEOUT) \
		--num-batching-workers $(NUM_BATCHING_WORKERS)

load-test:
	@./scripts/load-test.sh -H localhost -P $(PORT) -u 50 -r 10 -d "30s"
	@open reports/latest.html

docker-build:
	@docker-compose build

docker-up:
	@PORT=$(PORT) \
	MAX_BATCH_SIZE=$(MAX_BATCH_SIZE) \
	BATCH_TIMEOUT=$(BATCH_TIMEOUT) \
	NUM_BATCHING_WORKERS=$(NUM_BATCHING_WORKERS) \
	docker-compose up -d

docker-down:
	@docker-compose down