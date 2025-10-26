.PHONY: start-api load-test

API_PORT=8001

start-api:
	@uv run python src/inference/api/app.py \
		--host 0.0.0.0 \
		--port $(API_PORT) \
		--max-batch-size 32 \
		--batch-timeout 0.01 \
		--num-batching-workers 2

load-test:
	@./scripts/load-test.sh -H localhost -P $(API_PORT) -u 50 -r 10 -d "30s"
	@open reports/latest.html
