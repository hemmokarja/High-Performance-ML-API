from prometheus_client import Counter, Histogram, Gauge

REQUESTS_TOTAL = Counter(
    "batcher_requests_total",
    "Total number of prediction requests",
    ["status"]  # success, error
)

BATCH_SIZE = Histogram(
    "batcher_batch_size",
    "Distribution of batch sizes processed",
    buckets=[1, 2, 4, 8, 16, 32, 64, 128]
)

REQUEST_LATENCY = Histogram(
    "batcher_request_latency_seconds",
    "End-to-end request latency",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

BATCH_WAIT_TIME = Histogram(
    "batcher_batch_wait_time_seconds",
    "Time spent waiting to form a batch",
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1]
)

INFERENCE_TIME = Histogram(
    "batcher_inference_time_seconds",
    "Model inference time per batch",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)

QUEUE_SIZE = Gauge(
    "batcher_queue_size",
    "Current number of requests in queue"
)

INFLIGHT_BATCHES = Gauge(
    "batcher_inflight_batches",
    "Current number of batches being processed"
)
