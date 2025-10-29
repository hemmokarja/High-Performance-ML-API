# Monitoring Setup Guide

This guide explains how to set up and use Prometheus and Grafana monitoring for your inference API.

## Architecture

```
┌─────────────┐     scrapes       ┌────────────┐     queries     ┌─────────┐
│ Inference   │ ─────────────────▶│ Prometheus │◀────────────────│ Grafana │
│ API:8001    │  /metrics every   │   :9090    │  PromQL         │  :3000  │
│             │     10 seconds    │            │                 │         │
└─────────────┘                   └────────────┘                 └─────────┘
                                        │
                                        │ stores time-series data
                                        ▼
                                  ┌────────────┐
                                  │ Prometheus │
                                  │   Volume   │
                                  └────────────┘
```

## Access Monitoring Tools

| Service | URL | Credentials |
|---------|-----|-------------|
| Inference API | http://localhost:8001 | - |
| Raw Metrics | http://localhost:8001/metrics | - |
| Prometheus | http://localhost:9090 | - |
| Grafana | http://localhost:3000 | admin/admin |

## What Gets Monitored

### Request Metrics
- **Request Rate**: Total requests per second (success/error breakdown)
- **Error Rate**: Percentage of failed requests
- **Request Latency**: End-to-end latency (p50, p95, p99 percentiles)

### Batching Metrics
- **Batch Size**: Distribution and average batch sizes
- **Batch Wait Time**: Time requests wait before batch processing starts
- **Queue Size**: Current number of requests waiting in queue
- **Inflight Batches**: Number of batches currently being processed

### Model Performance
- **Inference Time**: Time spent in model.predict() (p50, p95, p99)

## Using Grafana

### First Login
1. Go to http://localhost:3000
2. Login with `admin` / `admin`
3. Change password (or skip)
4. Navigate to Dashboards → Inference API - Dynamic Batcher

### Dashboard Panels

The pre-configured dashboard includes:

1. **Request Rate** - Monitors throughput
2. **Error Rate** - Tracks reliability
3. **Request Latency** - End-to-end user experience
4. **Average Batch Size** - Batching efficiency
5. **Queue Size** - Request backlog
6. **Inflight Batches** - Processing concurrency
7. **Batch Wait Time** - Queueing delays
8. **Inference Time** - Model performance
9. **Batch Size Distribution** - Batching patterns

## Using Prometheus

### Query Examples

Access Prometheus at http://localhost:9090/graph and try these queries:

```promql
# Current request rate
rate(batcher_requests_total[1m])

# Error rate percentage
100 * rate(batcher_requests_total{status="error"}[1m]) / rate(batcher_requests_total[1m])

# P95 latency
histogram_quantile(0.95, rate(batcher_request_latency_seconds_bucket[5m]))

# Average batch size
rate(batcher_batch_size_sum[5m]) / rate(batcher_batch_size_count[5m])

# Queue depth
batcher_queue_size

# Inference throughput (batches per second)
rate(batcher_batch_size_count[1m])
```

## Viewing Raw Metrics

To see the raw Prometheus metrics text format:

```bash
curl http://localhost:8001/metrics
```

Example output:
```
# HELP batcher_requests_total Total number of prediction requests
# TYPE batcher_requests_total counter
batcher_requests_total{status="success"} 1523.0
batcher_requests_total{status="error"} 12.0

# HELP batcher_batch_size Distribution of batch sizes processed
# TYPE batcher_batch_size histogram
batcher_batch_size_bucket{le="1.0"} 45.0
batcher_batch_size_bucket{le="2.0"} 89.0
...
```

## Data Retention

- **Prometheus**: Retains data for 30 days (configurable in docker-compose.yml)
- **Grafana**: Queries Prometheus for historical data
- **Volumes**: Data persists across container restarts

## Performance Impact

Monitoring overhead is minimal:
- Prometheus scrapes every 5 seconds (configurable)
- Metrics collection is ~microseconds per request
- CPU: <0.5 cores combined (Prometheus + Grafana)
- Memory: ~512MB combined

## Troubleshooting

### Prometheus Not Scraping

Check Prometheus targets:
```bash
# Visit http://localhost:9090/targets
# All targets should show "UP" status
```

If inference-api shows "DOWN":
1. Check inference API is running: `docker-compose ps`
2. Verify /metrics endpoint: `curl http://localhost:8001/metrics`
3. Check Prometheus logs: `docker-compose logs prometheus`

### Grafana Shows No Data

1. Verify Prometheus datasource: Grafana → Configuration → Data Sources → Prometheus → Test
2. Check Prometheus has data: http://localhost:9090/graph
3. Ensure time range in Grafana includes data (try "Last 15 minutes")

### Dashboard Not Appearing

1. Check provisioning: `docker-compose logs grafana`
2. Verify files exist:
   ```bash
   ls -la monitoring/grafana/provisioning/dashboards/
   ls -la monitoring/grafana/dashboards/
   ```
3. Restart Grafana: `docker-compose restart grafana`

## Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [PromQL Cheat Sheet](https://promlabs.com/promql-cheat-sheet/)
