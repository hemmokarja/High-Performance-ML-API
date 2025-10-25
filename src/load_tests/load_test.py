import asyncio
import random
import time
from dataclasses import dataclass
from typing import List, Any, Dict, Optional

import numpy as np
import structlog

from server.inference.batcher import DynamicBatcher, NoBatchingWrapper

logger = structlog.get_logger(__name__)


@dataclass
class LoadTestConfig:
    num_requests: int
    steady_wait: float  # in steady mode, wait between requests
    random_min_wait: float  # in random mode, min wait between requests
    random_max_wait: float  # in random mode, max wait between requests


@dataclass
class LoadTestResult:
    name: str
    batch_size: int
    results: Dict[str, Dict[str, Any]]


@dataclass
class LoadTestDynamicBatcherConfig:
    batch_sizes: list[int]  # batch sizes to test in load test
    batch_interval: float  # batcher max wait between batches
    num_workers: int  # num of batcher inference workers


async def _load_test_pattern(
    batcher: DynamicBatcher | NoBatchingWrapper,
    load_test_config: LoadTestConfig,
    request_pattern: str,
) -> Dict[str, Any]:
    tasks = []
    task_start_times = []
    start_time = time.time()

    for i in range(load_test_config.num_requests):
        task_start = time.time()
        task = asyncio.create_task(batcher.predict(f"input_{i}"))
        tasks.append(task)
        task_start_times.append(task_start)
        
        if request_pattern == "burst":
            continue
        elif request_pattern == "steady":
            await asyncio.sleep(load_test_config.steady_wait)
        elif request_pattern == "random":
            wait = random.uniform(
                load_test_config.random_min_wait, load_test_config.random_max_wait
            )
            await asyncio.sleep(wait)
    
    results = await asyncio.gather(*tasks)
    end_time = time.time()

    latencies = [end_time - start for start in task_start_times]
    
    total_time = end_time - start_time
    throughput = load_test_config.num_requests / total_time
    
    return {
        "num_requests": load_test_config.num_requests,
        "total_time": total_time,
        "throughput": throughput,
        "latency_p50": np.percentile(latencies, 50),
        "latency_p95": np.percentile(latencies, 95),
        "latency_p99": np.percentile(latencies, 99),
        "latency_mean": np.mean(latencies),
        "latency_max": np.max(latencies),
    }


async def _load_test_patterns(
    batcher: DynamicBatcher | NoBatchingWrapper, load_test_config: LoadTestConfig
) -> Dict[str, Dict[str, Any]]:
    results = {}
    for pattern in ["burst", "steady", "random"]:
        result = await _load_test_pattern(batcher, load_test_config, pattern)
        results[pattern] = result
    return results


async def _load_test_batcher(
    batcher: DynamicBatcher | NoBatchingWrapper,
    load_test_config: LoadTestConfig,
    warmup_requests: int = 10,
) -> Dict[str, Dict[str, Any]]:

    await batcher.start()

    if warmup_requests > 0:
        warmup_tasks = [
            batcher.predict(f"warmup_{i}") 
            for i in range(warmup_requests)
        ]
        await asyncio.gather(*warmup_tasks)

    result = await _load_test_patterns(batcher, load_test_config)
    await batcher.shutdown()

    return result


async def run_full_load_test(
    model,
    load_test_config: LoadTestConfig,
    batcher_config: LoadTestDynamicBatcherConfig
) -> List[LoadTestResult]:
    results = []

    logger.info("Load testing NoBatchingWrapper...")
    batcher = NoBatchingWrapper(model)
    result = await _load_test_batcher(batcher, load_test_config)
    results.append(LoadTestResult("no_batching", 1, result))

    for batch_size in batcher_config.batch_sizes:
        logger.info(f"Load testing DynamicBatcher with batch size {batch_size}...")
        batcher = DynamicBatcher(
            model,
            max_batch_size=batch_size,
            batch_interval=batcher_config.batch_interval,
            num_workers=batcher_config.num_workers
        )
        result = await _load_test_batcher(batcher, load_test_config)
        results.append(
            LoadTestResult(f"dynamic_batch_size_{batch_size}", batch_size, result)
        )

    return results


def _print_metrics(
    metrics: Dict[str, Any],
    time_speedup: Optional[float] = None,
    latency_improvement: Optional[float] = None
):
    print(f"   Total time:          {metrics['total_time']:.3f}s")
    print(f"   Throughput:          {metrics['throughput']:.1f} req/s")
    print(f"   Latency (p50):       {metrics['latency_p50']*1000:.1f}ms")
    print(f"   Latency (p95):       {metrics['latency_p95']*1000:.1f}ms")
    print(f"   Latency (p99):       {metrics['latency_p99']*1000:.1f}ms")

    if time_speedup is not None:
        print(f"   Time speedup:        {time_speedup:.2f}x")
    if latency_improvement is not None:
        print(f"   Latency improvement: {latency_improvement:.1f}%")


def print_load_test_results(
    results: list[LoadTestResult], batcher_config: LoadTestDynamicBatcherConfig
):
    if not results:
        print("No results to display")
        return

    baseline = [r for r in results if r.name == "no_batching"]
    assert len(baseline) == 1
    baseline = baseline[0]
    batch_results = [r for r in results if r.name != "no_batching"]
    
    patterns = list(baseline.results.keys())
    num_requests = baseline.results[patterns[0]]["num_requests"]

    print("=" * 80)
    print("LOAD TEST RESULTS")
    print("=" * 80)
    print(f"\nNumber of requests: {num_requests}\n")

    # print results for each pattern
    for pattern in patterns:
        print("=" * 80)
        print(f"PATTERN: {pattern.upper()}")
        print("=" * 80, "\n")

        # baseline
        print(f"WITHOUT BATCHING (batch_size=1):")
        baseline_metrics = baseline.results[pattern]
        _print_metrics(baseline_metrics)

        # batched results
        for result in batch_results:
            metrics = result.results[pattern]

            time_speedup = baseline_metrics["total_time"] / metrics["total_time"]
            latency_improvement = (
                (baseline_metrics["latency_p95"] - metrics["latency_p95"])
                / baseline_metrics["latency_p95"] * 100
            )

            print(
                f"\nWITH BATCHING (max_batch={result.batch_size}, "
                f"interval={batcher_config.batch_interval}, "
                f"workers={batcher_config.num_workers}):"
            )
            _print_metrics(metrics, time_speedup, latency_improvement)

        print()
