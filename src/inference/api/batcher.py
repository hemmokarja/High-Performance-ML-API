import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Any

from inference.api import metrics

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class Request:
    id: int
    data: Any
    future: asyncio.Future
    timestamp: float  # for latency tracking


class DynamicBatcher:
    """
    Batches multiple inference requests together to improve throughput.

    This class collects incoming prediction requests into batches and processes
    them together, which is much more efficient for ML models than processing
    requests one-by-one (especially for GPU inference).

    How it works:
    - Start long-lived background workers that continuously process requests
    - When predict() is called, the request is added to a queue
    - Workers collect requests into batches until either:
        * The batch reaches max_batch_size, OR
        * batch_timeout time has elapsed since the first request
    - The batch is then processed by the model in a separate thread pool (this allows
      background workers to collect batches while model does inference)
    - Results are returned to the original callers via their futures
    """
    def __init__(
        self, 
        model,
        max_batch_size: int = 32,
        batch_timeout: float = 0.01,
        num_workers: int = 1
    ):
        self.model = model
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.num_workers = num_workers

        # use asyncio.Queue (thread-safe, proper async primitive)
        self.request_queue = asyncio.Queue()

        # thread pool for blocking model inference
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")

        # long-lived worker tasks
        self.worker_tasks: List[asyncio.Task] = []

        self.inflight_batches = 0
        self._started = False

        # start background task to update gauge metrics
        self._metrics_task = None

    def is_started(self):
        return self._started

    async def start(self):
        """Start worker pool - call during app startup"""
        if self._started:
            return

        self._started = True
        loop = asyncio.get_event_loop()

        # create fixed pool of workers
        for i in range(self.num_workers):
            task = asyncio.create_task(
                self._batch_collector(loop, worker_id=i)
            )
            self.worker_tasks.append(task)

        # start metrics updater
        self._metrics_task = asyncio.create_task(self._update_gauges())

    async def shutdown(self):
        """Graceful shutdown - call during app teardown"""
        if not self._started:
            return

        # cancel metrics task
        if self._metrics_task:
            self._metrics_task.cancel()
            try:
                await self._metrics_task
            except asyncio.CancelledError:
                pass

        # send sentinel values to signal workers to stop
        for _ in range(self.num_workers):
            await self.request_queue.put(None)

        # wait for all workers to finish
        for task in self.worker_tasks:
            await task

        # shutdown thread pool
        self.executor.shutdown(wait=True)

    async def _update_gauges(self):
        """Background task to periodically update gauge metrics"""
        while True:
            try:
                metrics.QUEUE_SIZE.set(self.request_queue.qsize())
                metrics.INFLIGHT_BATCHES.set(self.inflight_batches)
                await asyncio.sleep(1)  # update every second
            except asyncio.CancelledError:
                break

    async def predict(self, data: Any) -> Any:
        if not self._started:
            raise RuntimeError("Batcher not started. Call start() first.")

        request_start = time.time()
        future = asyncio.Future()
        request = Request(
            id=id(data), data=data, future=future, timestamp=request_start
        )

        await self.request_queue.put(request)

        try:
            result = await future
            metrics.REQUESTS_TOTAL.labels(status="success").inc()
            metrics.REQUEST_LATENCY.observe(time.time() - request_start)
            return result
        except Exception as e:
            metrics.REQUESTS_TOTAL.labels(status="error").inc()
            metrics.REQUEST_LATENCY.observe(time.time() - request_start)
            raise

    async def _batch_collector(self, loop, worker_id: int):
        """Long-lived worker that processes batches"""
        while True:
            # wait for first request (blocking)
            first_item = await self.request_queue.get()

            # check for shutdown sentinel
            if first_item is None:
                return

            # start batch with first request
            batch = [first_item]
            batch_start = time.time()
            deadline = batch_start + self.batch_timeout

            # collect more requests until timeout or batch full
            while len(batch) < self.max_batch_size:
                remaining_time = deadline - time.time()

                # if we've exceeded the timeout, process what we have
                if remaining_time <= 0:
                    break

                try:
                    # wait for next request with timeout
                    item = await asyncio.wait_for(
                        self.request_queue.get(),
                        timeout=remaining_time
                    )

                    # check for shutdown sentinel and process current batch before
                    # exiting if exist
                    if item is None:
                        if batch:
                            await self._process_batch(loop, batch, worker_id)
                        return

                    batch.append(item)

                except asyncio.TimeoutError:
                    # timeout reached, process what we have
                    break

            # process the batch
            wait_time = time.time() - batch_start
            await self._process_batch(loop, batch, worker_id, wait_time)

    async def _process_batch(
        self, loop, batch: List[Request], worker_id: int, wait_time: float = 0
    ):
        start_time = time.time()
        batch_data = [req.data for req in batch]

        metrics.BATCH_SIZE.observe(len(batch))
        metrics.BATCH_WAIT_TIME.observe(wait_time)

        self.inflight_batches += 1
        try:
            # execute blocking inference in a separate thread pool to not block
            # event loop
            results = await loop.run_in_executor(
                self.executor,
                self.model.predict,
                batch_data
            )
        except Exception as e:
            # set exception on all futures
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)
            return
        finally:
            self.inflight_batches -= 1

        end_time = time.time()
        inference_time = end_time - start_time

        # record inference time
        metrics.INFERENCE_TIME.observe(inference_time)

        logger.info(
            "Batch processed",
            worker_id=worker_id,
            batch_size=len(batch),
            wait_ms=round(wait_time * 1000, 1),
            inference_ms=round(inference_time * 1000, 1),
        )

        # return results to waiting futures
        for req, result in zip(batch, results):
            if not req.future.done():
                req.future.set_result(result)


class NoBatchingWrapper:
    """Wrapper that processes requests individually"""
    def __init__(self, model, **kwargs):
        self.model = model
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")
        self._started = False

    def is_started(self):
        return self._started

    async def start(self):
        self._started = True
    
    async def shutdown(self):
        self.executor.shutdown(wait=True)

    async def predict(self, data: Any) -> Any:
        """Process single request in thread pool"""
        request_start = time.time()
        loop = asyncio.get_event_loop()

        try:
            results = await loop.run_in_executor(
                self.executor,
                self.model.predict,
                [data]
            )
            metrics.REQUESTS_TOTAL.labels(status="success").inc()
            metrics.REQUEST_LATENCY.observe(time.time() - request_start)
            return results[0]
        except Exception as e:
            metrics.REQUESTS_TOTAL.labels(status="error").inc()
            metrics.REQUEST_LATENCY.observe(time.time() - request_start)
            raise
