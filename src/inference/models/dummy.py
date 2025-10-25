import time
import random
from typing import List, Any


class DummyModel:
    """Simulates a model with realistic CPU/GPU-bound computation"""
    def __init__(self, base_latency: float = 0.05, per_item_latency: float = 0.005):
        self.base_latency = base_latency
        self.per_item_latency = per_item_latency

    def predict(self, batch: List[Any]) -> List[Any]:
        """Synchronous batch inference (simulates blocking CPU/GPU work)"""
        batch_size = len(batch)
        total_time = self.base_latency + (self.per_item_latency * batch_size * 0.3)
        total_time += random.uniform(-0.005, 0.005)
        time.sleep(total_time)
        return [f"result_{item}" for item in batch]
