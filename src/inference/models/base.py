from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch


class BaseEmbeddingModel(ABC):

    @abstractmethod
    def predict(self, batch: List[str]) -> List[np.ndarray]:
        pass

    @abstractmethod
    def to(self, device: torch.device | str):
        return self

    @abstractmethod
    def eval(self):
        return self
