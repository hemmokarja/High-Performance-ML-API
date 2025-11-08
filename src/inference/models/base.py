from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import torch


class BaseEmbeddingModel(ABC):

    @abstractmethod
    def predict(
        self, batch_tensors: Dict[str, torch.Tensor | np.ndarray]
    ) -> List[np.ndarray]:
        pass

    @abstractmethod
    def encode_inputs(self, texts: str) -> Dict[str, torch.Tensor | np.ndarray]:
        pass

    @abstractmethod
    def to(self, device: torch.device | str):
        return self

    @abstractmethod
    def eval(self):
        return self
