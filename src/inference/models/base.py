from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import torch


class BaseEmbeddingModel(ABC):

    @abstractmethod
    def predict(self, batch_data: List[Dict[str, torch.Tensor]]) -> List[np.ndarray]:
        pass

    @abstractmethod
    def encode_inputs(self, text: str) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def to(self, device: torch.device | str):
        return self

    @abstractmethod
    def eval(self):
        return self
