import os
from typing import Dict, Optional, List

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
import structlog
from torch.nn.utils import rnn
from transformers import AutoTokenizer, AutoModel

from inference.models.base import BaseEmbeddingModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = structlog.get_logger(__name__)


def model_factory(
    model_name: str,
    hf_token: Optional[str] = None,
    device: Optional[torch.device | str] = None,
    use_onnx: bool = False
):
    if use_onnx:
        return HuggingFaceONNXEmbeddingModel(model_name, device)
    else:
        return HuggingFaceEmbeddingModel(model_name, hf_token, device)


class HuggingFaceEmbeddingModel(BaseEmbeddingModel):
    def __init__(
        self,
        model_name: str,
        hf_token: Optional[str] = None,
        device: Optional[torch.device | str] = None
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        self.model = AutoModel.from_pretrained(model_name, token=hf_token)
        self.device = device or torch.device("cpu")
        self.device_str = str(self.device)

        self.model.to(self.device)
        self.model.eval()

        logger.info(
            f"Initialized HuggingFaceEmbeddingModel({model_name}) "
            f"on device {self.device}"
        )

    def encode_inputs(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize single text on CPU, async safe"""
        encoded = self.tokenizer(
            text, 
            padding=False,
            truncation=True, 
            return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in encoded.items()}

    def predict(self, batch_data: List[Dict[str, torch.Tensor]]) -> List[np.ndarray]:

        input_ids = rnn.pad_sequence(
            [item["input_ids"] for item in batch_data],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = rnn.pad_sequence(
            [item["attention_mask"] for item in batch_data],
            batch_first=True,
            padding_value=0
        )
        batch = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device)
        }

        with torch.no_grad():
            model_out = self.model(**batch)

        last_hidden = model_out[0]  # first element contains token embeddings
        emb = self._mean_pool(last_hidden, batch["attention_mask"])  # [B, T]
        emb = F.normalize(emb, p=2, dim=1)  # [B, T]
        emb = emb.cpu().numpy()
        return [emb[i] for i in range(emb.shape[0])]

    @staticmethod
    def _mean_pool(
        last_hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean pool last hidden states into a single embedding, accounting for different
        length sequences.
        """
        expanded_mask = (
            attention_mask
            .unsqueeze(-1)  # [B, T, 1]
            .expand(-1, -1, last_hidden.size(-1))  # [B, T, C]
            .float()
        )
        return (
            torch.sum(last_hidden * expanded_mask, 1)  # [B, C]
            / torch.clamp(expanded_mask.sum(1), min=1e-9)  # [B, C]
        )  # [B, C]

    def to(self, device: torch.device | str):
        self.device = torch.device(device)
        self.device_str = str(self.device)
        self.model.to(self.device)
        return self

    def eval(self):
        self.model.eval()
        return self


def _get_and_validate_onnx_path(model_name):
    model_id = model_name.replace("/", "_")
    onnx_path = f"./onnx_models/huggingface/{model_id}.onnx"
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: '{onnx_path}'")
    return onnx_path


def _set_up_providers(device_str):
    if device_str == "cuda":
        return [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2GB
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                }
            )
        ]
    else:
        return ["CPUExecutionProvider"]


class HuggingFaceONNXEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name: str, device: Optional[torch.device | str] = None):
        self.model_name = model_name
        self.device = device or torch.device("cpu")
        self.device_str = str(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.onnx_path = _get_and_validate_onnx_path(model_name)
        providers = _set_up_providers(self.device_str)
        self.session = ort.InferenceSession(self.onnx_path, providers=providers)

        logger.info(f"Initialized ONNX model: {self.onnx_path}", providers=providers)
    
    def encode_inputs(self, text: str) -> Dict[str, np.ndarray]:
        """Tokenize single text, async safe, CPU only"""
        encoded = self.tokenizer(
            text,
            padding=False,  # Don't pad yet
            truncation=True,
            return_tensors="np"
        )
        return {k: v.squeeze(0) for k, v in encoded.items()}

    def predict(self, batch_data: List[Dict[str, np.ndarray]]) -> List[np.ndarray]:

        # collate to batch
        max_len = max(item["input_ids"].shape[0] for item in batch_data)
        batch_size = len(batch_data)

        input_ids = np.full(
            (batch_size, max_len), self.tokenizer.pad_token_id, dtype=np.int64
        )
        attention_mask = np.zeros((batch_size, max_len), dtype=np.int64)

        for i, item in enumerate(batch_data):
            seq_len = item["input_ids"].shape[0]
            input_ids[i, :seq_len] = item["input_ids"]
            attention_mask[i, :seq_len] = item["attention_mask"]

        batch = {"input_ids": input_ids, "attention_mask": attention_mask}
        outputs = self.session.run(None, batch)

        last_hidden = outputs[0]
        pooled = self._mean_pool(last_hidden, attention_mask)
        pooled = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)

        return [pooled[i] for i in range(pooled.shape[0])]

    @staticmethod
    def _mean_pool(last_hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        expanded_mask = np.expand_dims(attention_mask, -1)  # [B, T, 1]
        return (
            np.sum(last_hidden * expanded_mask, axis=1)  # [B, C]
            / np.clip(expanded_mask.sum(1), 1e-9, None)  # [B, 1]
        )

    def to(*args):
        pass

    def eval(*args):
        pass
