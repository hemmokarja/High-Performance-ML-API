from typing import Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class HugginFaceEmbeddingModel:
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

        self.model.eval()

    def predict(self, batch: List[str]) -> List[np.ndarray]:
        encoded_input = self.tokenizer(
            batch, padding=True, truncation=True, return_tensors="pt"
        )
        encoded_input = encoded_input.to(self.device)
        with torch.no_grad():
            model_out = self.model(**encoded_input)

        latent_emb = model_out[0]  # first element contains token embeddings
        emb = self._mean_pool(latent_emb, encoded_input["attention_mask"])  # [B, T]
        emb = F.normalize(emb, p=2, dim=1)  # [B, T]
        emb = emb.numpy()
        return [emb[i] for i in range(emb.shape[0])]

    @staticmethod
    def _mean_pool(
        latent_emb: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean pool last hidden states into a single embedding, accounting for different
        length sequences.
        """
        expanded_mask = (
            attention_mask
            .unsqueeze(-1)  # [B, T, 1]
            .expand(-1, -1, latent_emb.size(-1))  # [B, T, C]
            .float()
        )
        return (
            torch.sum(latent_emb * expanded_mask, 1)  # [B, C]
            / torch.clamp(expanded_mask.sum(1), min=1e-9)  # [B, C]
        )  # [B, C]
