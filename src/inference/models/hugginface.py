import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class HugginFaceEmbeddingModel:
    def __init__(self, model_name, hf_token=None, device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        self.model = AutoModel.from_pretrained(model_name, token=hf_token)
        self.device = device or torch.device("cpu")

        self.model.eval()

    def predict(self, batch):
        encoded_input = self.tokenizer(
            batch, padding=True, truncation=True, return_tensors="pt"
        )
        encoded_input = encoded_input.to(self.device)
        with torch.no_grad():
            model_out = self.model(**encoded_input)

        emb = self._mean_pool(model_out, encoded_input["attention_mask"])  # [B, T]
        emb = F.normalize(emb, p=2, dim=1)
        return emb

    @staticmethod
    def _mean_pool(model_output, attention_mask):
        """
        Mean pool last hidden states into a single embedding, accounting for different
        length sequences.
        """
        # first element contains token embeddings
        token_embeddings = model_output[0]  # [B, T, C]
        expanded_mask = (
            attention_mask
            .unsqueeze(-1)  # [B, T, 1]
            .expand(-1, -1, token_embeddings.size(-1))  # [B, T, C]
            .float()
        )
        return (
            torch.sum(token_embeddings * expanded_mask, 1)  # [B, C]
            / torch.clamp(expanded_mask.sum(1), min=1e-9)  # [B, C]
        )  # [B, C]
