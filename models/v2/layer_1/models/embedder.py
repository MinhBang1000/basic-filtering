import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class HFEmbedder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size

    @torch.no_grad()
    def encode(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        summed = (last_hidden * mask).sum(1)
        denom = mask.sum(1).clamp(min=1e-6)
        return summed / denom
