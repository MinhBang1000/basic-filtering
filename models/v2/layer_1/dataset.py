import pandas as pd
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import List, Optional
from transformers import AutoTokenizer

class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: Optional[List[int]] = None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if self.labels is None:
            return self.texts[idx], -1
        return self.texts[idx], int(self.labels[idx])

@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor

def collate_fn(tokenizer, max_length):
    def _fn(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return Batch(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            labels=torch.tensor(labels, dtype=torch.long),
        )
    return _fn

def load_csv(path, text_col, label_col=None):
    df = pd.read_csv(path)
    texts = df[text_col].astype(str).tolist()
    labels = None
    if label_col and label_col in df.columns:
        labels = df[label_col].astype(int).tolist()
    return texts, labels
