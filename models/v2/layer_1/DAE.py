import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_csv(path: str, text_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if text_col not in df.columns:
        raise ValueError(f"{path} must contain a '{text_col}' column. Found: {list(df.columns)[:20]} ...")
    return df


# ----------------------------
# Dataset
# ----------------------------
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


def collate_fn(tokenizer, max_length: int):
    def _fn(batch_items):
        texts, labels = zip(*batch_items)
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


# ----------------------------
# Embedding model wrapper
# ----------------------------
class HFEmbedder(nn.Module):
    """
    Text -> fixed-size embedding via transformer + mean pooling (masked) + optional L2 normalization.
    """
    def __init__(self, model_name: str, l2_norm: bool = True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.l2_norm = l2_norm

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state  # (B, T, H)

        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)  # (B, T, 1)
        summed = (last_hidden * mask).sum(dim=1)                  # (B, H)
        denom = mask.sum(dim=1).clamp(min=1e-6)                   # (B, 1)
        mean_pooled = summed / denom                               # (B, H)

        # ✅ IMPORTANT for sentence-transformers + anomaly detection stability
        if self.l2_norm:
            mean_pooled = F.normalize(mean_pooled, p=2, dim=1)

        return mean_pooled


# ----------------------------
# AutoEncoder
# ----------------------------
class MLPAutoEncoder(nn.Module):
    def __init__(self, dim_in: int, dim_latent: int = 128, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim_latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(dim_latent, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim_in),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


# ----------------------------
# Train / Eval
# ----------------------------
def reconstruction_error(x: torch.Tensor, x_hat: torch.Tensor, kind: str = "mse") -> torch.Tensor:
    if kind == "mse":
        return torch.mean((x_hat - x) ** 2, dim=1)
    elif kind == "l1":
        return torch.mean(torch.abs(x_hat - x), dim=1)
    else:
        raise ValueError("kind must be 'mse' or 'l1'")


def train_ae(
    ae: nn.Module,
    embedder: HFEmbedder,
    loader: DataLoader,
    device: torch.device,
    lr: float,
    epochs: int,
    err_kind: str = "mse",
    grad_clip: float = 1.0,
) -> None:
    ae.train()
    embedder.eval()
    opt = torch.optim.AdamW(ae.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for ep in range(1, epochs + 1):
        pbar = tqdm(loader, desc=f"Train epoch {ep}/{epochs}")
        losses = []
        for batch in pbar:
            input_ids = batch.input_ids.to(device, non_blocking=True)
            attn = batch.attention_mask.to(device, non_blocking=True)

            with torch.no_grad():
                x = embedder.encode(input_ids, attn)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                x_hat, _ = ae(x)
                loss = reconstruction_error(x, x_hat, kind=err_kind).mean()

            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(ae.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            losses.append(loss.item())
            pbar.set_postfix(loss=float(np.mean(losses)))


@torch.no_grad()
def score_dataset(
    ae: nn.Module,
    embedder: HFEmbedder,
    loader: DataLoader,
    device: torch.device,
    err_kind: str = "mse",
) -> Tuple[np.ndarray, np.ndarray]:
    ae.eval()
    embedder.eval()
    all_scores = []
    all_labels = []

    for batch in tqdm(loader, desc="Scoring"):
        input_ids = batch.input_ids.to(device, non_blocking=True)
        attn = batch.attention_mask.to(device, non_blocking=True)
        labels = batch.labels.cpu().numpy()

        x = embedder.encode(input_ids, attn)
        x_hat, _ = ae(x)
        scores = reconstruction_error(x, x_hat, kind=err_kind).cpu().numpy()

        all_scores.append(scores)
        all_labels.append(labels)

    return np.concatenate(all_scores), np.concatenate(all_labels)


def pick_threshold_from_benign(scores: np.ndarray, labels: np.ndarray, fpr: float = 0.01) -> float:
    benign_scores = scores[labels == 0]
    if len(benign_scores) == 0:
        raise ValueError("No benign samples (label=0) in validation to set threshold.")
    percentile = 100.0 * (1.0 - fpr)
    return float(np.percentile(benign_scores, percentile))


def evaluate_with_threshold(scores: np.ndarray, labels: np.ndarray, thr: float) -> dict:
    y_true = labels.astype(int)
    y_pred = (scores >= thr).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()

    out = {
        "threshold": thr,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "FPR": fp / (fp + tn + 1e-12),
        "TPR": tp / (tp + fn + 1e-12),
        "TNR": tn / (tn + fp + 1e-12),
        "FNR": fn / (fn + tp + 1e-12),
    }

    if len(np.unique(y_true)) == 2:
        out["AUROC"] = float(roc_auc_score(y_true, scores))
    else:
        out["AUROC"] = None

    out["classification_report"] = classification_report(
        y_true, y_pred, digits=4, target_names=["benign(0)", "malicious(1)"]
    )
    return out


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benignset", type=str, default="../../../datasets/processed_datasets/benign_universal.csv")
    parser.add_argument("--maliciousset", type=str, default="../../../datasets/processed_datasets/malicious_universal.csv")

    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="label")

    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--l2_norm", action="store_true", help="L2-normalize embeddings (recommended).")
    parser.add_argument("--no_l2_norm", action="store_true", help="Disable L2 normalization.")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--err_kind", type=str, default="mse", choices=["mse", "l1"])

    parser.add_argument("--target_fpr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="ae_out")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Decide l2_norm default True unless explicitly disabled
    l2_norm = True
    if args.no_l2_norm:
        l2_norm = False
    if args.l2_norm:
        l2_norm = True

    # Load data
    benign_df = read_csv(args.benignset, args.text_col)
    if args.label_col not in benign_df.columns:
        benign_df[args.label_col] = 0
    benign_df[args.label_col] = benign_df[args.label_col].astype(int)

    train_df, temp_df = train_test_split(
        benign_df, random_state=args.seed, shuffle=True, test_size=0.2
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=args.seed, shuffle=True
    )

    malicious_df = read_csv(args.maliciousset, args.text_col)
    if args.label_col not in malicious_df.columns:
        malicious_df[args.label_col] = 1
    malicious_df[args.label_col] = 1  # force

    # take 3% malicious as probe
    mprobe_df = malicious_df.sample(frac=0.03, random_state=args.seed).reset_index(drop=True)

    # mix test
    test_df = pd.concat([test_df, mprobe_df], ignore_index=True)
    test_df = test_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    print("Test label counts:", test_df[args.label_col].value_counts().to_dict())

    # Train data: benign-only if label exists
    if args.label_col in train_df.columns:
        train_df = train_df[train_df[args.label_col].astype(int) == 0].reset_index(drop=True)

    train_texts = train_df[args.text_col].astype(str).tolist()
    val_texts   = val_df[args.text_col].astype(str).tolist()
    test_texts  = test_df[args.text_col].astype(str).tolist()

    val_labels  = val_df[args.label_col].astype(int).tolist()
    test_labels = test_df[args.label_col].astype(int).tolist()

    # Models
    embedder = HFEmbedder(args.model_name, l2_norm=l2_norm).to(device)
    ae = MLPAutoEncoder(
        dim_in=embedder.hidden_size,
        dim_latent=args.latent_dim,
        hidden=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    # DataLoaders
    pin = (device.type == "cuda")
    train_loader = DataLoader(
        TextDataset(train_texts, labels=None),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn(embedder.tokenizer, args.max_length),
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        TextDataset(val_texts, labels=val_labels),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn(embedder.tokenizer, args.max_length),
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        TextDataset(test_texts, labels=test_labels),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn(embedder.tokenizer, args.max_length),
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    # Train
    print(f"Training AE on {len(train_texts)} benign samples...")
    train_ae(ae, embedder, train_loader, device, lr=args.lr, epochs=args.epochs, err_kind=args.err_kind)

    # Validation -> threshold
    print("Scoring validation set...")
    val_scores, val_y = score_dataset(ae, embedder, val_loader, device, err_kind=args.err_kind)
    thr = pick_threshold_from_benign(val_scores, val_y, fpr=args.target_fpr)
    print(f"Chosen threshold @ target_fpr={args.target_fpr}: {thr:.6f}")

    # Test eval
    print("Scoring test set...")
    test_scores, test_y = score_dataset(ae, embedder, test_loader, device, err_kind=args.err_kind)
    report = evaluate_with_threshold(test_scores, test_y, thr)

    print("\n=== Test Report ===")
    print(json.dumps({k: v for k, v in report.items() if k != "classification_report"}, indent=2))
    print(report["classification_report"])

    # Save artifacts
    ckpt_path = os.path.join(args.out_dir, "ae.pt")
    meta_path = os.path.join(args.out_dir, "meta.json")
    npy_path  = os.path.join(args.out_dir, "test_scores.npy")

    torch.save(ae.state_dict(), ckpt_path)
    np.save(npy_path, test_scores)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "l2_norm": l2_norm,
                "max_length": args.max_length,
                "latent_dim": args.latent_dim,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "err_kind": args.err_kind,
                "threshold": thr,
                "target_fpr": args.target_fpr,
                "seed": args.seed,
            },
            f,
            indent=2,
        )

    print(f"\nSaved: {ckpt_path}")
    print(f"Saved: {meta_path}")
    print(f"Saved: {npy_path}")


if __name__ == "__main__":
    main()