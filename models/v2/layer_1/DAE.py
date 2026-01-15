import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

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
        raise ValueError(f"{path} must contain '{text_col}'. Found columns: {list(df.columns)[:30]}")
    return df


def ensure_label(df: pd.DataFrame, label_col: str, value: int) -> pd.DataFrame:
    if label_col not in df.columns:
        df[label_col] = value
    df[label_col] = value
    df[label_col] = df[label_col].astype(int)
    return df


def safe_value_counts(df: pd.DataFrame, col: str) -> Dict:
    if col in df.columns:
        return df[col].value_counts().to_dict()
    return {}


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
        summed = (last_hidden * mask).sum(dim=1)                   # (B, H)
        denom = mask.sum(dim=1).clamp(min=1e-6)                    # (B, 1)
        mean_pooled = summed / denom                               # (B, H)

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

    # New AMP API (fix FutureWarning)
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for ep in range(1, epochs + 1):
        pbar = tqdm(loader, desc=f"Train epoch {ep}/{epochs}")
        losses = []
        for batch in pbar:
            input_ids = batch.input_ids.to(device, non_blocking=True)
            attn = batch.attention_mask.to(device, non_blocking=True)

            # embeddings are frozen
            with torch.no_grad():
                x = embedder.encode(input_ids, attn)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
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


def threshold_from_benign(scores: np.ndarray, labels: np.ndarray, fpr: float) -> float:
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
        "threshold": float(thr),
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
        y_true, y_pred, digits=4,
        target_names=["benign(0)", "malicious(1)"],
        zero_division=0
    )
    return out


def tpr_at_fpr_sweep(scores: np.ndarray, labels: np.ndarray, fprs=(0.001, 0.005, 0.01, 0.02, 0.05)) -> dict:
    """
    Report TPR when threshold is set by benign at various target FPRs.
    """
    y = labels.astype(int)
    out = {}
    for fpr in fprs:
        thr = threshold_from_benign(scores, y, fpr=fpr)
        y_pred = (scores >= thr).astype(int)
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel().tolist()
        out[str(fpr)] = {
            "thr": float(thr),
            "FPR": fp / (fp + tn + 1e-12),
            "TPR": tp / (tp + fn + 1e-12),
            "TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn),
        }
    return out


def sample_malicious(df: pd.DataFrame, seed: int, mal_frac: float, mal_n: int) -> pd.DataFrame:
    if mal_n and mal_n > 0:
        n = min(mal_n, len(df))
        return df.sample(n=n, random_state=seed).reset_index(drop=True)
    if mal_frac and mal_frac > 0:
        frac = min(mal_frac, 1.0)
        return df.sample(frac=frac, random_state=seed).reset_index(drop=True)
    # default: use all
    return df.reset_index(drop=True)


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--benignset", type=str, default="../../../datasets/processed_datasets/benign_universal.csv")
    parser.add_argument("--maliciousset", type=str, default="../../../datasets/processed_datasets/malicious_universal.csv")
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="label")

    # model
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--l2_norm", action="store_true")
    parser.add_argument("--no_l2_norm", action="store_true")
    parser.add_argument("--max_length", type=int, default=256)

    # AE
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    # train
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--err_kind", type=str, default="mse", choices=["mse", "l1"])
    parser.add_argument("--target_fpr", type=float, default=0.01)

    # loader
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)

    # splits
    parser.add_argument("--seed", type=int, default=42)

    # malicious probe control (use mal_n if set; else mal_frac)
    parser.add_argument("--mal_frac", type=float, default=0.03, help="malicious fraction used in test mix (0=disable)")
    parser.add_argument("--mal_n", type=int, default=0, help="malicious count used in test mix (0=disable)")

    # output
    parser.add_argument("--out_dir", type=str, default="ae_out")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # L2 norm decision
    l2_norm = True
    if args.no_l2_norm:
        l2_norm = False
    if args.l2_norm:
        l2_norm = True

    # ----------------------------
    # 1) Load & split benign (80/10/10)
    # ----------------------------
    benign_df = read_csv(args.benignset, args.text_col)
    benign_df = ensure_label(benign_df, args.label_col, 0)

    train_df, temp_df = train_test_split(
        benign_df, test_size=0.2, shuffle=True, random_state=args.seed
    )
    val_df, benign_test_df = train_test_split(
        temp_df, test_size=0.5, shuffle=True, random_state=args.seed
    )

    # ----------------------------
    # 2) Load malicious & sample probe for test mix
    # ----------------------------
    malicious_df = read_csv(args.maliciousset, args.text_col)
    malicious_df = ensure_label(malicious_df, args.label_col, 1)

    mprobe_df = sample_malicious(malicious_df, seed=args.seed, mal_frac=args.mal_frac, mal_n=args.mal_n)

    # test mix
    test_df = pd.concat([benign_test_df, mprobe_df], ignore_index=True)
    test_df = test_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    print("Split sizes:", {"train": len(train_df), "val": len(val_df), "benign_test": len(benign_test_df), "mprobe": len(mprobe_df), "test_mix": len(test_df)})
    print("Test label counts:", test_df[args.label_col].value_counts().to_dict())

    # ----------------------------
    # 3) Build models
    # ----------------------------
    embedder = HFEmbedder(args.model_name, l2_norm=l2_norm).to(device)
    ae = MLPAutoEncoder(
        dim_in=embedder.hidden_size,
        dim_latent=args.latent_dim,
        hidden=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    # ----------------------------
    # 4) Dataloaders
    # ----------------------------
    pin = (device.type == "cuda")

    train_loader = DataLoader(
        TextDataset(train_df[args.text_col].astype(str).tolist(), labels=None),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn(embedder.tokenizer, args.max_length),
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    val_loader = DataLoader(
        TextDataset(val_df[args.text_col].astype(str).tolist(), labels=val_df[args.label_col].astype(int).tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn(embedder.tokenizer, args.max_length),
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    test_loader = DataLoader(
        TextDataset(test_df[args.text_col].astype(str).tolist(), labels=test_df[args.label_col].astype(int).tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn(embedder.tokenizer, args.max_length),
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    # ----------------------------
    # 5) Train
    # ----------------------------
    print(f"Training AE on {len(train_df)} benign samples...")
    train_ae(
        ae=ae,
        embedder=embedder,
        loader=train_loader,
        device=device,
        lr=args.lr,
        epochs=args.epochs,
        err_kind=args.err_kind,
    )

    # ----------------------------
    # 6) Threshold from benign val
    # ----------------------------
    print("Scoring validation set (benign-only) ...")
    val_scores, val_y = score_dataset(ae, embedder, val_loader, device, err_kind=args.err_kind)
    thr = threshold_from_benign(val_scores, val_y, fpr=args.target_fpr)
    print(f"Chosen threshold @ target_fpr={args.target_fpr}: {thr:.6f}")

    # ----------------------------
    # 7) Test on mix
    # ----------------------------
    print("Scoring test mix ...")
    test_scores, test_y = score_dataset(ae, embedder, test_loader, device, err_kind=args.err_kind)
    report = evaluate_with_threshold(test_scores, test_y, thr)

    print("\n=== Test Report ===")
    print(json.dumps({k: v for k, v in report.items() if k != "classification_report"}, indent=2))
    print(report["classification_report"])

    # extra: sweep operating points
    sweep = tpr_at_fpr_sweep(test_scores, test_y, fprs=(0.001, 0.005, 0.01, 0.02, 0.05))
    print("\n=== TPR @ FPR sweep (threshold set by benign in test_mix) ===")
    print(json.dumps(sweep, indent=2))

    # extra: per-source TPR if available
    per_source = {}
    if "source" in test_df.columns:
        # align sources with scores order (test_df order == loader order == scores order)
        srcs = test_df["source"].astype(str).tolist()
        for src in sorted(set(srcs)):
            idx = np.array([s == src for s in srcs], dtype=bool)
            if idx.sum() == 0:
                continue
            y_s = test_y[idx]
            sc_s = test_scores[idx]
            # only meaningful if this slice has malicious
            if (y_s == 1).sum() == 0:
                continue
            rep_s = evaluate_with_threshold(sc_s, y_s, thr)
            per_source[src] = {
                "n": int(idx.sum()),
                "malicious": int((y_s == 1).sum()),
                "TPR": rep_s["TPR"],
                "FNR": rep_s["FNR"],
                "FPR": rep_s["FPR"],
            }

        print("\n=== Per-source summary (if source column exists) ===")
        print(json.dumps(per_source, indent=2))

    # ----------------------------
    # 8) Save artifacts
    # ----------------------------
    ckpt_path = os.path.join(args.out_dir, "ae.pt")
    meta_path = os.path.join(args.out_dir, "meta.json")
    npy_path  = os.path.join(args.out_dir, "test_scores.npy")
    stats_path = os.path.join(args.out_dir, "test_mix_stats.json")

    torch.save(ae.state_dict(), ckpt_path)
    np.save(npy_path, test_scores)

    meta = {
        "model_name": args.model_name,
        "l2_norm": l2_norm,
        "max_length": args.max_length,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "err_kind": args.err_kind,
        "threshold": float(thr),
        "target_fpr": float(args.target_fpr),
        "seed": int(args.seed),
        "mal_frac": float(args.mal_frac),
        "mal_n": int(args.mal_n),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    stats = {
        "sizes": {"train": len(train_df), "val": len(val_df), "benign_test": len(benign_test_df), "mprobe": len(mprobe_df), "test_mix": len(test_df)},
        "test_label_counts": test_df[args.label_col].value_counts().to_dict(),
        "mal_source_counts": safe_value_counts(mprobe_df, "source"),
        "sweep": sweep,
        "per_source": per_source,
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved: {ckpt_path}")
    print(f"Saved: {meta_path}")
    print(f"Saved: {npy_path}")
    print(f"Saved: {stats_path}")


if __name__ == "__main__":
    main()
