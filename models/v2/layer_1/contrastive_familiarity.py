import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, f1_score


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


def ensure_col(df: pd.DataFrame, col: str, default_val):
    if col not in df.columns:
        df[col] = default_val
    return df


def l2norm(x: torch.Tensor, dim: int = 1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True).clamp(min=eps))


# ----------------------------
# Dataset
# ----------------------------
class ContrastiveDataset(Dataset):
    """Each item: (text, domain_id)"""
    def __init__(self, texts: List[str], domain_ids: List[int]):
        self.texts = texts
        self.domain_ids = domain_ids

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], int(self.domain_ids[i])


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    domain_ids: torch.Tensor


def collate_fn(tokenizer, max_length: int):
    def _fn(items):
        texts, domain_ids = zip(*items)
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
            domain_ids=torch.tensor(domain_ids, dtype=torch.long),
        )
    return _fn


# ----------------------------
# Model: Encoder + Projection head
# ----------------------------
class MeanPoolEncoder(nn.Module):
    """
    Transformer -> mean pool -> projection -> normalized embedding
    """
    def __init__(self, model_name: str, proj_dim: int = 256):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.backbone = AutoModel.from_pretrained(model_name)
        hid = self.backbone.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, proj_dim),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state  # (B,T,H)
        mask = attention_mask.unsqueeze(-1).to(last.dtype)  # (B,T,1)
        pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)  # (B,H)
        z = self.proj(pooled)  # (B,proj_dim)
        z = l2norm(z, dim=1)
        return z


# ----------------------------
# SupCon Loss (with optional hard negatives)
# ----------------------------
def supervised_contrastive_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
    hard_neg_k: int = 0,
) -> torch.Tensor:
    device = z.device
    B = z.size(0)

    # (B,B) similarity
    sim = (z @ z.T) / temperature

    # IMPORTANT: avoid fp16 overflow
    neg_inf = torch.finfo(sim.dtype).min

    # mask self
    self_mask = torch.eye(B, device=device).bool()
    sim = sim.masked_fill(self_mask, neg_inf)

    labels = labels.view(-1, 1)
    pos_mask = torch.eq(labels, labels.T) & (~self_mask)
    neg_mask = (~pos_mask) & (~self_mask)

    if hard_neg_k and hard_neg_k > 0:
        sim_neg = sim.masked_fill(~neg_mask, neg_inf)
        topk_vals, topk_idx = torch.topk(sim_neg, k=min(hard_neg_k, B - 1), dim=1)
        hard_neg_mask = torch.zeros_like(neg_mask, dtype=torch.bool)
        hard_neg_mask.scatter_(1, topk_idx, True)
        neg_mask = hard_neg_mask

    denom_mask = pos_mask | neg_mask
    denom_sim = sim.masked_fill(~denom_mask, neg_inf)

    log_prob = sim - torch.logsumexp(denom_sim, dim=1, keepdim=True)

    pos_count = pos_mask.sum(dim=1).clamp(min=1.0)
    loss = -(pos_mask * log_prob).sum(dim=1) / pos_count
    return loss.mean()



# ----------------------------
# Train
# ----------------------------
def train_contrastive(
    model: MeanPoolEncoder,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    temperature: float,
    hard_neg_k: int,
    grad_clip: float = 1.0,
):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)

    for ep in range(1, epochs + 1):
        pbar = tqdm(loader, desc=f"Train {ep}/{epochs}")
        losses = []
        for batch in pbar:
            input_ids = batch.input_ids.to(device, non_blocking=True)
            attn = batch.attention_mask.to(device, non_blocking=True)
            domain_ids = batch.domain_ids.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                z = model(input_ids, attn)  # (B,D)
                loss = supervised_contrastive_loss(
                    z, domain_ids, temperature=temperature, hard_neg_k=hard_neg_k
                )

            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            losses.append(loss.item())
            pbar.set_postfix(loss=float(np.mean(losses)))


# ----------------------------
# Embedding
# ----------------------------
@torch.no_grad()
def embed_texts(
    model: MeanPoolEncoder,
    texts: List[str],
    device: torch.device,
    batch_size: int,
    max_length: int,
    num_workers: int,
) -> np.ndarray:
    model.eval()
    ds = ContrastiveDataset(texts=texts, domain_ids=[0] * len(texts))  # dummy labels
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn(model.tokenizer, max_length),
        pin_memory=(device.type == "cuda"),
    )

    all_z = []
    for batch in tqdm(loader, desc="Embedding"):
        input_ids = batch.input_ids.to(device, non_blocking=True)
        attn = batch.attention_mask.to(device, non_blocking=True)
        z = model(input_ids, attn).detach().cpu().numpy()
        all_z.append(z)
    return np.concatenate(all_z, axis=0)


# ----------------------------
# Scoring: kNN top-k similarity (recommended) + centroid fallback
# ----------------------------
def build_centroids(z: np.ndarray, domain_ids: np.ndarray) -> Dict[int, np.ndarray]:
    centroids = {}
    for d in np.unique(domain_ids):
        zd = z[domain_ids == d]
        c = zd.mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-12)
        centroids[int(d)] = c
    return centroids


def max_cosine_to_centroids(z: np.ndarray, centroids: Dict[int, np.ndarray]) -> np.ndarray:
    C = np.stack([centroids[k] for k in sorted(centroids.keys())], axis=0)  # (K,D)
    sims = z @ C.T  # (N,K)
    return sims.max(axis=1)


def max_cosine_to_knn(
    z: np.ndarray,
    z_bank: np.ndarray,
    k: int = 10,
    bank_subsample: int = 50000,
    seed: int = 42,
    chunk: int = 20000,
) -> np.ndarray:
    """
    z: (N,D) normalized
    z_bank: (M,D) normalized
    returns max cosine among top-k nearest neighbors (approx by chunked matmul)
    """
    rng = np.random.RandomState(seed)
    bank = z_bank
    if bank_subsample and bank.shape[0] > bank_subsample:
        idx = rng.choice(bank.shape[0], bank_subsample, replace=False)
        bank = bank[idx]

    N = z.shape[0]
    best = np.full((N, k), -1.0, dtype=np.float32)

    for start in range(0, bank.shape[0], chunk):
        end = min(start + chunk, bank.shape[0])
        sims = z @ bank[start:end].T  # (N, chunk)
        part = np.partition(sims, -k, axis=1)[:, -k:]  # (N,k)
        merged = np.concatenate([best, part], axis=1)
        best = np.partition(merged, -k, axis=1)[:, -k:]

    return best.max(axis=1)


# ----------------------------
# Thresholding & Evaluation
# ----------------------------
def threshold_from_benign(unf_scores: np.ndarray, labels: np.ndarray, target_fpr: float) -> float:
    benign_scores = unf_scores[labels == 0]
    if len(benign_scores) == 0:
        raise ValueError("No benign samples to set threshold.")
    perc = 100.0 * (1.0 - target_fpr)
    return float(np.percentile(benign_scores, perc))


def evaluate(scores_unf: np.ndarray, labels: np.ndarray, thr: float) -> dict:
    y_true = labels.astype(int)
    y_pred = (scores_unf >= thr).astype(int)  # unfamiliar => suspicious(1)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()

    out = {
        "threshold": float(thr),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "FPR": fp / (fp + tn + 1e-12),
        "TPR": tp / (tp + fn + 1e-12),
        "TNR": tn / (tn + fp + 1e-12),
        "FNR": fn / (fn + tp + 1e-12),
        "AUROC": float(roc_auc_score(y_true, scores_unf)) if len(np.unique(y_true)) == 2 else None,
        "F1_mal": float(f1_score(y_true, y_pred, zero_division=0)),
        "classification_report": classification_report(
            y_true, y_pred, digits=4,
            target_names=["benign(0)", "unf/mal(1)"],
            zero_division=0
        )
    }
    return out


def pick_threshold_max_f1(scores_unf: np.ndarray, labels: np.ndarray, n_grid: int = 400) -> Tuple[float, float]:
    y = labels.astype(int)
    lo, hi = float(scores_unf.min()), float(scores_unf.max())
    thrs = np.linspace(lo, hi, n_grid)

    best_thr, best_f1 = thrs[0], -1.0
    for thr in thrs:
        pred = (scores_unf >= thr).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = float(f1), float(thr)
    return best_thr, best_f1


def pick_threshold_max_f1_cap_fpr(
    scores_unf: np.ndarray, labels: np.ndarray, fpr_cap: float, n_grid: int = 400
) -> Tuple[float, float, float]:
    """
    Maximize F1 subject to FPR <= fpr_cap (measured on full dev mix).
    Returns (best_thr, best_f1, best_fpr).
    """
    y = labels.astype(int)
    lo, hi = float(scores_unf.min()), float(scores_unf.max())
    thrs = np.linspace(lo, hi, n_grid)

    best_thr, best_f1, best_fpr = thrs[0], -1.0, 1.0
    for thr in thrs:
        pred = (scores_unf >= thr).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        tn, fp, fn, tp = cm.tolist()
        fpr = fp / (fp + tn + 1e-12)
        if fpr <= fpr_cap:
            f1 = f1_score(y, pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr, best_fpr = float(f1), float(thr), float(fpr)
    return best_thr, best_f1, best_fpr


def tpr_at_fpr_sweep(scores_unf: np.ndarray, labels: np.ndarray, fprs=(0.001, 0.005, 0.01, 0.02, 0.05)) -> dict:
    y = labels.astype(int)
    out = {}
    for fpr in fprs:
        thr = threshold_from_benign(scores_unf, y, target_fpr=fpr)
        rep = evaluate(scores_unf, y, thr)
        out[str(fpr)] = {
            "thr": thr, "FPR": rep["FPR"], "TPR": rep["TPR"],
            "TP": rep["TP"], "FN": rep["FN"], "FP": rep["FP"], "TN": rep["TN"],
            "F1_mal": rep["F1_mal"],
        }
    return out


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--benignset", type=str, required=True)
    parser.add_argument("--maliciousset", type=str, required=True)
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--source_col", type=str, default="source")
    parser.add_argument("--label_col", type=str, default="label")

    # model
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.05)  # sharper by default

    # train
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--hard_neg_k", type=int, default=32, help="0=off, else keep top-k hardest negatives per anchor")

    # splits
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_fpr", type=float, default=0.01)

    # probe control
    parser.add_argument("--mal_n", type=int, default=3000)
    parser.add_argument("--mal_frac", type=float, default=0.0)
    parser.add_argument("--mal_dev_n", type=int, default=1000, help="malicious used for dev threshold tuning (only for thr_mode max_f1*)")

    # loader
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)

    # scoring
    parser.add_argument("--score_mode", type=str, default="knn", choices=["knn", "centroid"])
    parser.add_argument("--knn_k", type=int, default=10)
    parser.add_argument("--bank_subsample", type=int, default=50000)

    # threshold selection
    parser.add_argument("--thr_mode", type=str, default="benign_fpr", choices=["benign_fpr", "max_f1", "max_f1_cap_fpr"])

    # output
    parser.add_argument("--out_dir", type=str, default="familiarity_out_v3")

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ----- Load benign -----
    benign_df = read_csv(args.benignset, args.text_col)
    benign_df = ensure_col(benign_df, args.source_col, "unknown_benign")
    benign_df = ensure_col(benign_df, args.label_col, 0)
    benign_df[args.label_col] = 0

    # Map source -> domain_id
    sources = sorted(benign_df[args.source_col].astype(str).unique().tolist())
    src2id = {s: i for i, s in enumerate(sources)}
    benign_df["domain_id"] = benign_df[args.source_col].astype(str).map(src2id).astype(int)

    # Split benign into train/val/test (80/10/10) stratified by domain
    train_df, temp_df = train_test_split(
        benign_df, test_size=0.2, shuffle=True, random_state=args.seed,
        stratify=benign_df["domain_id"]
    )
    val_df, benign_test_df = train_test_split(
        temp_df, test_size=0.5, shuffle=True, random_state=args.seed,
        stratify=temp_df["domain_id"]
    )

    # ----- Load malicious -----
    mal_df = read_csv(args.maliciousset, args.text_col)
    mal_df = ensure_col(mal_df, args.source_col, "unknown_malicious")
    mal_df = ensure_col(mal_df, args.label_col, 1)
    mal_df[args.label_col] = 1

    # malicious probe for TEST
    if args.mal_n and args.mal_n > 0:
        mprobe_df = mal_df.sample(n=min(args.mal_n, len(mal_df)), random_state=args.seed).reset_index(drop=True)
    elif args.mal_frac and args.mal_frac > 0:
        mprobe_df = mal_df.sample(frac=min(args.mal_frac, 1.0), random_state=args.seed).reset_index(drop=True)
    else:
        mprobe_df = mal_df.reset_index(drop=True)

    # malicious dev for threshold tuning (optional)
    mdev_df = mal_df.sample(n=min(args.mal_dev_n, len(mal_df)), random_state=args.seed + 7).reset_index(drop=True)

    # Test mix: benign_test + malicious_probe
    test_df = pd.concat([benign_test_df, mprobe_df], ignore_index=True)
    test_df = test_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    test_df.to_csv("../../../datasets/processed_datasets/test_mix.csv")

    print("Split sizes:", {
        "train": len(train_df),
        "val": len(val_df),
        "benign_test": len(benign_test_df),
        "mprobe": len(mprobe_df),
        "test_mix": len(test_df)
    })
    print("Test label counts:", test_df[args.label_col].value_counts().to_dict())
    print("Benign domains:", len(src2id), "->", src2id)

    # ----- Build model -----
    model = MeanPoolEncoder(args.model_name, proj_dim=args.proj_dim).to(device)

    # ----- Balanced sampling by domain (CRITICAL) -----
    domain_counts = train_df["domain_id"].value_counts().to_dict()
    weights = train_df["domain_id"].map(lambda d: 1.0 / domain_counts[int(d)]).astype(float).to_numpy()

    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )

    # ----- Train SupCon on benign train -----
    train_texts = train_df[args.text_col].astype(str).tolist()
    train_domain_ids = train_df["domain_id"].astype(int).tolist()

    train_loader = DataLoader(
        ContrastiveDataset(train_texts, train_domain_ids),
        batch_size=args.batch_size,
        sampler=sampler,          # <-- balanced
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn(model.tokenizer, args.max_length),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    print(f"Training SupCon (balanced domains) on {len(train_df)} benign samples...")
    train_contrastive(
        model=model,
        loader=train_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        temperature=args.temperature,
        hard_neg_k=args.hard_neg_k,
    )

    # ----- Embed benign TRAIN (bank) -----
    print("Embedding benign TRAIN (bank)...")
    z_train = embed_texts(
        model, train_df[args.text_col].astype(str).tolist(),
        device=device, batch_size=args.batch_size, max_length=args.max_length, num_workers=args.num_workers
    )

    # ----- Score benign VAL to set threshold -----
    print("Scoring benign VAL...")
    z_val = embed_texts(
        model, val_df[args.text_col].astype(str).tolist(),
        device=device, batch_size=args.batch_size, max_length=args.max_length, num_workers=args.num_workers
    )

    if args.score_mode == "centroid":
        centroids = build_centroids(z_train, train_df["domain_id"].astype(int).to_numpy())
        max_sim_val = max_cosine_to_centroids(z_val, centroids)
    else:
        max_sim_val = max_cosine_to_knn(
            z_val, z_train, k=args.knn_k, bank_subsample=args.bank_subsample,
            seed=args.seed, chunk=20000
        )

    unf_val = 1.0 - max_sim_val
    y_val = val_df[args.label_col].astype(int).to_numpy()  # all 0

    # ----- Threshold selection mode -----
    if args.thr_mode == "benign_fpr":
        thr = threshold_from_benign(unf_val, y_val, target_fpr=args.target_fpr)
        print(f"Chosen thr (benign_fpr) @ target_fpr={args.target_fpr}: {thr:.6f}")
    else:
        # build dev mix: benign_val + malicious_dev
        dev_df = pd.concat([val_df, mdev_df], ignore_index=True).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        z_dev = embed_texts(
            model, dev_df[args.text_col].astype(str).tolist(),
            device=device, batch_size=args.batch_size, max_length=args.max_length, num_workers=args.num_workers
        )
        if args.score_mode == "centroid":
            max_sim_dev = max_cosine_to_centroids(z_dev, centroids)
        else:
            max_sim_dev = max_cosine_to_knn(
                z_dev, z_train, k=args.knn_k, bank_subsample=args.bank_subsample,
                seed=args.seed, chunk=20000
            )
        unf_dev = 1.0 - max_sim_dev
        y_dev = dev_df[args.label_col].astype(int).to_numpy()

        if args.thr_mode == "max_f1":
            thr, best_f1 = pick_threshold_max_f1(unf_dev, y_dev)
            print(f"Chosen thr (max_f1): {thr:.6f} (dev_f1={best_f1:.4f})")
        else:
            thr, best_f1, best_fpr = pick_threshold_max_f1_cap_fpr(unf_dev, y_dev, fpr_cap=args.target_fpr)
            print(f"Chosen thr (max_f1_cap_fpr<= {args.target_fpr}): {thr:.6f} (dev_f1={best_f1:.4f}, dev_fpr={best_fpr:.4f})")

    # ----- Test mix -----
    print("Scoring TEST MIX...")
    z_test = embed_texts(
        model, test_df[args.text_col].astype(str).tolist(),
        device=device, batch_size=args.batch_size, max_length=args.max_length, num_workers=args.num_workers
    )

    if args.score_mode == "centroid":
        if "centroids" not in locals():
            centroids = build_centroids(z_train, train_df["domain_id"].astype(int).to_numpy())
        max_sim_test = max_cosine_to_centroids(z_test, centroids)
    else:
        max_sim_test = max_cosine_to_knn(
            z_test, z_train, k=args.knn_k, bank_subsample=args.bank_subsample,
            seed=args.seed, chunk=20000
        )

    unf_test = 1.0 - max_sim_test
    y_test = test_df[args.label_col].astype(int).to_numpy()

    report = evaluate(unf_test, y_test, thr)
    print("\n=== Test Report (Familiarity -> Unfamiliarity) ===")
    print(json.dumps({k: v for k, v in report.items() if k != "classification_report"}, indent=2))
    print(report["classification_report"])

    sweep = tpr_at_fpr_sweep(unf_test, y_test, fprs=(0.001, 0.005, 0.01, 0.02, 0.05))
    print("\n=== TPR @ FPR sweep (threshold from benign in TEST MIX) ===")
    print(json.dumps(sweep, indent=2))

    # ----- Save -----
    ckpt_path = os.path.join(args.out_dir, "encoder.pt")
    meta_path = os.path.join(args.out_dir, "meta.json")
    srcmap_path = os.path.join(args.out_dir, "src2id.json")
    test_scores_path = os.path.join(args.out_dir, "unf_test.npy")
    bank_path = os.path.join(args.out_dir, "z_train_bank.npy")

    torch.save(model.state_dict(), ckpt_path)
    np.save(test_scores_path, unf_test)
    np.save(bank_path, z_train)

    with open(srcmap_path, "w", encoding="utf-8") as f:
        json.dump(src2id, f, indent=2)

    meta = {
        "model_name": args.model_name,
        "proj_dim": args.proj_dim,
        "temperature": args.temperature,
        "max_length": args.max_length,
        "target_fpr": args.target_fpr,
        "threshold": float(thr),
        "seed": args.seed,
        "mal_n": args.mal_n,
        "mal_frac": args.mal_frac,
        "num_domains": len(src2id),
        "score_mode": args.score_mode,
        "knn_k": args.knn_k,
        "bank_subsample": args.bank_subsample,
        "thr_mode": args.thr_mode,
        "hard_neg_k": args.hard_neg_k,
        "epochs": args.epochs,
        "lr": args.lr,
        "report_F1_mal": report["F1_mal"],
        "report_FPR": report["FPR"],
        "report_TPR": report["TPR"],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved: {ckpt_path}")
    print(f"Saved: {srcmap_path}")
    print(f"Saved: {meta_path}")
    print(f"Saved: {test_scores_path}")
    print(f"Saved: {bank_path}")


if __name__ == "__main__":
    main()
