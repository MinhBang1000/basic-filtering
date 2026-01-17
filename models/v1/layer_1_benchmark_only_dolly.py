#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import time
import math
import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------
# 1. ARGUMENT PARSING
# -------------------------
parser = argparse.ArgumentParser(description="Layer-1 NLP Anomaly Detection Benchmark (Multi-Embedding Support)")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--openai", action="store_true", help="OpenAI text-embedding-3-large (3072 dims)")
group.add_argument("--bert", action="store_true", help="BERT-base-nli-mean-tokens (768 dims)")
group.add_argument("--minilm", action="store_true", help="All-MiniLM-L6-v2 (384 dims)")

args = parser.parse_args()

# -------------------------
# 2. CONFIG & SELECTION
# -------------------------
RANDOM_SEED = 42
BATCH_SIZE = 64
MAX_SAMPLES = None 
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.60, 0.20, 0.20
VAL_PERCENTILE = 95
SYNTH_OUTLIER_RATIO = 1.0
SYNTH_MODE = "gaussian"

# AE Training Config
AE_EPOCHS = 20
AE_LR = 1e-3
AE_BATCH = 512

# Determine Embedding Config
if args.openai:
    EMBED_TYPE = "openai"
    MODEL_NAME = "text-embedding-3-large"
    DIM = 3072
    # Check API Key only for OpenAI
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing in .env or environment.")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
elif args.bert:
    EMBED_TYPE = "local"
    MODEL_NAME = "bert-base-nli-mean-tokens"
    DIM = 768
elif args.minilm:
    EMBED_TYPE = "local"
    MODEL_NAME = "all-MiniLM-L6-v2"
    DIM = 384

print(f"\n[INFO] Mode: {EMBED_TYPE.upper()} | Model: {MODEL_NAME} | Dimensions: {DIM}")

# -------------------------
# 3. EMBEDDING ENGINE
# -------------------------
def get_embeddings(text_list):
    if EMBED_TYPE == "openai":
        embs = []
        for i in tqdm(range(0, len(text_list), BATCH_SIZE), desc="OpenAI Embedding"):
            batch = text_list[i:i+BATCH_SIZE]
            resp = client.embeddings.create(model=MODEL_NAME, input=batch)
            embs.extend([e.embedding for e in resp.data])
        return np.array(embs, dtype=np.float32)
    else:
        # GPU detection for SentenceTransformers
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(MODEL_NAME, device=device)
        print(f"Loading local model to: {device}")
        return model.encode(text_list, batch_size=BATCH_SIZE, show_progress_bar=True).astype(np.float32)

# -------------------------
# 4. UTILS & SYNTHETIC DATA
# -------------------------
np.random.seed(RANDOM_SEED)

def make_synth_outliers(X_ref, n_out, mode="gaussian", seed=42):
    rng = np.random.default_rng(seed)
    if mode == "gaussian":
        mu, sigma = X_ref.mean(axis=0), X_ref.std(axis=0) + 1e-6
        inflate = 1.2
        return rng.normal(loc=mu, scale=sigma * inflate, size=(n_out, X_ref.shape[1])).astype(np.float32)
    elif mode == "shuffle":
        Z = X_ref.copy()
        for d in range(Z.shape[1]): rng.shuffle(Z[:, d])
        reps = math.ceil(n_out / Z.shape[0])
        return np.concatenate([Z] * reps, axis=0)[:n_out].astype(np.float32)

def compute_threshold(val_scores, percentile):
    return float(np.percentile(val_scores, percentile))

def benign_metrics(test_scores, threshold):
    tnr = float((test_scores < threshold).mean())
    return tnr, 1.0 - tnr

def synthetic_auc(score_fn, X_val_benign, seed=RANDOM_SEED):
    n_out = int(len(X_val_benign) * SYNTH_OUTLIER_RATIO)
    X_out = make_synth_outliers(X_val_benign, n_out=n_out, mode=SYNTH_MODE, seed=seed)
    s_b, s_o = score_fn(X_val_benign), score_fn(X_out)
    y = np.concatenate([np.zeros_like(s_b), np.ones_like(s_o)])
    s = np.concatenate([s_b, s_o])
    return float(roc_auc_score(y, s))

def pack_row(name, threshold, tnr, fpr, auc_syn, test_scores, latency):
    return {
        "Model": name,
        "Threshold(VAL_p{})".format(VAL_PERCENTILE): threshold,
        "Benign_Recall(TNR)_TEST": tnr,
        "FPR_TEST": fpr,
        "AUC_SYNTH_VAL": auc_syn,
        "Latency_sec": float(latency),
    }

# -------------------------
# 5. DYNAMIC AUTOENCODER CLASS
# -------------------------
class DynamicAE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Kiến trúc co giãn theo số chiều đầu vào
        h1 = 512 if input_dim > 512 else input_dim // 2
        h2 = 128 if input_dim > 128 else input_dim // 4
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(h2, h1), nn.ReLU(),
            nn.Linear(h1, input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

# -------------------------
# 6. PIPELINE EXECUTION
# -------------------------
dolly_size = "100%" if MAX_SAMPLES is None else f"{(MAX_SAMPLES/15000)*100:.1f}%"
print(f"\n[{dolly_size}] Loading Dolly 15K...")
ds = load_dataset("databricks/databricks-dolly-15k")["train"]
texts = list(ds["instruction"])[:MAX_SAMPLES] if MAX_SAMPLES else list(ds["instruction"])

print(f"[2/5] Generating Embeddings via {MODEL_NAME}...")
X = get_embeddings(texts)

print("[3/5] Splitting and Scaling Data...")
X_train, X_tmp = train_test_split(X, test_size=(1.0 - TRAIN_FRAC), random_state=RANDOM_SEED)
val_ratio_tmp = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
X_val, X_test = train_test_split(X_tmp, test_size=(1.0 - val_ratio_tmp), random_state=RANDOM_SEED)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val, X_test = scaler.transform(X_val).astype(np.float32), scaler.transform(X_test).astype(np.float32)

print("[4/5] Running Anomaly Detection Models...")
results = []

# --- MODEL WRAPPERS ---
models_to_run = {
    "IsolationForest": lambda: IsolationForest(contamination=0.05, random_state=RANDOM_SEED).fit(X_train),
    "OneClassSVM": lambda: OneClassSVM(nu=0.05, gamma="scale").fit(X_train),
    "kNN-Distance": lambda: NearestNeighbors(n_neighbors=5).fit(X_train),
    "PCA-Reconstruction": lambda: PCA(n_components=0.95, random_state=RANDOM_SEED).fit(X_train)
}

# Run Standard Models
for name, init_fn in models_to_run.items():
    print(f"  > Training {name}...")
    start_t = time.time()
    m = init_fn()
    
    if name == "IsolationForest" or name == "OneClassSVM":
        score_fn = lambda x: (-m.score_samples(x)).astype(np.float32)
    elif name == "kNN-Distance":
        score_fn = lambda x: m.kneighbors(x)[0].mean(axis=1).astype(np.float32)
    elif name == "PCA-Reconstruction":
        score_fn = lambda x: np.linalg.norm(x - m.inverse_transform(m.transform(x)), axis=1).astype(np.float32)
    
    val_s, test_s = score_fn(X_val), score_fn(X_test)
    thr = compute_threshold(val_s, VAL_PERCENTILE)
    tnr, fpr = benign_metrics(test_s, thr)
    auc_s = synthetic_auc(score_fn, X_val)
    results.append(pack_row(name, thr, tnr, fpr, auc_s, test_s, time.time() - start_t))

# Run Deep AutoEncoder
print("  > Training Deep AutoEncoder...")
start_t = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"
ae = DynamicAE(DIM).to(device)
optimizer = optim.Adam(ae.parameters(), lr=AE_LR)
criterion = nn.MSELoss()

Xtr_tensor = torch.tensor(X_train, device=device)
for epoch in range(AE_EPOCHS):
    idx = np.random.permutation(len(X_train))
    for i in range(0, len(idx), AE_BATCH):
        batch = Xtr_tensor[idx[i:i+AE_BATCH]]
        optimizer.zero_grad()
        loss = criterion(ae(batch), batch)
        loss.backward()
        optimizer.step()

ae.eval()
def ae_score(X_):
    with torch.no_grad():
        xt = torch.tensor(X_, device=device)
        return ((ae(xt) - xt) ** 2).mean(dim=1).cpu().numpy().astype(np.float32)

val_s, test_s = ae_score(X_val), ae_score(X_test)
thr = compute_threshold(val_s, VAL_PERCENTILE)
tnr, fpr = benign_metrics(test_s, thr)
auc_s = synthetic_auc(ae_score, X_val)
results.append(pack_row("Deep-AutoEncoder", thr, tnr, fpr, auc_s, test_s, time.time() - start_t))

# -------------------------
# 7. RESULTS
# -------------------------
final_df = pd.DataFrame(results).sort_values("Benign_Recall(TNR)_TEST", ascending=False)
print("\n" + "="*50)
print(f"FINAL RESULTS ({MODEL_NAME})")
print("="*50)
print(final_df.to_string(index=False))

out_name = f"benchmark_l1_{MODEL_NAME.replace('/', '_')}.csv"
final_df.to_csv(out_name, index=False)
print(f"\n[DONE] Saved to {out_name}")