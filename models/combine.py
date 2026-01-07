#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Layer-1 NLP Anomaly Detection Benchmark (Benign-only training)
- Dataset: databricks/databricks-dolly-15k (instruction field)
- Embeddings: OpenAI text-embedding-3-large (3072 dims)
- Models: IsolationForest, OneClassSVM, kNN-Distance, PCA-Reconstruction, Deep AutoEncoder
- Metrics:
    * Benign_Recall(TNR) on TEST  (aka Acceptance Rate)
    * FPR on TEST
    * Synthetic_AUC on VAL (benign vs synthetic pseudo-outliers)  [meaningful AUC without real malicious GT]
    * Threshold (picked on VAL by percentile)
    * Score stats on TEST
    * Latency_sec (fit + score)

Plug & play:
    export OPENAI_API_KEY="..."
    pip install datasets tqdm python-dotenv openai scikit-learn torch
    python benchmark_layer1.py
"""

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

# -------------------------
# CONFIG
# -------------------------
OPENAI_MODEL = "text-embedding-3-large"   # 3072 dims
RANDOM_SEED = 42
BATCH_SIZE = 64

MAX_SAMPLES = 5000        # set None for full ~15k
TRAIN_FRAC = 0.60
VAL_FRAC = 0.20
TEST_FRAC = 0.20

# Threshold policy: choose threshold as VAL_PERCENTILE of VAL scores (higher score => more suspicious)
VAL_PERCENTILE = 95

# Synthetic outliers for AUC (benign VAL vs synthetic)
SYNTH_OUTLIER_RATIO = 1.0   # number of synthetic outliers relative to benign val size (1.0 = same count)
SYNTH_MODE = "gaussian"     # "gaussian" or "shuffle"

# AE config
AE_EPOCHS = 20
AE_LR = 1e-3
AE_BATCH = 512             # mini-batch for AE training (avoid full-batch RAM spikes)

# -------------------------
# INIT
# -------------------------
np.random.seed(RANDOM_SEED)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is missing. Put it in .env or export it.")

# -------------------------
# LOAD DATA
# -------------------------
print("Loading Dolly 15K...")
ds = load_dataset("databricks/databricks-dolly-15k")["train"]
texts = ds["instruction"]
if MAX_SAMPLES is not None:
    texts = texts[:MAX_SAMPLES]
texts = list(texts)

# -------------------------
# EMBEDDINGS
# -------------------------
def embed_texts(text_list):
    embs = []
    for i in tqdm(range(0, len(text_list), BATCH_SIZE), desc="Embedding"):
        batch = text_list[i:i+BATCH_SIZE]
        resp = client.embeddings.create(model=OPENAI_MODEL, input=batch)
        embs.extend([e.embedding for e in resp.data])
    X = np.array(embs, dtype=np.float32)
    return X

X = embed_texts(texts)
print(f"Embedding shape: {X.shape} (dim={X.shape[1]})")

# -------------------------
# SPLIT (train/val/test)
# -------------------------
# First split train vs (val+test)
X_train, X_tmp = train_test_split(
    X,
    test_size=(1.0 - TRAIN_FRAC),
    random_state=RANDOM_SEED
)

# Then split val vs test equally from tmp (since VAL_FRAC == TEST_FRAC)
val_ratio_within_tmp = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
X_val, X_test = train_test_split(
    X_tmp,
    test_size=(1.0 - val_ratio_within_tmp),
    random_state=RANDOM_SEED
)

# -------------------------
# SCALE (fit on train only) + enforce float32
# -------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val   = scaler.transform(X_val).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)

# -------------------------
# SYNTHETIC OUTLIERS FOR AUC
# -------------------------
def make_synth_outliers(X_ref, n_out, mode="gaussian", seed=42):
    rng = np.random.default_rng(seed)
    if mode == "gaussian":
        # sample from a wider Gaussian using ref mean/std per dimension
        mu = X_ref.mean(axis=0)
        sigma = X_ref.std(axis=0) + 1e-6
        # inflate sigma to push samples away (tunable)
        inflate = 3.0
        Z = rng.normal(loc=mu, scale=sigma * inflate, size=(n_out, X_ref.shape[1])).astype(np.float32)
        return Z
    elif mode == "shuffle":
        # shuffle each dimension independently (break correlations)
        Z = X_ref.copy()
        idx = rng.permutation(Z.shape[0])
        Z = Z[idx]
        for d in range(Z.shape[1]):
            rng.shuffle(Z[:, d])
        if Z.shape[0] >= n_out:
            return Z[:n_out].astype(np.float32)
        reps = math.ceil(n_out / Z.shape[0])
        Zrep = np.concatenate([Z] * reps, axis=0)[:n_out]
        return Zrep.astype(np.float32)
    else:
        raise ValueError("SYNTH_MODE must be 'gaussian' or 'shuffle'.")

# -------------------------
# METRIC / EVAL
# -------------------------
def compute_threshold(val_scores, percentile):
    return float(np.percentile(val_scores, percentile))

def benign_metrics(test_scores, threshold):
    # benign is "normal": accepted if score < threshold
    tnr = float((test_scores < threshold).mean())
    fpr = float(1.0 - tnr)
    return tnr, fpr

def synthetic_auc(model_name, score_fn, X_val_benign, seed=RANDOM_SEED):
    # Create synthetic outliers and compute AUC on VAL (benign vs synth).
    n_out = int(len(X_val_benign) * SYNTH_OUTLIER_RATIO)
    X_out = make_synth_outliers(X_val_benign, n_out=n_out, mode=SYNTH_MODE, seed=seed)

    s_b = score_fn(X_val_benign)
    s_o = score_fn(X_out)

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
        "Mean_Test_Score": float(np.mean(test_scores)),
        "Std_Test_Score": float(np.std(test_scores)),
        "Latency_sec": float(latency),
    }

# -------------------------
# MODELS
# -------------------------
results = []

# 1) Isolation Forest
print("\n=== IsolationForest ===")
start = time.time()
iforest = IsolationForest(contamination=0.05, random_state=RANDOM_SEED)
iforest.fit(X_train)

def iforest_score(X_):
    return (-iforest.score_samples(X_)).astype(np.float32)  # higher => more suspicious

val_scores = iforest_score(X_val)
test_scores = iforest_score(X_test)
thr = compute_threshold(val_scores, VAL_PERCENTILE)
tnr, fpr = benign_metrics(test_scores, thr)
auc_syn = synthetic_auc("IsolationForest", iforest_score, X_val, seed=RANDOM_SEED+1)
lat = time.time() - start
results.append(pack_row("IsolationForest", thr, tnr, fpr, auc_syn, test_scores, lat))

# 2) One-Class SVM
print("\n=== OneClassSVM ===")
start = time.time()
ocsvm = OneClassSVM(nu=0.05, gamma="scale")
ocsvm.fit(X_train)

def ocsvm_score(X_):
    return (-ocsvm.score_samples(X_)).astype(np.float32)  # higher => more suspicious

val_scores = ocsvm_score(X_val)
test_scores = ocsvm_score(X_test)
thr = compute_threshold(val_scores, VAL_PERCENTILE)
tnr, fpr = benign_metrics(test_scores, thr)
auc_syn = synthetic_auc("OneClassSVM", ocsvm_score, X_val, seed=RANDOM_SEED+2)
lat = time.time() - start
results.append(pack_row("OneClassSVM", thr, tnr, fpr, auc_syn, test_scores, lat))

# 3) kNN Distance
print("\n=== kNN-Distance ===")
start = time.time()
knn = NearestNeighbors(n_neighbors=5)
knn.fit(X_train)

def knn_score(X_):
    dists = knn.kneighbors(X_)[0].mean(axis=1)
    return dists.astype(np.float32)  # higher => more suspicious

val_scores = knn_score(X_val)
test_scores = knn_score(X_test)
thr = compute_threshold(val_scores, VAL_PERCENTILE)
tnr, fpr = benign_metrics(test_scores, thr)
auc_syn = synthetic_auc("kNN-Distance", knn_score, X_val, seed=RANDOM_SEED+3)
lat = time.time() - start
results.append(pack_row("kNN-Distance", thr, tnr, fpr, auc_syn, test_scores, lat))

# 4) PCA Reconstruction
print("\n=== PCA-Reconstruction ===")
start = time.time()
pca = PCA(n_components=0.95, random_state=RANDOM_SEED)
pca.fit(X_train)

def pca_score(X_):
    rec = pca.inverse_transform(pca.transform(X_))
    err = np.linalg.norm(X_ - rec, axis=1)
    return err.astype(np.float32)  # higher => more suspicious

val_scores = pca_score(X_val)
test_scores = pca_score(X_test)
thr = compute_threshold(val_scores, VAL_PERCENTILE)
tnr, fpr = benign_metrics(test_scores, thr)
auc_syn = synthetic_auc("PCA-Reconstruction", pca_score, X_val, seed=RANDOM_SEED+4)
lat = time.time() - start
results.append(pack_row("PCA-Reconstruction", thr, tnr, fpr, auc_syn, test_scores, lat))

# 5) Deep AutoEncoder
print("\n=== Deep AutoEncoder ===")
start = time.time()

import torch
import torch.nn as nn
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"

class AE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 512), nn.ReLU(),
            nn.Linear(512, dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

dim = X_train.shape[1]
ae = AE(dim).to(device)
opt = optim.Adam(ae.parameters(), lr=AE_LR)
loss_fn = nn.MSELoss()

# mini-batch training
Xtr = torch.tensor(X_train, dtype=torch.float32)
loader_idx = np.arange(len(X_train))

ae.train()
for epoch in range(AE_EPOCHS):
    np.random.shuffle(loader_idx)
    for i in range(0, len(loader_idx), AE_BATCH):
        idx = loader_idx[i:i+AE_BATCH]
        xb = torch.tensor(X_train[idx], dtype=torch.float32, device=device)
        opt.zero_grad()
        rec = ae(xb)
        loss = loss_fn(rec, xb)
        loss.backward()
        opt.step()

ae.eval()
@torch.no_grad()
def ae_score_np(X_):
    xt = torch.tensor(X_, dtype=torch.float32, device=device)
    rec = ae(xt)
    err = ((rec - xt) ** 2).mean(dim=1).detach().cpu().numpy()
    return err.astype(np.float32)  # higher => more suspicious

val_scores = ae_score_np(X_val)
test_scores = ae_score_np(X_test)
thr = compute_threshold(val_scores, VAL_PERCENTILE)
tnr, fpr = benign_metrics(test_scores, thr)
auc_syn = synthetic_auc("Deep-AutoEncoder", ae_score_np, X_val, seed=RANDOM_SEED+5)
lat = time.time() - start
results.append(pack_row("Deep-AutoEncoder", thr, tnr, fpr, auc_syn, test_scores, lat))

# -------------------------
# RESULTS TABLE
# -------------------------
df = pd.DataFrame(results).sort_values("Benign_Recall(TNR)_TEST", ascending=False)
print("\n=== FINAL RESULTS ===")
print(df.to_string(index=False))

out_csv = "ad_layer1_full_benchmark.csv"
df.to_csv(out_csv, index=False)
print(f"\nSaved to {out_csv}")
