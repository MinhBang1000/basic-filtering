#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import time
import math
import json
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
parser = argparse.ArgumentParser(description="Layer-1 NLP Anomaly Detection Benchmark (Multi-Embedding + Suspicious Data)")
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
MAX_SAMPLES = None  # Giới hạn để tiết kiệm tài nguyên
VAL_PERCENTILE = 95
DOLLY_FILE = "../datasets/corporate_dolly_v2.jsonl"
JSONL_FILE = "../datasets/deepset_prompt_injections_refactored.jsonl"

# AE Training Config
AE_EPOCHS = 20
AE_LR = 1e-3
AE_BATCH = 512

if args.openai:
    EMBED_TYPE, MODEL_NAME, DIM = "openai", "text-embedding-3-large", 3072
    if not os.getenv("OPENAI_API_KEY"): raise RuntimeError("OPENAI_API_KEY missing.")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
elif args.bert:
    EMBED_TYPE, MODEL_NAME, DIM = "local", "bert-base-nli-mean-tokens", 768
elif args.minilm:
    EMBED_TYPE, MODEL_NAME, DIM = "local", "all-MiniLM-L6-v2", 384

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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(MODEL_NAME, device=device)
        return model.encode(text_list, batch_size=BATCH_SIZE, show_progress_bar=True).astype(np.float32)

# -------------------------
# 4. METRICS & UTILS
# -------------------------
def compute_metrics(score_fn, X_val, X_test, X_suspicious, threshold_p):
    val_scores = score_fn(X_val)
    thr = np.percentile(val_scores, threshold_p)
    
    test_scores = score_fn(X_test)
    susp_scores = score_fn(X_suspicious)
    
    tnr = (test_scores < thr).mean() # Benign Recall
    detection_rate = (susp_scores >= thr).mean() # Khả năng bắt dữ liệu "lạ"
    
    # AUC giữa Dolly Test (0) và Suspicious Data (1)
    y_true = np.concatenate([np.zeros(len(test_scores)), np.ones(len(susp_scores))])
    y_scores = np.concatenate([test_scores, susp_scores])
    auc = roc_auc_score(y_true, y_scores)
    
    return thr, tnr, detection_rate, auc

class DynamicAE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        h1, h2 = (512, 128) if input_dim > 512 else (input_dim // 2, input_dim // 4)
        self.encoder = nn.Sequential(nn.Linear(input_dim, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(h2, h1), nn.ReLU(), nn.Linear(h1, input_dim))
    def forward(self, x): return self.decoder(self.encoder(x))

# -------------------------
# 5. DATA LOADING
# -------------------------
print(f"\n[1/5] Loading Dolly 15K (Benign)...")
ds = load_dataset("json", data_files=DOLLY_FILE)["train"]
texts_benign = list(ds["instruction"])[:MAX_SAMPLES]

print(f"[2/5] Loading Suspicious Data from {JSONL_FILE}...")
texts_suspicious = []
if not os.path.exists(JSONL_FILE): raise FileNotFoundError(f"File {JSONL_FILE} not found!")
with open(JSONL_FILE, 'r') as f:
    for line in f: texts_suspicious.append(json.loads(line)["text"])

print(f"[3/5] Generating Embeddings...")
X_benign = get_embeddings(texts_benign)
X_suspicious_raw = get_embeddings(texts_suspicious)

X_train, X_tmp = train_test_split(X_benign, test_size=0.4, random_state=RANDOM_SEED)
X_val, X_test = train_test_split(X_tmp, test_size=0.5, random_state=RANDOM_SEED)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val = scaler.transform(X_val).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)
X_suspicious = scaler.transform(X_suspicious_raw).astype(np.float32)

# -------------------------
# 6. EXECUTION
# -------------------------
print("[4/5] Training Models...")
results = []
models_to_run = {
    "IsolationForest": lambda: IsolationForest(contamination=0.05, random_state=RANDOM_SEED).fit(X_train),
    "OneClassSVM": lambda: OneClassSVM(nu=0.05, gamma="scale").fit(X_train),
    "kNN-Distance": lambda: NearestNeighbors(n_neighbors=5).fit(X_train),
    "PCA-Reconstruction": lambda: PCA(n_components=0.95, random_state=RANDOM_SEED).fit(X_train)
}

for name, init_fn in models_to_run.items():
    print(f"  > {name}...")
    start_t = time.time()
    m = init_fn()
    if name in ["IsolationForest", "OneClassSVM"]: score_fn = lambda x: -m.score_samples(x)
    elif name == "kNN-Distance": score_fn = lambda x: m.kneighbors(x)[0].mean(axis=1)
    else: score_fn = lambda x: np.linalg.norm(x - m.inverse_transform(m.transform(x)), axis=1)
    
    thr, tnr, dr, auc = compute_metrics(score_fn, X_val, X_test, X_suspicious, VAL_PERCENTILE)
    results.append({"Model": name, "TNR": tnr, "Suspicious_Det_Rate": dr, "AUC": auc, "Lat_sec": time.time()-start_t})

print("  > Deep AutoEncoder...")
start_t = time.time()
ae = DynamicAE(DIM).to("cuda" if torch.cuda.is_available() else "cpu")
opt, crit = optim.Adam(ae.parameters(), lr=AE_LR), nn.MSELoss()
for _ in range(AE_EPOCHS):
    indices = np.random.permutation(len(X_train))
    for i in range(0, len(X_train), AE_BATCH):
        batch = torch.tensor(X_train[indices[i:i+AE_BATCH]], device=ae.encoder[0].weight.device)
        opt.zero_grad(); crit(ae(batch), batch).backward(); opt.step()

ae.eval()
def ae_score(x):
    with torch.no_grad():
        xt = torch.tensor(x, device=ae.encoder[0].weight.device)
        return ((ae(xt) - xt)**2).mean(dim=1).cpu().numpy()

thr, tnr, dr, auc = compute_metrics(ae_score, X_val, X_test, X_suspicious, VAL_PERCENTILE)
results.append({"Model": "Deep-AutoEncoder", "TNR": tnr, "Suspicious_Det_Rate": dr, "AUC": auc, "Lat_sec": time.time()-start_t})

# -------------------------
# 7. RESULTS
# -------------------------
final_df = pd.DataFrame(results).sort_values("AUC", ascending=False)
print("\n" + "="*60)
print(f"RESULTS FOR {MODEL_NAME} (Dolly vs. Suspicious)")
print("="*60)
print(final_df.to_string(index=False))
final_df.to_csv(f"benchmark_suspicious_{MODEL_NAME.replace('/', '_')}.csv", index=False)