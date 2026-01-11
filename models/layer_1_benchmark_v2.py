#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import time
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, confusion_matrix

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset

# -------------------------
# 1. ARGUMENT PARSING
# -------------------------
parser = argparse.ArgumentParser(description="Layer-1 Benchmark: Anomaly Detection on Dolly vs Deepset")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--openai", action="store_true", help="OpenAI text-embedding-3-large")
group.add_argument("--bert", action="store_true", help="BERT-base-nli-mean-tokens")
group.add_argument("--minilm", action="store_true", help="All-MiniLM-L6-v2")
args = parser.parse_args()

# -------------------------
# 2. CONFIG
# -------------------------
RANDOM_SEED = 42
BATCH_SIZE = 64
THRESHOLD_PERCENTILE = 95 # Threshold được chọn dựa trên bách phân vị thứ 95 của tập Validation
TRAIN_FILE = "../datasets/processed_datasets/train_corporate_dolly_v2.jsonl"
TEST_FILE = "../datasets/processed_datasets/test_corporate_dolly_v2.jsonl"

if args.openai:
    EMBED_TYPE, MODEL_NAME, DIM = "openai", "text-embedding-3-large", 3072
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
elif args.bert:
    EMBED_TYPE, MODEL_NAME, DIM = "local", "bert-base-nli-mean-tokens", 768
elif args.minilm:
    EMBED_TYPE, MODEL_NAME, DIM = "local", "all-MiniLM-L6-v2", 384

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
# 4. AUTOENCODER MODEL
# -------------------------
class DynamicAE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        h1, h2 = (512, 128) if input_dim > 512 else (input_dim // 2, input_dim // 4)
        self.encoder = nn.Sequential(nn.Linear(input_dim, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(h2, h1), nn.ReLU(), nn.Linear(h1, input_dim))
    def forward(self, x): return self.decoder(self.encoder(x))

# -------------------------
# 5. UTILS: HÀM TÍNH TOÁN METRICS TẬP TRUNG
# -------------------------
def get_performance_metrics(y_true, y_pred, y_scores, name, start_time):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # TPR
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auc = roc_auc_score(y_true, y_scores)

    return {
        "Model": name,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "TPR (Recall)": recall,
        "FPR (Fall-out)": fpr,
        "Precision": precision,
        "F1-Score": f1,
        "TNR (Specificity)": tnr,
        "AUC": auc,
        "Latency": time.time() - start_time
    }


# -------------------------
# 6. EXECUTION & EVALUATION
# -------------------------
def run_benchmark():
    print(f"\n[1/4] Loading Datasets...")
    # Load từ file bạn vừa tạo
    train_ds = load_dataset("json", data_files=TRAIN_FILE)["train"]
    test_ds = load_dataset("json", data_files=TEST_FILE)["train"]

    X_train_raw = get_embeddings(train_ds["instruction"])
    X_test_raw = get_embeddings(test_ds["instruction"])
    y_test = np.array(test_ds["category"]) # 0 = Benign, 1 = Suspicious

    # Scale dữ liệu
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # Chia tập train thành Train-fit và Validation để tìm threshold
    # Layer 1 học trên dữ liệu Benign hoàn toàn
    split_idx = int(len(X_train) * 0.8)
    X_train_fit = X_train[:split_idx]
    X_val = X_train[split_idx:]

    print(f"[2/4] Training Anomaly Detectors...")
    results = []
    
    models = {
        "IsolationForest": lambda: IsolationForest(contamination=0.05, random_state=RANDOM_SEED).fit(X_train_fit),
        "OneClassSVM": lambda: OneClassSVM(nu=0.05, kernel="rbf").fit(X_train_fit),
        "kNN-Distance": lambda: NearestNeighbors(n_neighbors=5).fit(X_train_fit),
        "PCA-Reconstruction": lambda: PCA(n_components=0.9, random_state=RANDOM_SEED).fit(X_train_fit)
    }

    # Chạy các mô hình ML truyền thống
    for name, init_fn in models.items():
        print(f"  > Processing {name}...")
        start_t = time.time()
        m = init_fn()
        
        if name in ["IsolationForest", "OneClassSVM"]: score_fn = lambda x: -m.score_samples(x)
        elif name == "kNN-Distance": score_fn = lambda x: m.kneighbors(x)[0].mean(axis=1)
        else: score_fn = lambda x: np.linalg.norm(x - m.inverse_transform(m.transform(x)), axis=1)

        val_scores = score_fn(X_val)
        threshold = np.percentile(val_scores, THRESHOLD_PERCENTILE)
        test_scores = score_fn(X_test)
        y_pred = (test_scores >= threshold).astype(int)

        results.append(get_performance_metrics(y_test, y_pred, test_scores, name, start_t))

    # Chạy Deep AutoEncoder
    print("  > Processing Deep-AutoEncoder...")
    start_t = time.time()
    ae = DynamicAE(DIM).to("cuda" if torch.cuda.is_available() else "cpu")
    opt, crit = optim.Adam(ae.parameters(), lr=1e-3), nn.MSELoss()
    X_train_tensor = torch.tensor(X_train_fit, device=ae.encoder[0].weight.device)
    
    for _ in range(30):
        opt.zero_grad()
        crit(ae(X_train_tensor), X_train_tensor).backward()
        opt.step()
    
    ae.eval()
    def ae_score(x):
        with torch.no_grad():
            xt = torch.tensor(x, device=ae.encoder[0].weight.device)
            return ((ae(xt) - xt)**2).mean(dim=1).cpu().numpy()

    val_scores_ae = ae_score(X_val)
    threshold_ae = np.percentile(val_scores_ae, THRESHOLD_PERCENTILE)
    test_scores_ae = ae_score(X_test)
    y_pred_ae = (test_scores_ae >= threshold_ae).astype(int)

    # Đã sửa: Truyền đúng tên "Deep-AutoEncoder"
    results.append(get_performance_metrics(y_test, y_pred_ae, test_scores_ae, "Deep-AutoEncoder", start_t))

    print("\n" + "="*85)
    print(f"LAYER 1 BENCHMARK RESULTS ({MODEL_NAME})")
    print("="*85)
    df = pd.DataFrame(results).sort_values("AUC", ascending=False)
    # Format lại bảng để dễ đọc hơn trong báo cáo
    print(df.to_string(index=False, float_format=lambda x: "{:.4f}".format(x) if isinstance(x, float) else str(x)))

if __name__ == "__main__":
    run_benchmark()