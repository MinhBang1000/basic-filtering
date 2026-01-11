import os
import time
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
# Thêm accuracy_score và roc_auc_score vào import
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from datasets import load_dataset
from dotenv import load_dotenv

# Giả sử import từ các file của bạn
from layer_1 import Layer1AnomalyDetector
from layer_3 import Layer3Auditor

load_dotenv()

# ============================================================
# 1. CẤU HÌNH & KHỞI TẠO
# ============================================================
TEST_FILE = "../datasets/processed_datasets/test_corporate_dolly_v2.jsonl"
L1_MODEL_DIR = "./saved_models/layer1_horse"
L3_MODEL_NAME = "gpt-4o-mini"
LIMIT_SAMPLES = int(os.getenv("LIMIT_SAMPLES", None))  # Đặt thành None để dùng toàn bộ dữ liệu

def run_integrated_benchmark():
    l1_detector = Layer1AnomalyDetector(model_dir=L1_MODEL_DIR)
    l3_auditor = Layer3Auditor(model=L3_MODEL_NAME)

    print(f"[*] Loading test dataset: {TEST_FILE}")
    dataset = load_dataset("json", data_files=TEST_FILE)["train"]
    if LIMIT_SAMPLES:
        dataset = dataset.shuffle(seed=42).select(range(LIMIT_SAMPLES))
    
    dataset = dataset.shuffle(seed=42)
    results = []

    print(f"[*] Starting Evaluation on {len(dataset)} samples...")
    
    for item in tqdm(dataset):
        text = item["instruction"]
        gt_category = item["category"]  
        gt_label = item["label"]        
        
        start_time = time.perf_counter()
        
        # --- LAYER 1 ---
        l1_start = time.perf_counter()
        l1_output = l1_detector.detect(text)[0] # Trả về {"is_anomaly": bool, "score": float}
        l1_latency = (time.perf_counter() - l1_start) * 1000
        
        l3_latency = 0
        triggered_l3 = False
        final_decision = False
        
        # --- INTEGRATED LOGIC ---
        if l1_output["is_anomaly"]:
            triggered_l3 = True
            l3_start = time.perf_counter()
            l3_output = l3_auditor.check_alignment(text)
            l3_latency = (time.perf_counter() - l3_start) * 1000
            final_decision = l3_output["malicious"]
        else:
            final_decision = False
            
        total_latency = (time.perf_counter() - start_time) * 1000
        
        results.append({
            "gt_l1": gt_category,
            "pred_l1": 1 if l1_output["is_anomaly"] else 0,
            "l1_score": l1_output["score"], # Lưu lại score để tính AUC cho L1
            "gt_system": gt_label,
            "pred_system": 1 if final_decision else 0,
            "l1_latency_ms": l1_latency,
            "l3_latency_ms": l3_latency,
            "total_latency_ms": total_latency,
            "triggered_l3": triggered_l3
        })

    df = pd.DataFrame(results)

    # ============================================================
    # 2. TÍNH TOÁN METRICS BỔ SUNG
    # ============================================================
    
    # Metrics cho Layer 1 (so với category)
    tn1, fp1, fn1, tp1 = confusion_matrix(df["gt_l1"], df["pred_l1"], labels=[0, 1]).ravel()
    acc1 = accuracy_score(df["gt_l1"], df["pred_l1"])

    # Metrics cho Toàn hệ thống (so với label)
    tnS, fpS, fnS, tpS = confusion_matrix(df["gt_system"], df["pred_system"], labels=[0, 1]).ravel()
    accS = accuracy_score(df["gt_system"], df["pred_system"])

    print("\n" + "="*85)
    print(f"INTEGRATED DEFENSE EVALUATION: L1 + L3")
    print("="*85)

    print(f"\n[LAYER 1 - ANOMALY DETECTION (vs Category)]")
    print(f"  - Confusion Matrix: TN={tn1}, FP={fp1}, FN={fn1}, TP={tp1}")
    print(f"  - ACC (Accuracy):    {acc1:.4f}")
    print(f"  - TPR (Recall):      {tp1/(tp1+fn1):.4f}")
    print(f"  - FPR:               {fp1/(fp1+tn1):.4f}")

    print(f"\n[TOTAL SYSTEM PERFORMANCE (vs Label)]")
    print(f"  - Confusion Matrix: TN={tnS}, FP={fpS}, FN={fnS}, TP={tpS}")
    print(f"  - ACC (Accuracy):    {accS:.4f}")
    print(f"  - TPR (Recall):      {tpS/(tpS+fnS):.4f}")
    print(f"  - F1-Score:          {f1_score(df['gt_system'], df['pred_system']):.4f}")
    print(f"  - FPR:               {fpS/(fpS+tnS):.4f}")

    print(f"\n[EFFICIENCY & LATENCY]")
    print(f"  - Avg Total Latency: {df['total_latency_ms'].mean():.2f} ms")
    print(f"  - L3 Trigger Rate:   {df['triggered_l3'].mean()*100:.2f}%")
    print("="*85)

    df.to_csv("integrated_system_extended_results.csv", index=False)

if __name__ == "__main__":
    run_integrated_benchmark()