import os
import time
import torch
import pandas as pd
import torch.nn.functional as F # Dùng để tính Softmax lấy xác suất
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 1. CẤU HÌNH HỆ THỐNG
# ============================================================
TEST_FILE = "../datasets/processed_datasets/test_corporate_dolly_v2.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LIMIT_SAMPLES = int(os.getenv("LIMIT_SAMPLES", None))  # Đặt thành None để dùng toàn bộ dữ liệu

SOTA_MODELS = {
    "ProtectAI-v1": "protectai/deberta-v3-base-prompt-injection",
    "ProtectAI-v2": "protectai/deberta-v3-base-prompt-injection-v2",
    "Injection-Sentinel": "qualifire/prompt-injection-sentinel",
    "Mdeberta":"proventra/mdeberta-v3-base-prompt-injection",
    "aibastion": "neeraj-kumar-47/aibastion-prompt-injection-jailbreak-detector"
}

# ============================================================
# 2. HÀM INFERENCE
# ============================================================
def run_sota_benchmark():
    print(f"[*] Loading test dataset: {TEST_FILE}")
    dataset = load_dataset("json", data_files=TEST_FILE)["train"]
    if LIMIT_SAMPLES:
        dataset = dataset.shuffle(seed=42).select(range(LIMIT_SAMPLES))
    dataset = dataset.shuffle(seed=42)
    texts = dataset["instruction"]
    y_true = dataset["label"]

    final_results = []

    for name, model_path in SOTA_MODELS.items():
        print(f"\n[+] Testing Model: {name} ({model_path})")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE)
            model.eval()

            y_pred = []
            y_probs = [] # Danh sách xác suất để tính AUC
            latencies = []

            for text in tqdm(texts, desc=f"Inference {name}"):
                start_t = time.perf_counter()
                
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
                with torch.no_grad():
                    outputs = model(**inputs)
                    # Chuyển Logits thành Xác suất bằng Softmax
                    probs = F.softmax(outputs.logits, dim=-1)
                    
                    # Lấy nhãn dự đoán (0 hoặc 1)
                    prediction = torch.argmax(outputs.logits, dim=-1).item()
                    # Lấy xác suất của class 1 (Malicious)
                    prob_malicious = probs[0][1].item()
                
                latencies.append((time.perf_counter() - start_t) * 1000)
                y_pred.append(prediction)
                y_probs.append(prob_malicious)

            del model
            torch.cuda.empty_cache()

            # --- TÍNH TOÁN METRICS ---
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            
            final_results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_true, y_pred),
                "AUC": roc_auc_score(y_true, y_probs), # Tính AUC từ xác suất
                "TPR (Recall)": recall_score(y_true, y_pred),
                "FPR": fp / (fp + tn) if (fp + tn) > 0 else 0,
                "F1-Score": f1_score(y_true, y_pred, zero_division=0),
                "Avg_Lat_ms": sum(latencies) / len(latencies)
            })

        except Exception as e:
            print(f"[!] Error testing {name}: {e}")

    # ============================================================
    # 3. HIỂN THỊ KẾT QUẢ
    # ============================================================
    report_df = pd.DataFrame(final_results)
    report_df = report_df.sort_values(by="AUC", ascending=False) # Sắp xếp theo AUC
    
    print("\n" + "="*110)
    print("SOTA CLASSIFIERS COMPARISON REPORT (ACC & AUC)")
    print("="*110)
    print(report_df.to_string(index=False, float_format=lambda x: "{:.4f}".format(x) if isinstance(x, float) else str(x)))
    
    report_df.to_csv("sota_comparison_results_full.csv", index=False)
    return report_df

if __name__ == "__main__":
    run_sota_benchmark()