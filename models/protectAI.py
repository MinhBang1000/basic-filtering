import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from tqdm import tqdm

# --- 1. SETUP & DATA LOADING ---
print("Đang tải dữ liệu Qualifire Benchmark...")
ds = load_dataset("qualifire/prompt-injections-benchmark")
df = pd.DataFrame(ds['test'])
TEXT_COL, LABEL_COL = 'text', 'label'

# Map nhãn sang số (benign: 0, jailbreak: 1)
label_map = {'benign': 0, 'jailbreak': 1}
df['label_num'] = df[LABEL_COL].map(label_map)

# --- 2. CHIA TẬP TEST GIỐNG HỆT XGBOOST ---
# Chúng ta dùng random_state=42 và test_size=0.3 để lấy đúng những hàng (rows) đó
df_train, df_test = train_test_split(
    df, 
    test_size=0.3, 
    random_state=42, 
    stratify=df['label_num']
)

print(f"Số lượng mẫu trong tập đối soát: {len(df_test)}")

# --- 3. KHỞI TẠO PROTECT AI ---
print("Đang tải mô hình ProtectAI (DeBERTa-v3)...")
model_name = "protectai/deberta-v3-base-prompt-injection-v2"
device = 0 if torch.cuda.is_available() else -1

classifier = pipeline(
    "text-classification", 
    model=model_name, 
    tokenizer=model_name, 
    device=device,
    truncation=True,
    max_length=512
)

# --- 4. INFERENCE TRÊN TẬP TEST ---
y_true = df_test['label_num'].values
y_pred_labels = []
y_probs = []

for txt in tqdm(df_test[TEXT_COL], desc="ProtectAI Evaluating Test Set"):
    result = classifier(txt)[0]
    
    # ProtectAI v2: LABEL_0 thường là Safe, LABEL_1 là Injection
    # Hoặc 'SAFE'/'INJECTION' tùy theo config. Ta sẽ check nhãn trả về:
    if result['label'] in ['INJECTION', 'LABEL_1']:
        pred_label = 1
        prob = result['score']
    else:
        pred_label = 0
        prob = 1 - result['score']
        
    y_pred_labels.append(pred_label)
    y_probs.append(prob)

# --- 5. KẾT QUẢ ĐỐI SOÁT CUỐI CÙNG ---
print("\n" + "="*45)
print("KẾT QUẢ PROTECT AI (TRÊN CÙNG TẬP TEST VỚI XGBOOST)")
print("="*45)
print(classification_report(y_true, y_pred_labels, target_names=['benign', 'jailbreak']))
print(f"ROC AUC Score: {roc_auc_score(y_true, y_probs):.4f}")


# =============================================
# KẾT QUẢ PROTECT AI (TRÊN CÙNG TẬP TEST VỚI XGBOOST)
# =============================================
#               precision    recall  f1-score   support

#       benign       0.78      0.76      0.77       900
#    jailbreak       0.65      0.67      0.66       600

#     accuracy                           0.73      1500
#    macro avg       0.72      0.72      0.72      1500
# weighted avg       0.73      0.73      0.73      1500

# ROC AUC Score: 0.8318