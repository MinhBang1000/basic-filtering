import torch
import numpy as np
import pandas as pd
import spacy
import time
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer, util
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm  # Thêm thư viện Progress Bar

# --- 1. SETUP & DATA LOADING ---
print("Đang tải dữ liệu từ Qualifire Benchmark...")
ds = load_dataset("qualifire/prompt-injections-benchmark")
df = pd.DataFrame(ds['test'])
TEXT_COL, LABEL_COL = 'text', 'label'

# --- 2. KHỞI TẠO MÔ HÌNH ---
print("Khởi tạo các mô hình đặc trưng (GPU if available)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
nlp = spacy.load("en_core_web_sm")
ppl_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
ppl_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
embed_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
SENSITIVE_KEYWORDS = ['ignore', 'system', 'developer', 'password', 'override', 'bypass', 'secret', 'admin', 'rules']

def get_features(text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    # A. Perplexity
    inputs = ppl_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        loss = ppl_model(**inputs, labels=inputs["input_ids"]).loss
        ppl = torch.exp(loss).item()
    
    # B. Semantic Divergence (Cải tiến: Max Divergence)
    # Chunk theo token window để payload nằm đâu cũng bắt được
    max_div = 0.0
    text_clean = " ".join(text.split())  # normalize whitespace

    # Tokenize theo tokenizer của SentenceTransformer (không cần spaCy sentence split)
    tokens = text_clean.split()
    if len(tokens) >= 20:  # quá ngắn thì divergence không đáng tin
        window = 40      # số từ / chunk (tùy chỉnh)
        stride = 20      # bước trượt (tùy chỉnh)

        chunks = []
        for start in range(0, len(tokens), stride):
            chunk = " ".join(tokens[start:start + window])
            if len(chunk) >= 10:
                chunks.append(chunk)

        # Nếu chỉ ra được >=2 chunks thì mới tính max divergence
        if len(chunks) >= 2:
            embs = embed_model.encode(chunks, convert_to_tensor=True, show_progress_bar=False)
            mean_emb = embs.mean(dim=0)
            cos_sims = util.cos_sim(embs, mean_emb)  # shape (n_chunks, 1)
            min_sim = cos_sims.min().item()
            max_div = 1.0 - min_sim

            
    # C. Instruction Density
    verbs = [t for t in doc if t.pos_ == "VERB" and t.dep_ == "ROOT"]
    instr_dens = len(verbs) / (len(sentences) + 1e-6)
    
    # D. Keyword Trigger
    keyword_count = sum(1 for word in SENSITIVE_KEYWORDS if word in text.lower())
    
    return [np.log(ppl + 1), max_div, instr_dens, keyword_count]

# --- 3. TRÍCH XUẤT ĐẶC TRƯNG VỚI PROGRESS BAR ---
print(f"\nBắt đầu trích xuất đặc trưng cho {len(df)} mẫu...")
features = []

# Sử dụng tqdm bao bọc vòng lặp để hiện thanh tiến trình
for i in tqdm(range(len(df)), desc="[Layer 1] Feature Extraction", unit="sample"):
    txt = df.iloc[i][TEXT_COL]
    features.append(get_features(txt))

X = np.array(features)
# --- SỬA LỖI LABEL ---
# Chuyển đổi nhãn từ chuỗi sang số: benign -> 0, jailbreak -> 1
label_map = {'benign': 0, 'jailbreak': 1}
y = np.array([label_map[l] for l in df[LABEL_COL]])

print(f"Nhãn sau khi chuyển đổi: {np.unique(y)} (0: Benign, 1: Jailbreak)")

# --- 4. HUẤN LUYỆN & ĐÁNH GIÁ (XGBOOST) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

ratio = float(np.sum(y == 0)) / np.sum(y == 1)
xgb_model = XGBClassifier(
    n_estimators=300, # Tăng số cây cho XGBoost
    max_depth=6,
    learning_rate=0.03,
    scale_pos_weight=ratio,
    eval_metric='logloss',
    random_state=42
)

print("\nĐang huấn luyện XGBoost...")
xgb_model.fit(X_train, y_train)

# Đánh giá
start = time.perf_counter()
y_prob = xgb_model.predict_proba(X_test)[:, 1]
y_pred = xgb_model.predict(X_test)
end = time.perf_counter()
total_time = end - start
avg_time_per_sample = total_time / len(X_test)
print(f"Total inference time: {total_time:.4f} seconds")
print(f"Avg inference time per sample: {avg_time_per_sample*1000:.4f} ms")

print("\n" + "="*40)
print("KẾT QUẢ ĐÁNH GIÁ VỚI XGBOOST NÂNG CẤP")
print("="*40)
print(classification_report(y_test, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# Feature Importance
importances = xgb_model.feature_importances_
f_names = ["Log-PPL", "Max-Semantic-Div", "Instr-Density", "Keyword-Count"]
print("\nFeature Importance (XGBoost):")
for name, imp in sorted(zip(f_names, importances), key=lambda x: x[1], reverse=True):
    print(f"- {name:18}: {imp:.4f}")


# ========================================
# KẾT QUẢ ĐÁNH GIÁ VỚI XGBOOST NÂNG CẤP
# ========================================
#               precision    recall  f1-score   support

#            0       0.84      0.62      0.71       900
#            1       0.59      0.82      0.69       600

#     accuracy                           0.70      1500
#    macro avg       0.72      0.72      0.70      1500
# weighted avg       0.74      0.70      0.70      1500

# ROC AUC Score: 0.8137

# Feature Importance (XGBoost):
# - Keyword-Count     : 0.5032
# - Max-Semantic-Div  : 0.2910
# - Log-PPL           : 0.1053
# - Instr-Density     : 0.1005