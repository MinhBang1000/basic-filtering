import torch
import numpy as np
import pandas as pd
import spacy
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
    # A. Perplexity
    inputs = ppl_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        loss = ppl_model(**inputs, labels=inputs["input_ids"]).loss
        ppl = torch.exp(loss).item()
    
    # B. Semantic Divergence (Cải tiến: Max Divergence)
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    max_div = 0.0
    if len(sentences) >= 2:
        embs = embed_model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
        # Tính toán vector trung bình của context
        mean_emb = embs.mean(dim=0)
        # So sánh độ lệch của từng câu
        cos_sims = util.cos_sim(embs, mean_emb)
        max_div = (1.0 - cos_sims.min()).item() # Lấy độ lệch lớn nhất (1 - similarity nhỏ nhất)
            
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
y_prob = xgb_model.predict_proba(X_test)[:, 1]
y_pred = xgb_model.predict(X_test)

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

#            0       0.83      0.63      0.72       900
#            1       0.59      0.81      0.68       600

#     accuracy                           0.70      1500
#    macro avg       0.71      0.72      0.70      1500
# weighted avg       0.74      0.70      0.70      1500

# ROC AUC Score: 0.8030

# Feature Importance (XGBoost):
# - Keyword-Count     : 0.4290
# - Max-Semantic-Div  : 0.3145
# - Instr-Density     : 0.1495
# - Log-PPL           : 0.1070