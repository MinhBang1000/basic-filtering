import torch
import numpy as np
import pandas as pd
import spacy
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer, util
from sklearn.ensemble import RandomForestClassifier  # Thay đổi ở đây
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

# --- 1. SETUP & DATA LOADING ---
print("Đang tải dữ liệu từ Qualifire Benchmark...")
ds = load_dataset("qualifire/prompt-injections-benchmark")
df = pd.DataFrame(ds['test'])

TEXT_COL = 'text' 
LABEL_COL = 'label'

# --- 2. MÔ HÌNH TRÍCH XUẤT ---
nlp = spacy.load("en_core_web_sm")
ppl_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
ppl_model = GPT2LMHeadModel.from_pretrained("gpt2").to("cuda" if torch.cuda.is_available() else "cpu")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Danh sách từ khóa nhạy cảm thường thấy trong Prompt Injection
SENSITIVE_KEYWORDS = ['ignore', 'system', 'developer', 'password', 'override', 'bypass', 'secret', 'admin', 'rules']

def get_features(text):
    # A. Perplexity
    inputs = ppl_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(ppl_model.device)
    with torch.no_grad():
        loss = ppl_model(**inputs, labels=inputs["input_ids"]).loss
        ppl = torch.exp(loss).item()
    
    # B. Semantic Divergence (Cải tiến: Lấy Max Divergence giữa các câu)
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    max_div = 0.0
    if len(sentences) >= 2:
        embs = embed_model.encode(sentences, convert_to_tensor=True)
        for i in range(len(sentences)):
            # So sánh từng câu với trung bình cộng của toàn bộ văn bản
            sim = util.cos_sim(embs[i], embs.mean(dim=0)).item()
            max_div = max(max_div, 1.0 - sim)
            
    # C. Instruction Density
    verbs = [t for t in doc if t.pos_ == "VERB" and t.dep_ == "ROOT"]
    instr_dens = len(verbs) / (len(sentences) + 1e-6)
    
    # D. Keyword Trigger (Mới)
    keyword_count = sum(1 for word in SENSITIVE_KEYWORDS if word in text.lower())
    
    return [np.log(ppl + 1), max_div, instr_dens, keyword_count]

# --- 3. TRÍCH XUẤT ĐẶC TRƯNG ---
print("Bắt đầu trích xuất đặc trưng nâng cao...")
X = np.array([get_features(txt) for txt in df[TEXT_COL]])
y = df[LABEL_COL].values

# --- 4. TRAIN ENSEMBLE (RANDOM FOREST) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Random Forest không bắt buộc StandardScaler nhưng nên dùng để ổn định
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Sử dụng Random Forest để bắt các quan hệ phi tuyến
model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)

# --- 5. ĐÁNH GIÁ ---
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("\n" + "="*30)
print("KẾT QUẢ ĐÁNH GIÁ NÂNG CẤP (RANDOM FOREST)")
print("="*30)
print(classification_report(y_test, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# Kiểm tra độ quan trọng
importances = model.feature_importances_
f_names = ["Log-PPL", "Max-Semantic-Div", "Instr-Density", "Keyword-Count"]
for name, imp in zip(f_names, importances):
    print(f"Feature {name}: {imp:.4f}")



# ==============================
# KẾT QUẢ ĐÁNH GIÁ NÂNG CẤP (RANDOM FOREST)
# ==============================
#               precision    recall  f1-score   support

#       benign       0.83      0.60      0.70       900
#    jailbreak       0.58      0.82      0.68       600

#     accuracy                           0.69      1500
#    macro avg       0.71      0.71      0.69      1500
# weighted avg       0.73      0.69      0.69      1500

# ROC AUC Score: 0.7957
# Feature Log-PPL: 0.2448
# Feature Max-Semantic-Div: 0.4285
# Feature Instr-Density: 0.1813
# Feature Keyword-Count: 0.1455