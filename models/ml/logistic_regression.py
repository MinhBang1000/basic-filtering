import torch
import numpy as np
import pandas as pd
import spacy
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer, util
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# --- 1. SETUP & DATA LOADING ---
print("Đang tải dữ liệu từ Hugging Face...")
# Dataset này yêu cầu login nếu là private, nhưng thường benchmark này là public
ds = load_dataset("qualifire/prompt-injections-benchmark")
df = pd.DataFrame(ds['test'])

# Giả sử cột text là 'text' và nhãn là 'label'. 
# (Nếu tên cột khác, bạn hãy điều chỉnh lại tại đây)
TEXT_COL = 'text' 
LABEL_COL = 'label'

# Tối ưu: Lấy mẫu nhỏ để test code nhanh nếu cần (ví dụ 200 mẫu)
# df = df.sample(200).reset_index(drop=True)

# --- 2. KHỞI TẠO CÁC MÔ HÌNH TRÍCH XUẤT ---
print("Khởi tạo các mô hình đặc trưng...")
nlp = spacy.load("en_core_web_sm")
ppl_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
ppl_model = GPT2LMHeadModel.from_pretrained("gpt2").to("cuda" if torch.cuda.is_available() else "cpu")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_ppl(text):
    inputs = ppl_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(ppl_model.device)
    with torch.no_grad():
        loss = ppl_model(**inputs, labels=inputs["input_ids"]).loss
    return torch.exp(loss).item()

def get_semantic_div(text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    if len(sentences) < 2: return 0.0
    embs = embed_model.encode(sentences, convert_to_tensor=True)
    context_emb = embs[:-1].mean(dim=0)
    target_emb = embs[-1]
    return (1.0 - util.cos_sim(context_emb, target_emb).item())

def get_instr_density(text):
    doc = nlp(text)
    # Đếm các động từ dạng Root trong các câu (đặc trưng của mệnh lệnh)
    verbs = [t for t in doc if t.pos_ == "VERB" and t.dep_ == "ROOT"]
    return len(verbs) / (len(list(doc.sents)) + 1e-6)

# --- 3. TRÍCH XUẤT ĐẶC TRƯNG (FEATURE ENGINEERING) ---
print("Bắt đầu trích xuất đặc trưng (có thể mất vài phút)...")
features = []
for i, row in df.iterrows():
    if i % 50 == 0: print(f"Đã xử lý: {i}/{len(df)}")
    txt = row[TEXT_COL]
    features.append([
        np.log(get_ppl(txt) + 1), # Log-scale Perplexity
        get_semantic_div(txt),    # Semantic Divergence
        get_instr_density(txt)    # Instruction Density
    ])

X = np.array(features)
y = df[LABEL_COL].values

# --- 4. TRAIN THRESHOLD & WEIGHTS (ENSEMBLE) ---
# Chia split 70/30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Chuẩn hóa dữ liệu (Rất quan trọng cho Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Huấn luyện mô hình Ensemble quyết định
ensemble = LogisticRegression(class_weight='balanced')
ensemble.fit(X_train_scaled, y_train)

# --- 5. ĐÁNH GIÁ KẾT QUẢ ---
y_pred = ensemble.predict(X_test_scaled)
y_prob = ensemble.predict_proba(X_test_scaled)[:, 1]

print("\n" + "="*30)
print("KẾT QUẢ ĐÁNH GIÁ TRÊN QUALIFIRE BENCHMARK")
print("="*30)
print(classification_report(y_test, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# Trích xuất tầm quan trọng của các đặc trưng
weights = ensemble.coef_[0]
features_names = ["Log-PPL", "Semantic-Div", "Instr-Density"]
importance_df = pd.DataFrame({'Feature': features_names, 'Weight': weights})
print("\nTầm quan trọng của các đặc trưng (Weights):")
print(importance_df.sort_values(by='Weight', ascending=False))