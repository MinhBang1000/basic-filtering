import torch
import numpy as np
import pandas as pd
import spacy
import time
from tqdm import tqdm
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, classification_report

# --- 1. SETUP & DATA ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {DEVICE}")

ds = load_dataset("qualifire/prompt-injections-benchmark")
df = pd.DataFrame(ds['test'])
label_map = {'benign': 0, 'jailbreak': 1}
df['label_num'] = df['label'].map(label_map)

# --- 2. MODELS INITIALIZATION ---
nlp = spacy.load("en_core_web_sm")
ppl_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
ppl_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
embed_model = SentenceTransformer('all-MiniLM-L6-v2').to(DEVICE)
SENSITIVE_KEYWORDS = ['ignore', 'system', 'developer', 'password', 'override', 'bypass', 'secret', 'admin', 'rules']

def get_custom_features(text):
    doc = nlp(text)
    # A. Perplexity
    inputs = ppl_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        loss = ppl_model(**inputs, labels=inputs["input_ids"]).loss
        ppl = torch.exp(loss).item()
    
    # B. Max Semantic Divergence (Sliding Window)
    max_div = 0.0
    tokens = text.split()
    if len(tokens) >= 20:
        window, stride = 40, 20
        chunks = [" ".join(tokens[i:i+window]) for i in range(0, len(tokens), stride) if len(tokens[i:i+window]) > 5]
        if len(chunks) >= 2:
            embs = embed_model.encode(chunks, convert_to_tensor=True, show_progress_bar=False)
            cos_sims = util.cos_sim(embs, embs.mean(dim=0))
            max_div = 1.0 - cos_sims.min().item()
            
    # C. Instruction Density & D. Keywords
    instr_dens = len([t for t in doc if t.pos_ == "VERB" and t.dep_ == "ROOT"]) / (len(list(doc.sents)) + 1e-6)
    keyword_count = sum(1 for word in SENSITIVE_KEYWORDS if word in text.lower())
    return [np.log(ppl + 1), max_div, instr_dens, keyword_count]

# --- 3. BENCHMARKING ENGINE ---
def run_hf_benchmark(model_id, test_texts):
    print(f"\nEvaluating: {model_id}")
    # Pipe hỗ trợ truncation và device tự động
    pipe = pipeline("text-classification", model=model_id, device=DEVICE, truncation=True, max_length=512)
    
    probs, preds = [], []
    start_time = time.time()
    for txt in tqdm(test_texts, desc=f"Scanning {model_id.split('/')[-1]}"):
        res = pipe(txt)[0]
        # Logic map nhãn linh hoạt cho nhiều model khác nhau
        is_inj = res['label'] in ['INJECTION', 'LABEL_1', 'jailbreak', 'injection', 'attacker']
        prob = res['score'] if is_inj else 1 - res['score']
        probs.append(prob)
        preds.append(1 if prob > 0.5 else 0)
    
    latency = (time.time() - start_time) / len(test_texts) * 1000
    return probs, preds, latency

# --- 4. EXECUTION ---
# Bắt đầu trích xuất đặc trưng XGBoost
print(f"Trích xuất đặc trưng cho {len(df)} mẫu...")
X = np.array([get_custom_features(t) for t in tqdm(df['text'], desc="Feature Extraction")])
y = df['label_num'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
indices_test = train_test_split(range(len(df)), test_size=0.3, random_state=42, stratify=y)[1]
df_test = df.iloc[indices_test]

# Huấn luyện XGBoost
xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.03, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)

# Danh sách 5 mô hình HF
hf_models = [
    "protectai/deberta-v3-base-prompt-injection-v2",
    "deepset/deberta-v3-base-injection",
    "fmops/distilbert-prompt-injection"
]

results_log = []

# Eval XGBoost
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
results_log.append({"Model": "Your Custom XGBoost", "AUC": roc_auc_score(y_test, y_prob_xgb), "Latency (ms)": "N/A (Fast)"})

# Eval HF Models
for mid in hf_models:
    probs, preds, lat = run_hf_benchmark(mid, df_test['text'].tolist())
    results_log.append({
        "Model": mid.split("/")[-1],
        "AUC": roc_auc_score(y_test, probs),
        "Latency (ms)": f"{lat:.2f}"
    })

# --- 5. FINAL LOGGING ---
report = pd.DataFrame(results_log)
print("\n" + "="*60)
print("FINAL BENCHMARK SUMMARY (QUALIFIRE TEST SET)")
print("="*60)
print(report.sort_values(by="AUC", ascending=False).to_string(index=False))