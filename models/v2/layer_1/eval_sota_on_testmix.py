import argparse
import json
import os
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report


SOTA_MODELS = {
    "ProtectAI-v1": "protectai/deberta-v3-base-prompt-injection",
    "ProtectAI-v2": "protectai/deberta-v3-base-prompt-injection-v2",
    "Injection-Sentinel": "qualifire/prompt-injection-sentinel",
    "Mdeberta": "proventra/mdeberta-v3-base-prompt-injection",
    "aibastion": "neeraj-kumar-47/aibastion-prompt-injection-jailbreak-detector",
}


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_csv(path: str, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if text_col not in df.columns:
        raise ValueError(f"{path} missing text_col='{text_col}'. Found: {list(df.columns)[:30]}")
    if label_col not in df.columns:
        raise ValueError(f"{path} missing label_col='{label_col}'. Found: {list(df.columns)[:30]}")
    df[text_col] = df[text_col].astype(str)
    df[label_col] = df[label_col].astype(int)
    return df


def infer_positive_index(model) -> int:
    """
    Try to find which class index means "malicious/injection/jailbreak".
    If cannot infer, default to index 1.
    """
    cfg = model.config
    id2label = getattr(cfg, "id2label", None) or {}
    # normalize
    mapping = {int(k): str(v).lower() for k, v in id2label.items()} if isinstance(id2label, dict) else {}

    keywords_pos = [
        "mal", "attack", "inject", "jailbreak", "unsafe", "harmful", "prompt_injection",
        "pi", "adversarial", "bad", "blocked"
    ]
    keywords_neg = ["benign", "safe", "normal", "clean", "ok"]

    if mapping:
        # If binary labels named, pick the one that looks like malicious.
        best = None
        for idx, name in mapping.items():
            score = 0
            for kw in keywords_pos:
                if kw in name:
                    score += 2
            for kw in keywords_neg:
                if kw in name:
                    score -= 2
            if best is None or score > best[0]:
                best = (score, idx, name)
        if best is not None:
            # If ambiguous (all scores 0), fallback to 1.
            if best[0] == 0:
                return 1 if cfg.num_labels > 1 else 0
            return int(best[1])

    # If num_labels==1 -> regression/logit for positive
    if getattr(cfg, "num_labels", 2) == 1:
        return 0

    # default binary: index 1
    return 1


@torch.no_grad()
def predict_scores(
    model_name: str,
    texts: List[str],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    mdl.eval().to(device)

    pos_idx = infer_positive_index(mdl)
    num_labels = mdl.config.num_labels

    all_scores = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Infer {model_name}", leave=False):
        batch = texts[i:i + batch_size]
        enc = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        out = mdl(**enc)
        logits = out.logits  # (B,C) or (B,1)

        if num_labels == 1:
            # single logit -> sigmoid
            prob_pos = torch.sigmoid(logits.squeeze(-1))
        else:
            prob = torch.softmax(logits, dim=-1)
            prob_pos = prob[:, pos_idx]

        all_scores.append(prob_pos.detach().float().cpu().numpy())

    scores = np.concatenate(all_scores, axis=0)
    meta = {
        "positive_index": int(pos_idx),
        "num_labels": int(num_labels),
        "id2label": getattr(mdl.config, "id2label", None),
        "model_name": model_name,
    }
    return scores, meta


def metrics_at_threshold(scores: np.ndarray, y: np.ndarray, thr: float) -> Dict[str, Any]:
    y = y.astype(int)
    y_pred = (scores >= thr).astype(int)

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()

    out = {
        "threshold": float(thr),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "FPR": fp / (fp + tn + 1e-12),
        "TPR": tp / (tp + fn + 1e-12),
        "TNR": tn / (tn + fp + 1e-12),
        "FNR": fn / (fn + tp + 1e-12),
    }
    # F1 for malicious (class=1)
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    out["F1_mal"] = (2 * prec * rec) / (prec + rec + 1e-12)

    out["classification_report"] = classification_report(
        y, y_pred,
        target_names=["benign(0)", "malicious(1)"],
        digits=4,
        zero_division=0,
    )
    if len(np.unique(y)) == 2:
        out["AUROC"] = float(roc_auc_score(y, scores))
    else:
        out["AUROC"] = None
    return out


def threshold_from_benign(scores: np.ndarray, y: np.ndarray, target_fpr: float) -> float:
    benign_scores = scores[y == 0]
    if len(benign_scores) == 0:
        raise ValueError("No benign samples to set threshold.")
    perc = 100.0 * (1.0 - target_fpr)
    return float(np.percentile(benign_scores, perc))


def tpr_fpr_sweep(scores: np.ndarray, y: np.ndarray, fprs=(0.001, 0.005, 0.01, 0.02, 0.05)) -> Dict[str, Any]:
    out = {}
    for fpr in fprs:
        thr = threshold_from_benign(scores, y, target_fpr=fpr)
        rep = metrics_at_threshold(scores, y, thr)
        out[str(fpr)] = {
            "thr": float(thr),
            "FPR": rep["FPR"],
            "TPR": rep["TPR"],
            "F1_mal": rep["F1_mal"],
            "TP": rep["TP"], "FN": rep["FN"], "FP": rep["FP"], "TN": rep["TN"],
        }
    return out


def per_source_summary(df: pd.DataFrame, scores: np.ndarray, thr: float, label_col: str, source_col: str) -> Dict[str, Any]:
    if source_col not in df.columns:
        return {}
    out = {}
    for src, g in df.groupby(source_col):
        idx = g.index.to_numpy()
        rep = metrics_at_threshold(scores[idx], g[label_col].to_numpy(), thr)
        out[str(src)] = {
            "n": int(len(g)),
            "benign": int((g[label_col] == 0).sum()),
            "malicious": int((g[label_col] == 1).sum()),
            "FPR": rep["FPR"],
            "TPR": rep["TPR"],
            "F1_mal": rep["F1_mal"],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", type=str, required=True)
    ap.add_argument("--text_col", type=str, default="text")
    ap.add_argument("--label_col", type=str, default="label")
    ap.add_argument("--source_col", type=str, default="source")

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--out_dir", type=str, default="sota_eval_out")
    ap.add_argument("--save_scores", action="store_true", help="save per-model scores as .npy")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    df = read_csv(args.test_csv, args.text_col, args.label_col)
    texts = df[args.text_col].astype(str).tolist()
    y = df[args.label_col].astype(int).to_numpy()

    print("Test size:", len(df))
    print("Label counts:", df[args.label_col].value_counts().to_dict())

    results = {}

    for pretty_name, hf_name in SOTA_MODELS.items():
        print(f"\n=== Evaluating: {pretty_name} ({hf_name}) ===")
        scores, meta = predict_scores(
            model_name=hf_name,
            texts=texts,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

        # Standard @0.5 threshold
        rep_05 = metrics_at_threshold(scores, y, thr=0.5)

        # FPR-target sweep (NOTE: threshold set from benign in THIS test set)
        sweep = tpr_fpr_sweep(scores, y, fprs=(0.001, 0.005, 0.01, 0.02, 0.05))

        # Also store a "thr@1%FPR" report for easy compare
        thr_1pct = threshold_from_benign(scores, y, target_fpr=0.01)
        rep_1pct = metrics_at_threshold(scores, y, thr=thr_1pct)

        # Per-source breakdown at thr@1%FPR
        src_sum = per_source_summary(df, scores, thr_1pct, args.label_col, args.source_col)

        results[pretty_name] = {
            "hf_name": hf_name,
            "meta": meta,
            "report_thr_0.5": {k: v for k, v in rep_05.items() if k != "classification_report"},
            "report_thr_0.5_text": rep_05["classification_report"],
            "report_thr_1pctFPR": {k: v for k, v in rep_1pct.items() if k != "classification_report"},
            "report_thr_1pctFPR_text": rep_1pct["classification_report"],
            "sweep": sweep,
            "per_source_at_1pctFPR": src_sum,
        }

        if args.save_scores:
            np.save(os.path.join(args.out_dir, f"{pretty_name}_scores.npy"), scores)

        # quick print (compare-friendly)
        print("AUROC:", results[pretty_name]["report_thr_0.5"]["AUROC"])
        print("thr=0.5  F1_mal:", rep_05["F1_mal"], "TPR:", rep_05["TPR"], "FPR:", rep_05["FPR"])
        print("thr@1%FPR F1_mal:", rep_1pct["F1_mal"], "TPR:", rep_1pct["TPR"], "FPR:", rep_1pct["FPR"])

    out_json = os.path.join(args.out_dir, "sota_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved:", out_json)


if __name__ == "__main__":
    main()
