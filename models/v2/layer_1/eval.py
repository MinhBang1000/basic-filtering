import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

def score_dataset(ae, embedder, loader, device, err_kind):
    ae.eval()
    all_scores, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            x = embedder.encode(
                batch.input_ids.to(device),
                batch.attention_mask.to(device),
            )
            x_hat, _ = ae(x)
            scores = ((x_hat - x) ** 2).mean(1).cpu().numpy()
            all_scores.append(scores)
            all_labels.append(batch.labels.numpy())

    return np.concatenate(all_scores), np.concatenate(all_labels)
