# src/evaluate.py
import torch
from sklearn.metrics import accuracy_score, f1_score

@torch.no_grad()
def evaluate_model(model, data_loader, device="cpu"):
    model.eval()
    all_preds = []
    all_labels = []
    for xb, yb in data_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        probs = model(xb)
        preds = (probs >= 0.5).long().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(yb.long().cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, f1
