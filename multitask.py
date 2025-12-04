import os
import json
import math
import argparse
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW


def set_seeds(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_arrays(data_dir: str, split: str):
    p = lambda name: os.path.join(data_dir, name)
    Xb = np.load(p(f"X_bert_{split}.npy"), allow_pickle=True).astype(np.float32)
    Xt = np.load(p(f"X_tabular_{split}.npy"), allow_pickle=True).astype(np.float32)
    yclf = np.load(p(f"y_classification_{split}.npy"), allow_pickle=True).astype(np.float32)
    yreg = np.load(p(f"y_regression_{split}.npy"), allow_pickle=True).astype(np.float32)
    return Xb, Xt, yclf, yreg


def compose_features(Xb: np.ndarray, Xt: np.ndarray, mode: str = "both") -> np.ndarray:
    m = (mode or "both").lower()
    if m == "bert":
        return Xb
    if m == "tabular":
        return Xt
    return np.hstack([Xb, Xt])


class MultiTaskMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims=(512, 256), dropout=0.2):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last = h
        self.backbone = nn.Sequential(*layers)
        self.clf_head = nn.Linear(last, 1)
        self.reg_head = nn.Linear(last, 1)

    def forward(self, x):
        h = self.backbone(x)
        logit = self.clf_head(h).squeeze(-1)
        reg = self.reg_head(h).squeeze(-1)
        return logit, reg

def linear_warmup_cosine_decay(step, total_steps, warmup_steps):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))

def evaluate(model, loader, device, bce, mse):
    model.eval()
    all_yc, all_yr = [], []
    all_pc, all_pr = [], []
    total_loss = 0.0
    with torch.no_grad():
        for xb, ycls, yreg in loader:
            xb = xb.to(device)
            ycls = ycls.to(device)
            yreg = yreg.to(device)
            logits, preds_reg = model(xb)
            loss = bce(logits, ycls) + mse(preds_reg, yreg)
            total_loss += loss.item() * xb.size(0)
            probs = torch.sigmoid(logits)
            all_yc.append(ycls.detach().cpu().numpy())
            all_yr.append(yreg.detach().cpu().numpy())
            all_pc.append(probs.detach().cpu().numpy())
            all_pr.append(preds_reg.detach().cpu().numpy())
    n = sum(len(a) for a in all_yc)
    avg_loss = total_loss / max(1, n)
    y_c = np.concatenate(all_yc).astype(int)
    y_r = np.concatenate(all_yr)
    p_c = np.concatenate(all_pc)
    p_r = np.concatenate(all_pr)

    y_pred_cls = (p_c >= 0.5).astype(int)
    acc = float((y_pred_cls == y_c).mean())

    tp = ((y_pred_cls == 1) & (y_c == 1)).sum()
    fp = ((y_pred_cls == 1) & (y_c == 0)).sum()
    fn = ((y_pred_cls == 0) & (y_c == 1)).sum()
    prec = float(tp / (tp + fp + 1e-12))
    rec = float(tp / (tp + fn + 1e-12))
    f1 = float(2 * prec * rec / (prec + rec + 1e-12))
    rmse = float(np.sqrt(np.mean((p_r - y_r) ** 2)))
    mae = float(np.mean(np.abs(p_r - y_r)))

    ss_res = float(np.sum((y_r - p_r) ** 2))
    ss_tot = float(np.sum((y_r - np.mean(y_r)) ** 2))
    r2 = float(1 - ss_res / (ss_tot + 1e-12))
    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def main():
    parser = argparse.ArgumentParser(description="Train a simple multi-task MLP on preprocessed features.")
    parser.add_argument("--data_dir", type=str, default=os.path.join("processed_data", "processed"))
    parser.add_argument("--features", type=str, default="both", choices=["bert", "tabular", "both"], help="Which feature branch to use")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--clf_weight", type=float, default=1.0)
    parser.add_argument("--reg_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden", type=str, default="512,256", help="Comma-separated hidden sizes, e.g., 512,256")

    args = parser.parse_args()

    set_seeds(args.seed)

    Xb_tr, Xt_tr, yclf_tr, yreg_tr = load_arrays(args.data_dir, "train")
    Xb_va, Xt_va, yclf_va, yreg_va = load_arrays(args.data_dir, "val")
    Xb_te, Xt_te, yclf_te, yreg_te = load_arrays(args.data_dir, "test")

    Xtr = compose_features(Xb_tr, Xt_tr, args.features)
    Xva = compose_features(Xb_va, Xt_va, args.features)
    Xte = compose_features(Xb_te, Xt_te, args.features)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_tensor(x):
        return torch.tensor(x, dtype=torch.float32)

    tr_ds = TensorDataset(to_tensor(Xtr), torch.tensor(yclf_tr, dtype=torch.float32), torch.tensor(yreg_tr, dtype=torch.float32))
    va_ds = TensorDataset(to_tensor(Xva), torch.tensor(yclf_va, dtype=torch.float32), torch.tensor(yreg_va, dtype=torch.float32))
    te_ds = TensorDataset(to_tensor(Xte), torch.tensor(yclf_te, dtype=torch.float32), torch.tensor(yreg_te, dtype=torch.float32))

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size)
    te_loader = DataLoader(te_ds, batch_size=args.batch_size)

    hidden = tuple(int(h.strip()) for h in args.hidden.split(",") if h.strip())
    model = MultiTaskMLP(in_dim=Xtr.shape[1], hidden_dims=hidden, dropout=args.dropout).to(device)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(tr_loader) * args.epochs)
    warmup_steps = int(args.warmup_ratio * total_steps)

    bce = nn.BCEWithLogitsLoss(reduction="mean")
    mse = nn.MSELoss(reduction="mean")

    best_val_f1 = -1.0
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("processed_data", "multitask")
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, f"multitask_best_{ts}.pt")

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, ycls, yreg in tr_loader:
            xb = xb.to(device)
            ycls = ycls.to(device)
            yreg = yreg.to(device)
            logits, preds_reg = model(xb)
            loss = args.clf_weight * bce(logits, ycls) + args.reg_weight * mse(preds_reg, yreg)
            loss.backward()

            scale = linear_warmup_cosine_decay(global_step, total_steps, warmup_steps)
            for g in opt.param_groups:
                g["lr"] = args.lr * scale
            opt.step()
            opt.zero_grad(set_to_none=True)
            global_step += 1

        val_metrics = evaluate(model, va_loader, device, bce, mse)
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save({"model": model.state_dict(), "config": vars(args)}, best_path)

    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])

    val_metrics = evaluate(model, va_loader, device, bce, mse)
    test_metrics = evaluate(model, te_loader, device, bce, mse)

    report = {
        "config": vars(args),
        "validation": val_metrics,
        "test": test_metrics,
    }

    out_json = os.path.join(out_dir, f"multitask_metrics_{ts}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()
