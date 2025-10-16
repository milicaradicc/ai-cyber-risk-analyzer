"""
Evaluation script for ai-cyber-risk-analyzer

Loads preprocessed arrays from data/processed and evaluates baseline models
for classification (vulnerable) and regression (exploit_probability) tasks,
reporting metrics aligned with the project specification.

Usage:
    python evaluation.py --data_dir data/processed --features both --seed 42 \
        --clf_model logistic --reg_model random_forest

Features option:
    - bert: use only BERT embeddings
    - tabular: use only engineered tabular features
    - both (default): concatenate [BERT | tabular]

Outputs:
    - Prints metrics for validation and test splits
    - Saves a JSON report under {data_dir}/eval_<timestamp>.json
"""

import os
import json
import argparse
import numpy as np
from datetime import datetime
import random
import warnings

# Optional torch for seeding; evaluation does not use torch models directly
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# Optional XGBoost support
try:
    from xgboost import XGBClassifier, XGBRegressor  # type: ignore
except Exception:  # pragma: no cover
    XGBClassifier = None
    XGBRegressor = None


def set_seeds(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    if torch is not None:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception as e:
            warnings.warn(f"Torch seeding failed: {e}")


def load_split_arrays(data_dir: str, split: str):
    """Load npy arrays for a given split name (train|val|test)."""
    def p(name):
        return os.path.join(data_dir, name)

    Xb_path = p(f"X_bert_{split}.npy")
    Xt_path = p(f"X_tabular_{split}.npy")
    yclf_path = p(f"y_classification_{split}.npy")
    yreg_path = p(f"y_regression_{split}.npy")

    if not (os.path.exists(Xb_path) and os.path.exists(Xt_path)
            and os.path.exists(yclf_path) and os.path.exists(yreg_path)):
        raise FileNotFoundError(
            f"Expected arrays not found for split='{split}' under '{data_dir}'. "
            f"Make sure to run preprocessing.py first."
        )

    X_bert = np.load(Xb_path, allow_pickle=True)
    X_tab = np.load(Xt_path, allow_pickle=True)
    y_clf = np.load(yclf_path, allow_pickle=True)
    y_reg = np.load(yreg_path, allow_pickle=True)

    return X_bert, X_tab, y_clf, y_reg


def compose_features(X_bert: np.ndarray, X_tab: np.ndarray, mode: str = "both") -> np.ndarray:
    mode = (mode or "both").lower()
    if mode == "bert":
        return X_bert
    if mode == "tabular":
        return X_tab
    # default both
    return np.hstack([X_bert, X_tab])


def get_classifier(name: str, seed: int):
    n = (name or "logistic").lower()
    if n in ("logreg", "logistic", "lr"):
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed, n_jobs=None)
    if n in ("rf", "random_forest", "random-forest"):
        return RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced_subsample", n_jobs=-1)
    if n in ("xgb", "xgboost"):
        if XGBClassifier is None:
            raise ValueError("XGBoost is not installed. Please install xgboost to use this option.")
        return XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            eval_metric="logloss",
            tree_method="hist",
            objective="binary:logistic",
        )
    raise ValueError(f"Unknown classifier '{name}'. Supported: logistic, random_forest, xgboost")


def get_regressor(name: str, seed: int):
    n = (name or "random_forest").lower()
    if n in ("linreg", "linear", "linear_regression"):
        return LinearRegression(n_jobs=None)
    if n in ("rf", "random_forest", "random-forest"):
        return RandomForestRegressor(n_estimators=400, random_state=seed, n_jobs=-1)
    if n in ("xgb", "xgboost"):
        if XGBRegressor is None:
            raise ValueError("XGBoost is not installed. Please install xgboost to use this option.")
        return XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
            objective="reg:squarederror",
        )
    raise ValueError(f"Unknown regressor '{name}'. Supported: linear, random_forest, xgboost")


def evaluate_classification(model, X_train, y_train, X_eval, y_eval):
    model.fit(X_train, y_train)
    probs = None
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_eval)[:, 1]
        elif hasattr(model, "decision_function"):
            # map decision scores to [0,1] with a sigmoid-like transform when needed
            scores = model.decision_function(X_eval)
            # normalize min-max if probabilities unavailable
            m, M = np.min(scores), np.max(scores)
            probs = (scores - m) / (M - m + 1e-12)
    except Exception:
        probs = None

    y_pred = model.predict(X_eval)

    metrics = {
        "accuracy": float(accuracy_score(y_eval, y_pred)),
        "precision": float(precision_score(y_eval, y_pred, zero_division=0)),
        "recall": float(recall_score(y_eval, y_pred, zero_division=0)),
        "f1": float(f1_score(y_eval, y_pred, zero_division=0)),
    }
    # ROC-AUC and PR-AUC when probabilities/scores exist and at least 2 classes present
    if probs is not None and len(np.unique(y_eval)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_eval, probs))
        except Exception:
            pass
        try:
            metrics["pr_auc"] = float(average_precision_score(y_eval, probs))
        except Exception:
            pass

    return metrics, model


def evaluate_regression(model, X_train, y_train, X_eval, y_eval):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_eval)
    rmse = float(np.sqrt(mean_squared_error(y_eval, y_pred)))
    mae = float(mean_absolute_error(y_eval, y_pred))
    r2 = float(r2_score(y_eval, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}, model


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline models on preprocessed CVE features.")
    parser.add_argument("--data_dir", type=str, default=os.path.join("processed_data", "processed"), help="Directory with preprocessed arrays")
    parser.add_argument("--features", type=str, default="both", choices=["bert", "tabular", "both"], help="Which feature branch to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--clf_model", type=str, default="logistic", help="Classifier: logistic | random_forest | xgboost")
    parser.add_argument("--reg_model", type=str, default="random_forest", help="Regressor: linear | random_forest | xgboost")
    parser.add_argument("--save_json", action="store_true", help="Save metrics JSON under data_dir")

    args = parser.parse_args()

    set_seeds(args.seed)

    # Load arrays
    Xb_tr, Xt_tr, yclf_tr, yreg_tr = load_split_arrays(args.data_dir, "train")
    Xb_va, Xt_va, yclf_va, yreg_va = load_split_arrays(args.data_dir, "val")
    Xb_te, Xt_te, yclf_te, yreg_te = load_split_arrays(args.data_dir, "test")

    # Compose features
    Xtr = compose_features(Xb_tr, Xt_tr, args.features)
    Xva = compose_features(Xb_va, Xt_va, args.features)
    Xte = compose_features(Xb_te, Xt_te, args.features)

    # Models
    clf = get_classifier(args.clf_model, args.seed)
    reg = get_regressor(args.reg_model, args.seed)

    # Evaluate on validation
    clf_val_metrics, clf_fitted = evaluate_classification(clf, Xtr, yclf_tr, Xva, yclf_va)
    reg_val_metrics, reg_fitted = evaluate_regression(reg, Xtr, yreg_tr, Xva, yreg_va)

    # Retrain on train+val for final test evaluation to use more data
    Xtr_full = np.vstack([Xtr, Xva])
    yclf_full = np.concatenate([yclf_tr, yclf_va])
    yreg_full = np.concatenate([yreg_tr, yreg_va])

    clf_test_metrics, _ = evaluate_classification(get_classifier(args.clf_model, args.seed), Xtr_full, yclf_full, Xte, yclf_te)
    reg_test_metrics, _ = evaluate_regression(get_regressor(args.reg_model, args.seed), Xtr_full, yreg_full, Xte, yreg_te)

    report = {
        "config": {
            "data_dir": args.data_dir,
            "features": args.features,
            "seed": args.seed,
            "classifier": args.clf_model,
            "regressor": args.reg_model,
        },
        "validation": {
            "classification": clf_val_metrics,
            "regression": reg_val_metrics,
        },
        "test": {
            "classification": clf_test_metrics,
            "regression": reg_test_metrics,
        },
    }

    # Pretty print
    print("\n=== EVALUATION REPORT ===")
    print(json.dumps(report, indent=2))

    if args.save_json:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(args.data_dir, f"eval_{ts}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved evaluation JSON to: {out_path}")


if __name__ == "__main__":
    main()
