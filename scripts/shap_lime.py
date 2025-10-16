"""
shap_lime.py — Generate SHAP feature importances and LIME explanations for tabular models.

Usage examples:
  # SHAP for XGBoost (classification):
  python shap_lime.py --data_dir data/processed --model xgboost --task classification --num_samples 500

  # LIME for a single instance (tabular features only):
  python shap_lime.py --data_dir data/processed --model random_forest --task classification --lime --instance_idx 0

Outputs saved under: data/processed/interpret/
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

try:
    from xgboost import XGBClassifier, XGBRegressor  # type: ignore
except Exception:  # pragma: no cover
    XGBClassifier = None
    XGBRegressor = None

try:
    import shap  # type: ignore
except Exception:  # pragma: no cover
    shap = None

try:
    from lime.lime_tabular import LimeTabularExplainer  # type: ignore
except Exception:  # pragma: no cover
    LimeTabularExplainer = None


def load_tabular(data_dir: str, split: str):
    Xt = np.load(os.path.join(data_dir, f"X_tabular_{split}.npy"), allow_pickle=True).astype(np.float32)
    y_clf = np.load(os.path.join(data_dir, f"y_classification_{split}.npy"), allow_pickle=True).astype(np.float32)
    y_reg = np.load(os.path.join(data_dir, f"y_regression_{split}.npy"), allow_pickle=True).astype(np.float32)
    return Xt, y_clf, y_reg


def get_model(name: str, task: str, seed: int):
    n = (name or "random_forest").lower()
    t = (task or "classification").lower()
    if t == "classification":
        if n in ("logistic", "logreg", "lr"):
            return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
        if n in ("rf", "random_forest"):
            return RandomForestClassifier(n_estimators=400, random_state=seed, class_weight="balanced_subsample", n_jobs=-1)
        if n in ("xgb", "xgboost"):
            if XGBClassifier is None:
                raise ValueError("xgboost not installed")
            return XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
                                 colsample_bytree=0.8, random_state=seed, n_jobs=-1, eval_metric="logloss",
                                 tree_method="hist")
        raise ValueError("Unknown classifier")
    else:
        if n in ("linreg", "linear"):
            return LinearRegression()
        if n in ("rf", "random_forest"):
            return RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=-1)
        if n in ("xgb", "xgboost"):
            if XGBRegressor is None:
                raise ValueError("xgboost not installed")
            return XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
                                colsample_bytree=0.8, random_state=seed, n_jobs=-1, tree_method="hist")
        raise ValueError("Unknown regressor")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=os.path.join("processed_data", "processed"))
    ap.add_argument("--model", type=str, default="xgboost")
    ap.add_argument("--task", type=str, default="classification", choices=["classification", "regression"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_samples", type=int, default=1000, help="Train on at most N samples for speed (0=all)")
    ap.add_argument("--lime", action="store_true", help="Run a LIME explanation for one instance")
    ap.add_argument("--instance_idx", type=int, default=0)
    args = ap.parse_args()

    out_dir = os.path.join(args.data_dir, "interpret")
    ensure_dir(out_dir)

    Xt_tr, yclf_tr, yreg_tr = load_tabular(args.data_dir, "train")
    Xt_va, yclf_va, yreg_va = load_tabular(args.data_dir, "val")
    Xt = np.vstack([Xt_tr, Xt_va])
    yclf = np.concatenate([yclf_tr, yclf_va])
    yreg = np.concatenate([yreg_tr, yreg_va])

    if args.num_samples and args.num_samples > 0 and args.num_samples < len(Xt):
        idx = np.random.choice(len(Xt), size=args.num_samples, replace=False)
        Xt = Xt[idx]
        yclf = yclf[idx]
        yreg = yreg[idx]

    task = args.task.lower()
    y = yclf if task == "classification" else yreg
    model = get_model(args.model, task, args.seed)
    model.fit(Xt, y)

    # SHAP (tree models best)
    if shap is not None:
        try:
            if isinstance(model, (RandomForestClassifier, RandomForestRegressor)) and hasattr(shap, 'TreeExplainer'):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(Xt)
                # For classifiers, shap_values can be list (per class)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                plt.figure()
                shap.summary_plot(shap_values, Xt, show=False, plot_type="bar")
                out_path = os.path.join(out_dir, f"shap_{args.model}_{task}_bar.png")
                plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
                plt.figure()
                shap.summary_plot(shap_values, Xt, show=False)
                out_path2 = os.path.join(out_dir, f"shap_{args.model}_{task}_summary.png")
                plt.tight_layout(); plt.savefig(out_path2, dpi=150); plt.close()
                print(f"Saved SHAP plots to {out_dir}")
            elif XGBClassifier is not None and isinstance(model, (XGBClassifier, XGBRegressor)):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(Xt)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                plt.figure()
                shap.summary_plot(shap_values, Xt, show=False, plot_type="bar")
                out_path = os.path.join(out_dir, f"shap_{args.model}_{task}_bar.png")
                plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
                plt.figure()
                shap.summary_plot(shap_values, Xt, show=False)
                out_path2 = os.path.join(out_dir, f"shap_{args.model}_{task}_summary.png")
                plt.tight_layout(); plt.savefig(out_path2, dpi=150); plt.close()
                print(f"Saved SHAP plots to {out_dir}")
            else:
                print("SHAP: model type not supported for tree explainer; skipping.")
        except Exception as e:
            print(f"SHAP failed: {e}")
    else:
        print("shap not installed; skipping SHAP plots.")

    # LIME (tabular)
    if args.lime and LimeTabularExplainer is not None:
        try:
            expl = LimeTabularExplainer(Xt, mode='classification' if task == 'classification' else 'regression')
            x = Xt[min(args.instance_idx, len(Xt)-1)]
            if task == 'classification' and hasattr(model, 'predict_proba'):
                exp = expl.explain_instance(x, model.predict_proba, num_features=10)
            else:
                exp = expl.explain_instance(x, model.predict, num_features=10)
            out_html = os.path.join(out_dir, f"lime_{args.model}_{task}_idx{args.instance_idx}.html")
            exp.save_to_file(out_html)
            print(f"Saved LIME explanation to {out_html}")
        except Exception as e:
            print(f"LIME failed: {e}")
    elif args.lime:
        print("lime not installed; skipping LIME.")


if __name__ == "__main__":
    main()
