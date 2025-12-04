import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    XGBClassifier = None
    XGBRegressor = None

try:
    import shap
except Exception:
    shap = None

try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception:
    LimeTabularExplainer = None

def load_feature_names(data_dir: str, expected_count: int):
    data_dir_abs = os.path.abspath(data_dir)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    candidate_files = [
        "feature_columns.pkl",
        "feature_names.pkl",
        "feature_columns.npy",
        "feature_names.npy",
    ]

    search_paths = []
    for filename in candidate_files:

        search_paths.append(os.path.join(data_dir_abs, filename))

        search_paths.append(os.path.join(os.path.dirname(data_dir_abs), filename))

        search_paths.append(os.path.join(repo_root, filename))

        search_paths.append(os.path.join(repo_root, "processed_data", filename))
        search_paths.append(os.path.join(data_dir_abs, "..", "processed_data", filename))

    parent = data_dir_abs
    for _ in range(3):
        parent = os.path.dirname(parent)
        if not parent or parent == '/':
            break
        for filename in candidate_files:
            search_paths.append(os.path.join(parent, filename))
            search_paths.append(os.path.join(parent, "processed_data", filename))

    seen = set()
    unique_paths = []
    for p in search_paths:
        norm_p = os.path.normpath(os.path.abspath(p))
        if norm_p not in seen:
            seen.add(norm_p)
            unique_paths.append(norm_p)

    feature_names = None

    for path in unique_paths:
        if not os.path.exists(path):
            continue

        print(f"[FOUND] {path}")

        try:
            if path.endswith('.pkl'):
                with open(path, "rb") as f:
                     obj = pickle.load(f)

                if isinstance(obj, list):
                    feature_names = obj
                elif isinstance(obj, np.ndarray):
                    feature_names = obj.tolist()
                elif isinstance(obj, dict):
                    for key in ["feature_columns", "feature_names", "features", "columns", "cols"]:
                        if key in obj:
                            feature_names = obj[key]
                            break
                    if feature_names is None and len(obj) == 1:
                        feature_names = list(obj.values())[0]
                else:
                    feature_names = obj

            elif path.endswith('.npy'):
                feature_names = np.load(path, allow_pickle=True)
                if isinstance(feature_names, np.ndarray):
                    feature_names = feature_names.tolist()

            if feature_names is not None:
                if isinstance(feature_names, np.ndarray):
                    feature_names = feature_names.tolist()
                feature_names = [str(x).strip() for x in feature_names]

                if len(feature_names) == expected_count:
                    feature_names = np.array(feature_names)
                    return feature_names
                else:
                    feature_names = None

        except Exception as e:
            print(f"Error loading: {e}")
            continue
    return None


def load_tabular(data_dir: str, split: str):
    Xt = np.load(os.path.join(data_dir, f"X_tabular_{split}.npy"), allow_pickle=True).astype(np.float32)
    y_clf = np.load(os.path.join(data_dir, f"y_classification_{split}.npy"), allow_pickle=True).astype(np.float32)
    y_reg = np.load(os.path.join(data_dir, f"y_regression_{split}.npy"), allow_pickle=True).astype(np.float32)

    feature_names = load_feature_names(data_dir, Xt.shape[1])

    if feature_names is None:
        feature_names = np.array([f"feature_{i}" for i in range(Xt.shape[1])])

    return Xt, y_clf, y_reg, feature_names

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

def _check_required_files(data_dir: str) -> bool:
    required = [
        "X_tabular_train.npy",
        "y_classification_train.npy",
        "y_regression_train.npy",
        "X_tabular_val.npy",
        "y_classification_val.npy",
        "y_regression_val.npy",
    ]
    for fn in required:
        if not os.path.exists(os.path.join(data_dir, fn)):
            return False
    return True


def resolve_data_dir(user_path: str) -> str:
    orig = user_path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    candidates = []
    candidates.append(user_path)

    if "scrpits" in user_path:
        candidates.append(user_path.replace("scrpits", "scripts"))

    candidates.append(os.path.join(repo_root, "processed_data", "processed"))
    candidates.append(os.path.join(repo_root, "scripts", "data", "processed"))
    candidates.append(os.path.join(repo_root, "data", "processed"))

    seen = set()
    uniq = []
    for c in candidates:
        cc = os.path.normpath(os.path.abspath(c))
        if cc not in seen:
            seen.add(cc)
            uniq.append(cc)

    for cand in uniq:
        if _check_required_files(cand):
            if os.path.normpath(os.path.abspath(orig)) != cand:
                print(f"Resolved data_dir: '{orig}' -> '{cand}'")
            return cand

    checked = []
    for cand in uniq:
        exists = os.path.isdir(cand)
        missing = []
        if exists:
            required = [
                "X_tabular_train.npy",
                "y_classification_train.npy",
                "y_regression_train.npy",
                "X_tabular_val.npy",
                "y_classification_val.npy",
                "y_regression_val.npy",
            ]
            for fn in required:
                if not os.path.exists(os.path.join(cand, fn)):
                    missing.append(fn)
        checked.append({"path": cand, "exists": exists, "missing": missing})

    msg_lines = [
        f"Could not find required arrays under data_dir='{orig}'.",
        "Checked these locations:",
    ]
    for item in checked:
        status = "exists" if item["exists"] else "missing_dir"
        miss = (", missing: " + ", ".join(item["missing"])) if item["exists"] else ""
        msg_lines.append(f" - {item['path']} [{status}]{miss}")
    msg_lines.append(
        "Run preprocessing.py to generate arrays, then use --data_dir processed_data/processed (recommended)."
    )
    raise FileNotFoundError("\n".join(msg_lines))


def resolve_model_path(requested_path: str, task: str, model_hint: str = None) -> str:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    search_dirs = [
        os.path.join(repo_root, "processed_data", "models"),
        os.path.join(repo_root, "processed_data"),
        os.path.join(repo_root, "scripts", "processed_data", "processed", "interpret"),
    ]

    model_hint = (model_hint or "").lower()
    task_key = "_classification_" if task == "classification" else "_regression_"

    candidates = []
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        try:
            for fname in os.listdir(d):
                if not fname.lower().endswith(".pkl"):
                    continue
                fpath = os.path.join(d, fname)
                name_l = fname.lower()
                has_task = task_key in name_l
                has_hint = (model_hint and model_hint in name_l)
                extra_hint = None
                if requested_path:
                    extra = os.path.basename(requested_path).lower()
                    extra_hint = os.path.splitext(extra)[0]
                has_extra = (extra_hint and extra_hint and (extra_hint in name_l))
                score = (
                    2 if has_task else 0
                ) + (
                    1 if has_hint else 0
                ) + (
                    1 if has_extra else 0
                )
                mtime = os.path.getmtime(fpath)
                candidates.append({
                    "path": fpath,
                    "score": score,
                    "mtime": mtime,
                    "fname": fname,
                })
        except Exception:
            continue

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x["score"], x["mtime"]), reverse=True)
    chosen = candidates[0]

    if chosen["score"] > 0 or len(candidates) == 1:
        print("Model auto-discovery:")
        print(f"  Using: {chosen['path']}")
        if requested_path and not os.path.exists(requested_path):
            print(f"  Resolved model_path: '{requested_path}' -> '{chosen['path']}'")
        return os.path.normpath(os.path.abspath(chosen["path"]))

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=os.path.join("data", "processed"))
    ap.add_argument("--model", type=str, default="xgboost")
    ap.add_argument("--task", type=str, default="classification", choices=["classification", "regression"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lime", action="store_true", help="Run a LIME explanation for one instance")
    ap.add_argument("--instance_idx", type=int, default=0)
    ap.add_argument("--num_samples", type=int, default=400000, help="Train on at most N samples for speed (0=all)")
    ap.add_argument("--model_path", type=str, default=None, help="Path to a pickled trained model to load instead of training")
    ap.add_argument("--no_train", action="store_true", help="If set with --model_path, skip training and only run explanations")
    args = ap.parse_args()

    args.data_dir = resolve_data_dir(args.data_dir)

    out_dir = os.path.join(args.data_dir, "interpret")
    ensure_dir(out_dir)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    Xt_tr, yclf_tr, yreg_tr, feature_names_tr = load_tabular(args.data_dir, "train")

    Xt_va, yclf_va, yreg_va, feature_names_va = load_tabular(args.data_dir, "val")

    assert np.array_equal(feature_names_tr, feature_names_va), "Feature imena u train i val se ne poklapaju!"
    feature_names = feature_names_tr

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

    model = None

    effective_model_path = args.model_path
    if effective_model_path and not os.path.exists(effective_model_path):
        resolved = resolve_model_path(effective_model_path, task, args.model)
        if resolved:
            effective_model_path = resolved
        else:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            mdl_dir = os.path.join(repo_root, "processed_data", "models")
            listing = []
            if os.path.isdir(mdl_dir):
                listing = [f for f in os.listdir(mdl_dir) if f.lower().endswith('.pkl')]
            msg = [f"Model file not found: {args.model_path}"]
            if listing:
                msg.append("Available saved models in processed_data/models:")
                for f in sorted(listing):
                    msg.append(f" - {f}")
            else:
                msg.append("No .pkl models found in processed_data/models. Did you run evaluation.py with --save_model?")
            raise FileNotFoundError("\n".join(msg))

    if (args.no_train and not effective_model_path):
        resolved = resolve_model_path(None, task, args.model)
        if resolved:
            effective_model_path = resolved
        else:
            raise ValueError("--no_train was set but no --model_path was provided, and no suitable saved model could be auto-discovered under processed_data/models.")

    if effective_model_path:
        with open(effective_model_path, "rb") as f:
            model = pickle.load(f)
        if not args.no_train:
            print("--model_path provided without --no_train; keeping loaded model and skipping training by default.")

    if model is None and not args.no_train:
        model = get_model(args.model, task, args.seed)
        model.fit(Xt, y)
    elif model is None and args.no_train:
        raise ValueError("--no_train was set but no --model_path was provided and no model was found. Nothing to explain.")

    if shap is not None:
        try:
            if isinstance(model, (RandomForestClassifier, RandomForestRegressor)) and hasattr(shap, 'TreeExplainer'):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(Xt)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                plt.figure()
                shap.summary_plot(shap_values, Xt, feature_names=feature_names, show=False, plot_type="bar")
                out_path = os.path.join(out_dir, f"shap_{args.model}_{task}_bar_{timestamp}.png")
                plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
                plt.figure()
                shap.summary_plot(shap_values, Xt, feature_names=feature_names, show=False)
                out_path2 = os.path.join(out_dir, f"shap_{args.model}_{task}_summary_{timestamp}.png")
                plt.tight_layout(); plt.savefig(out_path2, dpi=150); plt.close()
            elif XGBClassifier is not None and isinstance(model, (XGBClassifier, XGBRegressor)):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(Xt)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                plt.figure()
                shap.summary_plot(shap_values, Xt, feature_names=feature_names, show=False, plot_type="bar")
                out_path = os.path.join(out_dir, f"shap_{args.model}_{task}_bar_{timestamp}.png")
                plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
                plt.figure()
                shap.summary_plot(shap_values, Xt, feature_names=feature_names, show=False)
                out_path2 = os.path.join(out_dir, f"shap_{args.model}_{task}_summary_{timestamp}.png")
                plt.tight_layout(); plt.savefig(out_path2, dpi=150); plt.close()
            else:
                print("SHAP: model type not supported for tree explainer; skipping.")
        except Exception as e:
            print(f"✗ SHAP failed: {e}")
    else:
        print("shap not installed; skipping SHAP plots.")

    if args.lime and LimeTabularExplainer is not None:
        try:
            expl = LimeTabularExplainer(Xt, feature_names=feature_names.tolist(), mode='classification' if task == 'classification' else 'regression')
            x = Xt[min(args.instance_idx, len(Xt)-1)]
            if task == 'classification' and hasattr(model, 'predict_proba'):
                exp = expl.explain_instance(x, model.predict_proba, num_features=10)
            else:
                exp = expl.explain_instance(x, model.predict, num_features=10)
            out_html = os.path.join(out_dir, f"lime_{args.model}_{task}_idx{args.instance_idx}_{timestamp}.html")
            exp.save_to_file(out_html)
        except Exception as e:
            print(f"✗ LIME failed: {e}")
    elif args.lime:
        print("lime not installed; skipping LIME.")

if __name__ == "__main__":
    main()