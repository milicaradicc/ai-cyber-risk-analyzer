import os
import json
import argparse
import numpy as np
from datetime import datetime
import random
import warnings
import time
import pickle

from sklearn.model_selection import RandomizedSearchCV
from tqdm import tqdm

try:
    import torch
except Exception:
    torch = None

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, mean_squared_error, mean_absolute_error, r2_score
)

try:
    from xgboost import XGBClassifier, XGBRegressor
    from xgboost.callback import TrainingCallback
except Exception:
    XGBClassifier = None
    XGBRegressor = None
    TrainingCallback = None


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
    def p(name): return os.path.join(data_dir, name)

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
    return np.hstack([X_bert, X_tab])


class ProgressCallback(TrainingCallback if TrainingCallback else object):
    def __init__(self, n_estimators, task_name="Training"):
        self.n_estimators = n_estimators
        self.task_name = task_name
        self.pbar = None
        self.start_time = None
        self.iteration_times = []
        self.last_iteration = -1

    def after_iteration(self, model, epoch, evals_log):
        if self.pbar is None:
            self.pbar = tqdm(total=self.n_estimators, desc=self.task_name, unit="tree")
            self.start_time = time.time()

        if epoch > self.last_iteration:
            self.pbar.update(1)
            self.last_iteration = epoch

            elapsed = time.time() - self.start_time
            if epoch > 0:
                avg_time_per_tree = elapsed / (epoch + 1)
                self.iteration_times.append(avg_time_per_tree)
                recent_avg = np.mean(self.iteration_times[-10:]) if len(self.iteration_times) > 0 else avg_time_per_tree
                remaining_trees = self.n_estimators - (epoch + 1)
                eta_seconds = recent_avg * remaining_trees
                self.pbar.set_postfix({
                    'elapsed': f'{elapsed:.1f}s',
                    'eta': f'{eta_seconds:.1f}s',
                    'speed': f'{1/recent_avg:.2f} trees/s' if recent_avg > 0 else 'N/A'
                })
        return False

    def after_training(self, model):
        if self.pbar is not None:
            self.pbar.close()
        return model


class MLPProgressWrapper:
    def __init__(self, model, task_name="Training"):
        self.model = model
        self.task_name = task_name
        self.pbar = None
        self.start_time = None

    def fit(self, X, y):
        max_iter = self.model.max_iter
        self.pbar = tqdm(total=max_iter, desc=self.task_name, unit="epoch")
        self.start_time = time.time()

        original_verbose = self.model.verbose
        self.model.verbose = False
        original_fit = self.model.fit
        iteration_count = [0]

        def verbose_fit(X_fit, y_fit):
            result = original_fit(X_fit, y_fit)
            current_iter = getattr(self.model, 'n_iter_', iteration_count[0] + 1)
            steps = current_iter - iteration_count[0]
            if steps > 0:
                self.pbar.update(steps)
                iteration_count[0] = current_iter
                elapsed = time.time() - self.start_time
                postfix = {'elapsed': f'{elapsed:.1f}s'}
                if hasattr(self.model, 'loss_'):
                    postfix['loss'] = f'{self.model.loss_:.4f}'
                elif hasattr(self.model, 'loss_curve_') and len(self.model.loss_curve_) > 0:
                    postfix['loss'] = f'{self.model.loss_curve_[-1]:.4f}'
                avg_time_per_iter = elapsed / iteration_count[0]
                remaining = max_iter - iteration_count[0]
                postfix['eta'] = f'{avg_time_per_iter * remaining:.1f}s'
                self.pbar.set_postfix(postfix)
            return result

        self.model.fit = verbose_fit
        result = self.model.fit(X, y)
        self.model.verbose = original_verbose
        self.pbar.close()
        return result

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def __getattr__(self, name):
        return getattr(self.model, name)


def get_classifier(name: str, seed: int, show_progress=True):
    n = (name or "logistic").lower()
    if n in ("logreg", "logistic", "lr"):
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed, n_jobs=-1)
    if n in ("rf", "random_forest", "random-forest"):
        return RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=seed, verbose=2)
    if n in ("mlp", "neural_net", "neural-network", "nn"):
        base_model = MLPClassifier(hidden_layer_sizes=(256,128), activation="relu",
                                   solver="adam", alpha=1e-4, max_iter=200,
                                   shuffle=True, random_state=seed, early_stopping=True,
                                   n_iter_no_change=10, validation_fraction=0.1, verbose=False)
        return MLPProgressWrapper(base_model, "MLP Classification Training") if show_progress else base_model
    if n in ("xgb", "xgboost"):
        if XGBClassifier is None:
            raise ValueError("XGBoost is not installed.")
        n_estimators = 400
        callbacks = [ProgressCallback(n_estimators, "XGBoost Classification Training")] if show_progress else []
        return XGBClassifier(n_estimators=n_estimators, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                             random_state=seed, n_jobs=-1, eval_metric="logloss",
                             tree_method="hist", objective="binary:logistic",
                             verbosity=0, callbacks=callbacks)
    raise ValueError(f"Unknown classifier '{name}'.")


def get_regressor(name: str, seed: int, show_progress=True):
    n = (name or "random_forest").lower()
    if n in ("linreg", "linear", "linear_regression"):
        return LinearRegression(n_jobs=-1)
    if n in ("rf", "random_forest", "random-forest"):
        return RandomForestRegressor(n_estimators=400, n_jobs=-1, random_state=seed, verbose=2)
    if n in ("mlp", "neural_net", "neural-network", "nn"):
        base_model = MLPRegressor(hidden_layer_sizes=(256,128), activation="relu",
                                  solver="adam", alpha=1e-4, max_iter=300,
                                  shuffle=True, random_state=seed, early_stopping=True,
                                  n_iter_no_change=10, validation_fraction=0.1, verbose=False)
        return MLPProgressWrapper(base_model, "MLP Regression Training") if show_progress else base_model
    if n in ("xgb", "xgboost"):
        if XGBRegressor is None:
            raise ValueError("XGBoost is not installed.")
        n_estimators = 500
        callbacks = [ProgressCallback(n_estimators, "XGBoost Regression Training")] if show_progress else []
        return XGBRegressor(n_estimators=n_estimators, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                            random_state=seed, n_jobs=-1, tree_method="hist",
                            objective="reg:squarederror", verbosity=0, callbacks=callbacks)
    raise ValueError(f"Unknown regressor '{name}'.")


def evaluate_classification(model, X_train, y_train, X_eval, y_eval):
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    probs = None
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_eval)[:,1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_eval)
            m, M = np.min(scores), np.max(scores)
            probs = (scores - m)/(M-m+1e-12)
    except Exception:
        probs = None
    y_pred = model.predict(X_eval)
    metrics = {
        "accuracy": float(accuracy_score(y_eval, y_pred)),
        "precision": float(precision_score(y_eval, y_pred, zero_division=0)),
        "recall": float(recall_score(y_eval, y_pred, zero_division=0)),
        "f1": float(f1_score(y_eval, y_pred, zero_division=0)),
        "training_time_seconds": float(training_time)
    }
    if probs is not None and len(np.unique(y_eval)) == 2:
        try: metrics["roc_auc"] = float(roc_auc_score(y_eval, probs))
        except: pass
        try: metrics["pr_auc"] = float(average_precision_score(y_eval, probs))
        except: pass
    return metrics, model


def randomized_search_rf(X_train, y_train, n_iter=30, seed=42):
    rf = get_regressor("rf", seed=seed)

    param_grid = {
        'n_estimators': [200, 400, 600, 800],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=3,
        verbose=2,
        random_state=seed,
        n_jobs=-1,
        scoring='r2'
    )

    start_time = time.time()
    search.fit(X_train, y_train)
    total_time = time.time() - start_time

    best_model = search.best_estimator_

    return best_model, search.best_params_, total_time

def evaluate_regression(model, X_train, y_train, X_eval, y_eval):

    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    y_pred = model.predict(X_eval)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_eval, y_pred))),
        "mae": float(mean_absolute_error(y_eval, y_pred)),
        "r2": float(r2_score(y_eval, y_pred)),
        "training_time_seconds": float(training_time)
    }
    return metrics, model


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline models with detailed progress display.")
    parser.add_argument("--data_dir", type=str, default=os.path.join("data", "processed"))
    parser.add_argument("--features", type=str, default="both", choices=["bert", "tabular", "both"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clf_model", type=str, default="logistic")
    parser.add_argument("--reg_model", type=str, default="random_forest")
    parser.add_argument("--save_json", action="store_true")
    parser.add_argument("--save_model", action="store_true", help="If set, save the trained test-stage models to disk")
    parser.add_argument("--model_out_dir", type=str, default=os.path.join("processed_data", "models"), help="Output directory for saved models")
    parser.add_argument("--model_tag", type=str, default=None, help="Optional tag to include in saved model filenames")
    args = parser.parse_args()

    set_seeds(args.seed)

    Xb_tr, Xt_tr, yclf_tr, yreg_tr = load_split_arrays(args.data_dir, "train")
    Xb_va, Xt_va, yclf_va, yreg_va = load_split_arrays(args.data_dir, "val")
    Xb_te, Xt_te, yclf_te, yreg_te = load_split_arrays(args.data_dir, "test")

    Xtr = compose_features(Xb_tr, Xt_tr, args.features)
    Xva = compose_features(Xb_va, Xt_va, args.features)
    Xte = compose_features(Xb_te, Xt_te, args.features)

    clf = get_classifier(args.clf_model, args.seed)
    reg = get_regressor(args.reg_model, args.seed)

    clf_val_metrics, _ = evaluate_classification(clf, Xtr, yclf_tr, Xva, yclf_va)
    reg_val_metrics, _ = evaluate_regression(reg, Xtr, yreg_tr, Xva, yreg_va)

    Xtr_full = np.vstack([Xtr, Xva])
    yclf_full = np.concatenate([yclf_tr, yclf_va])
    yreg_full = np.concatenate([yreg_tr, yreg_va])

    clf_test_metrics, clf_test_model = evaluate_classification(get_classifier(args.clf_model, args.seed),
                                                   Xtr_full, yclf_full, Xte, yclf_te)

    reg_test_metrics, reg_test_model = evaluate_regression(get_regressor(args.reg_model, args.seed),
                                              Xtr_full, yreg_full, Xte, yreg_te)

    report = {
        "config": {"data_dir": args.data_dir, "features": args.features,
                   "seed": args.seed, "classifier": args.clf_model, "regressor": args.reg_model},
        "validation": {"classification": clf_val_metrics, "regression": reg_val_metrics},
        "test": {"classification": clf_test_metrics, "regression": reg_test_metrics},
    }

    if args.save_json:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("processed_data", "evaluation")
        os.makedirs(out_dir, exist_ok=True)
        clf_path = os.path.join(out_dir, f"{args.clf_model.lower()}_classification_{ts}.json")
        reg_path = os.path.join(out_dir, f"{args.reg_model.lower()}_regression_{ts}.json")
        with open(clf_path, "w", encoding="utf-8") as f:
            json.dump({"config": report["config"], "validation": clf_val_metrics, "test": clf_test_metrics}, f, indent=2)
        with open(reg_path, "w", encoding="utf-8") as f:
            json.dump({"config": report["config"], "validation": reg_val_metrics, "test": reg_test_metrics}, f, indent=2)

    if args.save_model:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(args.model_out_dir, exist_ok=True)
        tag = ("_" + args.model_tag) if args.model_tag else ""
        clf_model_fname = f"{args.clf_model.lower()}_{args.features.lower()}_classification{tag}_{ts}.pkl"
        clf_model_path = os.path.join(args.model_out_dir, clf_model_fname)
        with open(clf_model_path, "wb") as f:
            pickle.dump(clf_test_model, f)
        reg_model_fname = f"{args.reg_model.lower()}_{args.features.lower()}_regression{tag}_{ts}.pkl"
        reg_model_path = os.path.join(args.model_out_dir, reg_model_fname)
        with open(reg_model_path, "wb") as f:
            pickle.dump(reg_test_model, f)
        meta = {
            "data_dir": args.data_dir,
            "features": args.features,
            "seed": args.seed,
            "classifier": args.clf_model,
            "regressor": args.reg_model,
            "timestamp": ts,
            "artifacts": {
                "classifier_model_path": clf_model_path,
                "regressor_model_path": reg_model_path
            }
        }
        meta_path = os.path.join(args.model_out_dir, f"metadata{tag}_{ts}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
