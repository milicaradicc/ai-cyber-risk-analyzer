import os
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur

def list_metric_files(eval_dir: str) -> List[str]:
    if not os.path.isdir(eval_dir):
        raise FileNotFoundError(f"Evaluation directory not found: {eval_dir}")
    files = [os.path.join(eval_dir, f) for f in os.listdir(eval_dir) if f.endswith('.json')]
    files = [f for f in files if os.path.basename(f) not in ("best_model.json",)]
    return sorted(files)

def list_multitask_files(mt_dir: str) -> List[str]:
    if not os.path.isdir(mt_dir):
        return []
    files = [os.path.join(mt_dir, f) for f in os.listdir(mt_dir)
             if f.lower().startswith("multitask_metrics_") and f.lower().endswith('.json')]
    return sorted(files)

def parse_record(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return None

    base = os.path.basename(path).lower()
    task = 'classification' if '_classification_' in base else ('regression' if '_regression_' in base else None)

    val_metrics = safe_get(data, ["validation"], {}) or {}
    test_metrics = safe_get(data, ["test"], {}) or {}

    config = safe_get(data, ["config"], {}) or {}

    return {
        "path": path,
        "task": task,
        "config": config,
        "validation": val_metrics,
        "test": test_metrics,
        "filename": os.path.basename(path)
    }

def cls_rank_key(metrics: Dict[str, Any]) -> Tuple:
    auc = metrics.get('roc_auc')
    f1 = metrics.get('f1')
    acc = metrics.get('accuracy')
    t = metrics.get('training_time_seconds')
    import math
    f1_val = f1 if isinstance(f1, (int, float)) else -math.inf
    auc_primary = auc if isinstance(auc, (int, float)) else f1_val
    acc = acc if isinstance(acc, (int, float)) else -math.inf
    t = t if isinstance(t, (int, float)) else math.inf
    return (auc_primary, f1_val, acc, -t)

def reg_rank_key(metrics: Dict[str, Any]) -> Tuple:
    rmse = metrics.get('rmse')
    mae = metrics.get('mae')
    r2 = metrics.get('r2')
    t = metrics.get('training_time_seconds')
    import math
    rmse = rmse if isinstance(rmse, (int, float)) else math.inf
    mae = mae if isinstance(mae, (int, float)) else math.inf
    r2 = r2 if isinstance(r2, (int, float)) else -math.inf
    t = t if isinstance(t, (int, float)) else math.inf
    return (-rmse, -mae, r2, -t)

def parse_multitask_records(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return []
    val = safe_get(data, ["validation"], {}) or {}
    tes = safe_get(data, ["test"], {}) or {}
    cfg = safe_get(data, ["config"], {}) or {}
    cls_rec = {
        "path": path,
        "task": "classification",
        "config": {**cfg, "source": "multitask"},
        "validation": {k: v for k, v in val.items() if k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "training_time_seconds")},
        "test": {k: v for k, v in tes.items() if k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "training_time_seconds")},
        "filename": os.path.basename(path)
    }
    reg_rec = {
        "path": path,
        "task": "regression",
        "config": {**cfg, "source": "multitask"},
        "validation": {k: v for k, v in val.items() if k in ("rmse", "mae", "r2", "training_time_seconds")},
        "test": {k: v for k, v in tes.items() if k in ("rmse", "mae", "r2", "training_time_seconds")},
        "filename": os.path.basename(path)
    }
    return [cls_rec, reg_rec]

def choose_best(records: List[Dict[str, Any]], task: str) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    scored = []
    for r in records:
        if r.get('task') != task:
            continue
        test_m = r.get('test') or {}
        val_m = r.get('validation') or {}
        m = test_m if test_m else val_m
        if not isinstance(m, dict):
            continue
        key_fn = cls_rank_key if task == 'classification' else reg_rank_key
        score = key_fn(m)
        scored.append({**r, "_score": score, "_used_metrics": m})

    if not scored:
        return None, []

    scored.sort(key=lambda x: x['_score'], reverse=True)
    best = scored[0]
    return best, scored

def summarize(scored: List[Dict[str, Any]], task: str, topk: int = 3) -> str:
    lines = []
    lines.append(f"Top {min(topk, len(scored))} for {task} (by test metrics):")
    for i, r in enumerate(scored[:topk]):
        cfg = r.get('config', {})
        model_name = (cfg.get('classifier') if task == 'classification' else cfg.get('regressor')) or 'N/A'
        feats = cfg.get('features', 'N/A')
        m = r.get('_used_metrics', {})
        if task == 'classification':
            lines.append(f"  {i+1}. {model_name} [{feats}] -> AUC={m.get('roc_auc', 'NA')}, F1={m.get('f1', 'NA')}, Acc={m.get('accuracy', 'NA')} ({os.path.basename(r['path'])})")
        else:
            lines.append(f"  {i+1}. {model_name} [{feats}] -> RMSE={m.get('rmse', 'NA')}, MAE={m.get('mae', 'NA')}, R2={m.get('r2', 'NA')} ({os.path.basename(r['path'])})")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser(description="Select the best model based on evaluation JSON files.")
    ap.add_argument('--eval_dir', type=str, default=os.path.join('processed_data', 'evaluation'), help='Directory with evaluation JSONs')
    ap.add_argument('--multitask_dir', type=str, default=os.path.join('processed_data', 'multitask'), help='Directory with multitask metrics JSONs')
    ap.add_argument('--include_multitask', action='store_true', help='Include multitask metrics in comparison (if present)')
    ap.add_argument('--save_summary', action='store_true', help='If set, write best_model.json in eval_dir')
    ap.add_argument('--print_paths', action='store_true', help='Print only paths of best classification/regression JSONs (machine-readable)')
    args = ap.parse_args()

    files = list_metric_files(args.eval_dir)

    mt_files = list_multitask_files(args.multitask_dir) if args.include_multitask else []

    if not files and not mt_files:
        print(f"No evaluation JSON files found in: {args.eval_dir} and no multitask metrics in: {args.multitask_dir}")
        return 1

    records = []
    for p in files:
        rec = parse_record(p)
        if rec is not None:
            records.append(rec)

    for mp in mt_files:
        mt_recs = parse_multitask_records(mp)
        records.extend(mt_recs)

    best_cls, cls_ranked = choose_best(records, 'classification')
    best_reg, reg_ranked = choose_best(records, 'regression')

    if args.print_paths:
        return 0

    if best_cls:
        print("\n[Classification]")
        print(summarize(cls_ranked, 'classification'))
        cfg = best_cls.get('config', {})
        best_m = best_cls.get('_used_metrics', {})
        print("\n=> Best classification model:")
        print(json.dumps({
            "model": cfg.get('classifier'),
            "features": cfg.get('features'),
            "metrics": best_m,
            "metrics_file": best_cls.get('path')
        }, indent=2))
    else:
        print("\n[Classification] No classification JSONs found.")

    if best_reg:
        print("\n[Regression]")
        print(summarize(reg_ranked, 'regression'))
        cfg = best_reg.get('config', {})
        best_m = best_reg.get('_used_metrics', {})
        print("\n=> Best regression model:")
        print(json.dumps({
            "model": cfg.get('regressor'),
            "features": cfg.get('features'),
            "metrics": best_m,
            "metrics_file": best_reg.get('path')
        }, indent=2))
    else:
        print("\n[Regression] No regression JSONs found.")

    if args.save_summary:
        out = {
            "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S'),
            "criteria": {
                "classification": "Sort by roc_auc desc, then f1 desc, accuracy desc, training_time asc",
                "regression": "Sort by rmse asc, then mae asc, r2 desc, training_time asc"
            },
            "best": {
                "classification": {
                    "metrics_file": (best_cls or {}).get('path') if best_cls else None,
                    "config": (best_cls or {}).get('config') if best_cls else None,
                    "metrics": (best_cls or {}).get('_used_metrics') if best_cls else None
                },
                "regression": {
                    "metrics_file": (best_reg or {}).get('path') if best_reg else None,
                    "config": (best_reg or {}).get('config') if best_reg else None,
                    "metrics": (best_reg or {}).get('_used_metrics') if best_reg else None
                }
            }
        }
        out_path = os.path.join(args.eval_dir, 'best_model.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
