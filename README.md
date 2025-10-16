# ai-cyber-risk-analyzer

Preprocessing pipeline for CVE vulnerability prediction and exploitability regression, combining BERT embeddings with engineered tabular features. Aligned with the project specification “Predikcija i klasifikacija softverskih ranjivosti”.

## Quickstart

```bash
python preprocessing.py
```

This will:
- Load preprocessing.csv
- Clean and engineer features (auto-create targets if missing)
- Generate BERT embeddings in batches
- Split data into train/val/test = 70%/15%/15%
- Optionally balance the train split
- Save arrays and artifacts under data/processed

## Parameters (run_full_pipeline)
- filepath: path to CSV (default preprocessing.csv)
- save_dir: output directory (default data/processed)
- resample_mode: 'none' | 'tabular_oversample' | 'combined_smote'
- seed: random seed (default 42)
- batch_size: BERT embedding batch size (default 32)
- auto_create_targets: create targets if missing (default True)
- top_n_cwe: how many top CWE categories to keep before grouping others into OTHER (default 20)
- high_score_threshold: baseScore threshold for has_high_score (default 7.0)
- old_days_threshold: days_since_published threshold for is_old (default 365)

## Outputs
Saved to save_dir:
- X_bert_[train|val|test].npy
- X_tabular_[train|val|test].npy
- y_classification_[train|val|test].npy
- y_regression_[train|val|test].npy
- scaler.pkl (fitted StandardScaler)
- feature_columns.pkl (tabular feature order)
- cwe_top_list.pkl (if CWE grouping used)

## Requirements
See requirements.txt. Install with:
```bash
pip install -r requirements.txt
```

## Notes
- BERT model: bert-base-uncased by default; change via CVEPreprocessor(bert_model=...).
- Ensure preprocessing.csv contains at least: description, CVE_ID, and preferably CVSS-related columns. If targets are absent, auto_create_targets=True will derive them using num_exploits and exploitabilityScore (or baseScore fallback).

## Evaluation

After running preprocessing and generating arrays under data/processed, you can evaluate baseline models:

```bash
python evaluation.py --data_dir processed_data/processed --features both --seed 42 --clf_model xgboost --reg_model xgboost --save_json
```

- features: bert | tabular | both (default both concatenates [BERT | tabular])
- classifiers: logistic | random_forest | xgboost
- regressors: linear | random_forest | xgboost
- classification metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC (where applicable)
- regression metrics: RMSE, MAE, R²
- results are printed and optionally saved to data/processed/eval_<timestamp>.json

## Multi-task learning (classification + regression)

A simple PyTorch multi-task MLP is provided:

```bash
python multitask.py --data_dir processed_data/processed --features both --epochs 10 --batch_size 64 --lr 1e-3 --weight_decay 1e-4 --warmup_ratio 0.1 --clf_weight 1.0 --reg_weight 1.0 --seed 42
```

- Uses BCEWithLogitsLoss for classification and MSELoss for regression with AdamW and linear warmup.
- Saves best checkpoint to data/processed/multitask_best.pt and metrics to data/processed/multitask_metrics.json.

## Visualization (t-SNE / UMAP)

Create 2D projections of BERT embeddings colored by class:

```bash
python tsne_umap.py --data_dir processed_data/processed --split train --sample 1000 --perplexity 30 --n_neighbors 15 --min_dist 0.1
```

Images will be saved under data/processed/visualizations.

## Interpretability (SHAP / LIME)

Generate SHAP feature importances (tree/XGBoost) and optional LIME explanations:

```bash
python shap_lime.py --data_dir processed_data/processed --model xgboost --task classification --num_samples 500
# LIME for an instance
python shap_lime.py --data_dir processed_data/processed --model random_forest --task classification --lime --instance_idx 0
```

Artifacts will be saved under data/processed/interpret.

## Dashboard

A minimal Streamlit dashboard to browse evaluation metrics and artifacts:

```bash
streamlit run app.py
```

- Loads latest evaluation JSON from data/processed and displays validation/test metrics.
- Shows visualization images (t-SNE/UMAP) and SHAP/LIME outputs if available.
- Includes a simple similarity search over BERT embeddings.
