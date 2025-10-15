import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, precision_recall_curve,
    mean_squared_error, mean_absolute_error, r2_score
)
import torch
from inference import CVEVulnerabilityPredictor
from config import Config
from utils import load_pickle

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================================
# EVALUATION METRICS
# ============================================================================

def evaluate_classification(y_true, y_pred_proba, threshold=0.5):
    """Evaluacija klasifikacionog modela"""
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    print("\n" + "="*70)
    print("🎯 CLASSIFICATION EVALUATION")
    print("="*70)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    print("\nConfusion Matrix:")
    print(cm)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, 
                               target_names=['Not Vulnerable', 'Vulnerable'],
                               digits=4))
    
    # ROC-AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")
    
    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    print(f"PR-AUC Score: {pr_auc:.4f}")
    
    return {
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'pr_auc': pr_auc
    }

def evaluate_regression(y_true, y_pred):
    """Evaluacija regresionog modela"""
    print("\n" + "="*70)
    print("📈 REGRESSION EVALUATION")
    print("="*70)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\nMean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Error distribution
    errors = y_true - y_pred
    print(f"\nError Statistics:")
    print(f"  Mean Error: {np.mean(errors):.4f}")
    print(f"  Std Error: {np.std(errors):.4f}")
    print(f"  Min Error: {np.min(errors):.4f}")
    print(f"  Max Error: {np.max(errors):.4f}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'errors': errors
    }

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_confusion_matrix(cm, save_path):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Vulnerable', 'Vulnerable'],
                yticklabels=['Not Vulnerable', 'Vulnerable'],
                ax=ax, cbar_kws={'label': 'Count'})
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Confusion matrix saved to {save_path}")
    plt.close()

def plot_roc_curve(fpr, tpr, roc_auc, save_path):
    """Plot ROC curve"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', 
                fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 ROC curve saved to {save_path}")
    plt.close()

def plot_precision_recall_curve(precision, recall, pr_auc, save_path):
    """Plot Precision-Recall curve"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(recall, precision, color='blue', lw=2,
            label=f'PR curve (AUC = {pr_auc:.4f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Precision-Recall curve saved to {save_path}")
    plt.close()

def plot_regression_results(y_true, y_pred, save_path):
    """Plot regression predictions vs actual"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=20)
    axes[0].plot([y_true.min(), y_true.max()], 
                 [y_true.min(), y_true.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('True Exploit Probability', fontsize=12)
    axes[0].set_ylabel('Predicted Exploit Probability', fontsize=12)
    axes[0].set_title('Predictions vs Actual', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residual plot
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Exploit Probability', fontsize=12)
    axes[1].set_ylabel('Residuals', fontsize=12)
    axes[1].set_title('Residual Plot', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Regression plots saved to {save_path}")
    plt.close()

def plot_error_distribution(errors, save_path):
    """Plot error distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram
    axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Prediction Error', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Error Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Error distribution saved to {save_path}")
    plt.close()

def plot_prediction_distribution(y_pred_proba, y_true, save_path):
    """Plot distribution of predictions by class"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Separate by true class
    vulnerable_probs = y_pred_proba[y_true == 1]
    not_vulnerable_probs = y_pred_proba[y_true == 0]
    
    ax.hist(not_vulnerable_probs, bins=50, alpha=0.6, 
            label='Not Vulnerable (True)', color='blue', edgecolor='black')
    ax.hist(vulnerable_probs, bins=50, alpha=0.6, 
            label='Vulnerable (True)', color='red', edgecolor='black')
    
    ax.axvline(x=0.5, color='black', linestyle='--', lw=2, 
               label='Decision Threshold')
    
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Prediction Probability Distribution by True Class', 
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Prediction distribution saved to {save_path}")
    plt.close()

def plot_risk_distribution(results_df, save_path):
    """Plot risk score and category distribution"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Risk score histogram
    axes[0, 0].hist(results_df['risk_score'], bins=50, 
                    edgecolor='black', alpha=0.7, color='orange')
    axes[0, 0].set_xlabel('Risk Score', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Risk Score Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Risk category counts
    risk_counts = results_df['risk_category'].value_counts().sort_index()
    colors = ['green', 'yellow', 'orange', 'red']
    axes[0, 1].bar(risk_counts.index, risk_counts.values, 
                   color=colors, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Risk Category', fontsize=12)
    axes[0, 1].set_ylabel('Count', fontsize=12)
    axes[0, 1].set_title('Risk Category Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Vulnerable probability vs Exploit probability
    axes[1, 0].scatter(results_df['vulnerable_probability'], 
                       results_df['exploit_probability'],
                       c=results_df['risk_score'], cmap='RdYlGn_r',
                       alpha=0.6, s=20)
    axes[1, 0].set_xlabel('Vulnerable Probability', fontsize=12)
    axes[1, 0].set_ylabel('Exploit Probability', fontsize=12)
    axes[1, 0].set_title('Vulnerability vs Exploit Probability', 
                         fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Risk score by prediction
    vuln_risk = results_df[results_df['vulnerable_prediction'] == 1]['risk_score']
    not_vuln_risk = results_df[results_df['vulnerable_prediction'] == 0]['risk_score']
    
    axes[1, 1].boxplot([not_vuln_risk, vuln_risk], 
                        labels=['Not Vulnerable', 'Vulnerable'],
                        patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7))
    axes[1, 1].set_ylabel('Risk Score', fontsize=12)
    axes[1, 1].set_title('Risk Score by Vulnerability Prediction', 
                         fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Risk distribution saved to {save_path}")
    plt.close()

# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def comprehensive_evaluation(model_path, test_csv_path, output_dir):
    """
    Kompletna evaluacija modela sa svim metrikama i vizualizacijama
    """
    print("\n" + "="*70)
    print("🔬 COMPREHENSIVE MODEL EVALUATION")
    print("="*70)
    
    # Create evaluation output directory
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Load test data
    print(f"\n📂 Loading test data from {test_csv_path}...")
    test_df = pd.read_csv(test_csv_path)
    print(f"✅ Loaded {len(test_df)} test samples")
    
    # Load ground truth
    y_true_clf = load_pickle(os.path.join(output_dir, "y_test_clf.pkl"))
    y_true_reg = load_pickle(os.path.join(output_dir, "y_test_reg.pkl"))
    
    # Initialize predictor and make predictions
    predictor = CVEVulnerabilityPredictor(model_path, output_dir)
    results = predictor.predict(test_df, batch_size=32)
    
    # Extract predictions
    y_pred_clf_proba = results['vulnerable_probability'].values
    y_pred_reg = results['exploit_probability'].values
    
    # ========================================================================
    # CLASSIFICATION EVALUATION
    # ========================================================================
    clf_metrics = evaluate_classification(y_true_clf, y_pred_clf_proba)
    
    # Plot classification visualizations
    plot_confusion_matrix(
        clf_metrics['confusion_matrix'],
        os.path.join(eval_dir, "confusion_matrix.png")
    )
    
    plot_roc_curve(
        clf_metrics['fpr'], clf_metrics['tpr'], clf_metrics['roc_auc'],
        os.path.join(eval_dir, "roc_curve.png")
    )
    
    plot_precision_recall_curve(
        clf_metrics['precision'], clf_metrics['recall'], clf_metrics['pr_auc'],
        os.path.join(eval_dir, "precision_recall_curve.png")
    )
    
    plot_prediction_distribution(
        y_pred_clf_proba, y_true_clf,
        os.path.join(eval_dir, "prediction_distribution.png")
    )
    
    # ========================================================================
    # REGRESSION EVALUATION
    # ========================================================================
    reg_metrics = evaluate_regression(y_true_reg, y_pred_reg)
    
    # Plot regression visualizations
    plot_regression_results(
        y_true_reg, y_pred_reg,
        os.path.join(eval_dir, "regression_predictions.png")
    )
    
    plot_error_distribution(
        reg_metrics['errors'],
        os.path.join(eval_dir, "error_distribution.png")
    )
    
    # ========================================================================
    # RISK ANALYSIS
    # ========================================================================
    plot_risk_distribution(
        results,
        os.path.join(eval_dir, "risk_distribution.png")
    )
    
    # ========================================================================
    # SAVE DETAILED RESULTS
    # ========================================================================
    
    # Merge predictions with ground truth
    detailed_results = results.copy()
    detailed_results['true_vulnerable'] = y_true_clf
    detailed_results['true_exploit_prob'] = y_true_reg
    
    # Calculate prediction errors
    detailed_results['clf_error'] = (detailed_results['vulnerable_prediction'] != detailed_results['true_vulnerable']).astype(int)
    detailed_results['reg_error'] = detailed_results['exploit_probability'] - detailed_results['true_exploit_prob']
    detailed_results['abs_reg_error'] = np.abs(detailed_results['reg_error'])
    
    # Save to CSV
    results_path = os.path.join(eval_dir, "detailed_results.csv")
    detailed_results.to_csv(results_path, index=False)
    print(f"\n💾 Detailed results saved to {results_path}")
    
    # ========================================================================
    # GENERATE EVALUATION REPORT
    # ========================================================================
    
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("CVE VULNERABILITY PREDICTION - EVALUATION REPORT")
    report_lines.append("="*70)
    report_lines.append("")
    
    report_lines.append("DATASET INFORMATION")
    report_lines.append("-" * 70)
    report_lines.append(f"Total test samples: {len(test_df)}")
    report_lines.append(f"Vulnerable samples: {sum(y_true_clf)} ({sum(y_true_clf)/len(y_true_clf)*100:.2f}%)")
    report_lines.append(f"Not vulnerable samples: {len(y_true_clf) - sum(y_true_clf)} ({(len(y_true_clf) - sum(y_true_clf))/len(y_true_clf)*100:.2f}%)")
    report_lines.append("")
    
    report_lines.append("CLASSIFICATION METRICS")
    report_lines.append("-" * 70)
    report_lines.append(f"ROC-AUC Score: {clf_metrics['roc_auc']:.4f}")
    report_lines.append(f"PR-AUC Score: {clf_metrics['pr_auc']:.4f}")
    
    cm = clf_metrics['confusion_matrix']
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    report_lines.append(f"Accuracy: {accuracy:.4f}")
    report_lines.append(f"Precision: {precision:.4f}")
    report_lines.append(f"Recall (Sensitivity): {recall:.4f}")
    report_lines.append(f"Specificity: {specificity:.4f}")
    report_lines.append(f"F1-Score: {f1:.4f}")
    report_lines.append("")
    
    report_lines.append("CONFUSION MATRIX")
    report_lines.append("-" * 70)
    report_lines.append(f"True Negatives:  {tn:6d}")
    report_lines.append(f"False Positives: {fp:6d}")
    report_lines.append(f"False Negatives: {fn:6d}")
    report_lines.append(f"True Positives:  {tp:6d}")
    report_lines.append("")
    
    report_lines.append("REGRESSION METRICS")
    report_lines.append("-" * 70)
    report_lines.append(f"R² Score: {reg_metrics['r2']:.4f}")
    report_lines.append(f"RMSE: {reg_metrics['rmse']:.4f}")
    report_lines.append(f"MAE: {reg_metrics['mae']:.4f}")
    report_lines.append(f"MSE: {reg_metrics['mse']:.4f}")
    report_lines.append("")
    
    report_lines.append("RISK ASSESSMENT SUMMARY")
    report_lines.append("-" * 70)
    risk_counts = results['risk_category'].value_counts().sort_index()
    for category, count in risk_counts.items():
        percentage = count / len(results) * 100
        report_lines.append(f"{category:12s}: {count:6d} ({percentage:5.2f}%)")
    report_lines.append("")
    
    report_lines.append("TOP 10 HIGHEST RISK CVEs")
    report_lines.append("-" * 70)
    top_risk = detailed_results.nlargest(10, 'risk_score')[
        ['CVE_ID', 'risk_score', 'risk_category', 'vulnerable_probability', 'exploit_probability']
    ]
    for idx, row in top_risk.iterrows():
        report_lines.append(f"{row['CVE_ID']}: Risk={row['risk_score']:.3f} | Category={row['risk_category']} | "
                          f"VulnProb={row['vulnerable_probability']:.3f} | ExploitProb={row['exploit_probability']:.3f}")
    report_lines.append("")
    
    report_lines.append("PREDICTION ERRORS ANALYSIS")
    report_lines.append("-" * 70)
    false_positives = detailed_results[(detailed_results['vulnerable_prediction'] == 1) & 
                                      (detailed_results['true_vulnerable'] == 0)]
    false_negatives = detailed_results[(detailed_results['vulnerable_prediction'] == 0) & 
                                      (detailed_results['true_vulnerable'] == 1)]
    
    report_lines.append(f"False Positives: {len(false_positives)}")
    if len(false_positives) > 0:
        report_lines.append("  Top 5 False Positives (highest confidence):")
        for idx, row in false_positives.nlargest(5, 'vulnerable_probability').iterrows():
            report_lines.append(f"    {row['CVE_ID']}: Confidence={row['vulnerable_probability']:.3f}")
    
    report_lines.append(f"\nFalse Negatives: {len(false_negatives)}")
    if len(false_negatives) > 0:
        report_lines.append("  Top 5 False Negatives (lowest confidence):")
        for idx, row in false_negatives.nsmallest(5, 'vulnerable_probability').iterrows():
            report_lines.append(f"    {row['CVE_ID']}: Confidence={row['vulnerable_probability']:.3f}")
    
    report_lines.append("")
    report_lines.append("="*70)
    report_lines.append("END OF REPORT")
    report_lines.append("="*70)
    
    # Save report
    report_path = os.path.join(eval_dir, "evaluation_report.txt")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n📄 Evaluation report saved to {report_path}")
    
    # Print report to console
    print("\n" + '\n'.join(report_lines))
    
    return {
        'classification_metrics': clf_metrics,
        'regression_metrics': reg_metrics,
        'detailed_results': detailed_results
    }

# ============================================================================
# FEATURE IMPORTANCE ANALYSIS (using SHAP - optional)
# ============================================================================

def analyze_feature_importance_shap(model_path, test_df, output_dir, n_samples=100):
    """
    SHAP analiza važnosti feature-a (zahteva shap biblioteku)
    """
    try:
        import shap
    except ImportError:
        print("\n⚠️  SHAP library not installed. Skipping feature importance analysis.")
        print("   Install with: pip install shap")
        return
    
    print("\n" + "="*70)
    print("🔍 SHAP FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    # Load model and data
    predictor = CVEVulnerabilityPredictor(model_path, output_dir)
    
    # Sample data for SHAP (computational expensive)
    test_sample = test_df.sample(n=min(n_samples, len(test_df)), random_state=42)
    
    print(f"\n📊 Analyzing {len(test_sample)} samples...")
    
    # Preprocess
    texts, tabular_features = predictor.preprocess_input(test_sample)
    
    # Create wrapper function for SHAP
    def model_predict(tabular_input):
        """Wrapper for SHAP - only uses tabular features"""
        batch_size = 32
        all_preds = []
        
        predictor.model.eval()
        with torch.no_grad():
            for i in range(0, len(tabular_input), batch_size):
                batch_tab = tabular_input[i:i+batch_size]
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                encoding = predictor.tokenizer(
                    batch_texts,
                    add_special_tokens=True,
                    max_length=Config.EMBEDDING_MAX_LENGTH,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(predictor.device)
                attention_mask = encoding['attention_mask'].to(predictor.device)
                tabular_tensor = torch.tensor(batch_tab, dtype=torch.float32).to(predictor.device)
                
                clf_logits, _ = predictor.model(input_ids, attention_mask, tabular_tensor)
                clf_probs = torch.sigmoid(clf_logits).cpu().numpy()
                
                all_preds.extend(clf_probs)
        
        return np.array(all_preds)
    
    # Create SHAP explainer
    print("\n🔄 Creating SHAP explainer (this may take a while)...")
    explainer = shap.KernelExplainer(model_predict, tabular_features[:100])
    
    # Calculate SHAP values
    print("🔄 Calculating SHAP values...")
    shap_values = explainer.shap_values(tabular_features[:50])
    
    # Plot SHAP summary
    eval_dir = os.path.join(output_dir, "evaluation")
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, tabular_features[:50], 
                     feature_names=predictor.feature_names,
                     show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "shap_summary.png"), 
                dpi=300, bbox_inches='tight')
    print(f"📊 SHAP summary plot saved")
    plt.close()
    
    # Feature importance bar plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, tabular_features[:50],
                     feature_names=predictor.feature_names,
                     plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "shap_importance.png"),
                dpi=300, bbox_inches='tight')
    print(f"📊 SHAP importance plot saved")
    plt.close()

# ============================================================================
# THRESHOLD OPTIMIZATION
# ============================================================================

def optimize_classification_threshold(y_true, y_pred_proba, output_dir):
    """
    Optimizacija threshold-a za klasifikaciju
    """
    print("\n" + "="*70)
    print("🎯 CLASSIFICATION THRESHOLD OPTIMIZATION")
    print("="*70)
    
    thresholds = np.arange(0.1, 0.9, 0.05)
    metrics = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    metrics_df = pd.DataFrame(metrics)
    
    # Find optimal thresholds
    best_f1_idx = metrics_df['f1'].idxmax()
    best_accuracy_idx = metrics_df['accuracy'].idxmax()
    
    print(f"\nBest F1-Score: {metrics_df.loc[best_f1_idx, 'f1']:.4f} at threshold {metrics_df.loc[best_f1_idx, 'threshold']:.2f}")
    print(f"Best Accuracy: {metrics_df.loc[best_accuracy_idx, 'accuracy']:.4f} at threshold {metrics_df.loc[best_accuracy_idx, 'threshold']:.2f}")
    
    # Plot metrics vs threshold
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(metrics_df['threshold'], metrics_df['accuracy'], 
            label='Accuracy', marker='o', linewidth=2)
    ax.plot(metrics_df['threshold'], metrics_df['precision'], 
            label='Precision', marker='s', linewidth=2)
    ax.plot(metrics_df['threshold'], metrics_df['recall'], 
            label='Recall', marker='^', linewidth=2)
    ax.plot(metrics_df['threshold'], metrics_df['f1'], 
            label='F1-Score', marker='d', linewidth=2)
    
    ax.axvline(x=metrics_df.loc[best_f1_idx, 'threshold'], 
               color='red', linestyle='--', alpha=0.7, 
               label=f"Best F1 Threshold ({metrics_df.loc[best_f1_idx, 'threshold']:.2f})")
    
    ax.set_xlabel('Classification Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Classification Metrics vs Threshold', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    eval_dir = os.path.join(output_dir, "evaluation")
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "threshold_optimization.png"),
                dpi=300, bbox_inches='tight')
    print(f"\n📊 Threshold optimization plot saved")
    plt.close()
    
    return metrics_df

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main evaluation function
    """
    model_path = os.path.join(Config.OUTPUT_DIR, "multitask_model.pt")
    test_csv_path = os.path.join(Config.OUTPUT_DIR, "X_test_full.csv")
    
    # Comprehensive evaluation
    eval_results = comprehensive_evaluation(model_path, test_csv_path, Config.OUTPUT_DIR)
    
    # Threshold optimization
    y_test_clf = load_pickle(os.path.join(Config.OUTPUT_DIR, "y_test_clf.pkl"))
    y_pred_proba = eval_results['detailed_results']['vulnerable_probability'].values
    
    optimize_classification_threshold(y_test_clf, y_pred_proba, Config.OUTPUT_DIR)
    
    # SHAP analysis (optional - comment out if too slow)
    # test_df = pd.read_csv(test_csv_path)
    # analyze_feature_importance_shap(model_path, test_df, Config.OUTPUT_DIR, n_samples=100)
    
    print("\n✅ Evaluation completed successfully!")
    print(f"\n📁 All results saved in: {os.path.join(Config.OUTPUT_DIR, 'evaluation')}")

if __name__ == "__main__":
    main()