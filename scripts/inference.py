import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from transfer_learning import CVEMultiTaskModel
from utils import load_pickle
from config import Config

# ============================================================================
# INFERENCE CLASS
# ============================================================================

class CVEVulnerabilityPredictor:
    """
    Klasa za inference - predviđanje ranjivosti na novim CVE podacima
    """
    
    def __init__(self, model_path, output_dir=Config.OUTPUT_DIR):
        """
        Args:
            model_path: putanja do sačuvanog modela (.pt fajl)
            output_dir: folder sa preprocessor objektima
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")
        
        # Load model checkpoint
        print(f"\n📦 Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model_config = checkpoint['config']
        self.model = CVEMultiTaskModel(
            model_name=model_config['model_name'],
            n_tabular_features=model_config['n_tabular_features'],
            dropout_rate=model_config['dropout_rate']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print("✅ Model loaded successfully")
        
        # Load tokenizer
        print(f"\n🔤 Loading tokenizer: {model_config['model_name']}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_config['model_name'])
        
        # Load preprocessors
        print("\n📂 Loading preprocessors...")
        self.imputer = load_pickle(os.path.join(output_dir, "imputer.pkl"))
        self.encoder = load_pickle(os.path.join(output_dir, "encoder.pkl"))
        self.scaler = load_pickle(os.path.join(output_dir, "scaler.pkl"))
        
        # Check if PCA was used
        pca_path = os.path.join(output_dir, "pca.pkl")
        if os.path.exists(pca_path):
            self.pca = load_pickle(pca_path)
            print("  ✅ PCA loaded")
        else:
            self.pca = None
            print("  ℹ️  PCA not used")
        
        # Load feature names
        self.feature_names = load_pickle(os.path.join(output_dir, "feature_names.pkl"))
        
        print(f"\n✅ Loaded {len(self.feature_names)} feature names")
    
    def preprocess_input(self, df):
        """
        Preprocesira novi input DataFrame na isti način kao u treningu
        
        Args:
            df: pandas DataFrame sa istim kolonama kao original dataset
            
        Returns:
            texts: lista tekstualnih opisa
            tabular_features: numpy array sa preprocesiranim tabularnim feature-ima
        """
        print("\n🔧 Preprocessing input data...")
        
        # 1. Handle missing values
        num_cols = ["baseScore", "exploitabilityScore", "impactScore", 
                    "days_since_published", "description_length", "description_words",
                    "num_references"]
        
        for col in num_cols:
            if col not in df.columns:
                df[col] = 0
        
        df[num_cols] = self.imputer.transform(df[num_cols])
        
        # 2. Categorical encoding
        cat_cols = ["attackVector", "attackComplexity", "privilegesRequired", 
                    "userInteraction", "scope", "confidentialityImpact", 
                    "integrityImpact", "availabilityImpact"]
        
        for col in cat_cols:
            if col not in df.columns:
                df[col] = "UNKNOWN"
            df[col] = df[col].fillna("UNKNOWN")
        
        cat_encoded = self.encoder.transform(df[cat_cols])
        feature_names_cat = self.encoder.get_feature_names_out(cat_cols)
        df_cat = pd.DataFrame(cat_encoded, columns=feature_names_cat, index=df.index)
        
        # 3. Scale numerical features
        num_cols_scaled = ["baseScore", "exploitabilityScore", "impactScore",
                          "days_since_published", "description_length", "description_words",
                          "num_references", "num_exploits"]
        
        for col in num_cols_scaled:
            if col not in df.columns:
                df[col] = 0
        
        scaled = self.scaler.transform(df[num_cols_scaled])
        df_scaled = pd.DataFrame(
            scaled,
            columns=[f"{col}_scaled" for col in num_cols_scaled],
            index=df.index
        )
        
        # 4. Keep additional features
        keep_cols = ['is_remote', 'requires_auth', 'requires_user_interaction',
                     'high_severity', 'critical_severity', 'is_critical_cwe',
                     'cvss_missing', 'year_published', 'month_published', 'quarter_published']
        
        for col in keep_cols:
            if col not in df.columns:
                df[col] = 0
        
        # 5. Combine all features
        X_combined = pd.concat([
            df[keep_cols].reset_index(drop=True),
            df_cat.reset_index(drop=True),
            df_scaled.reset_index(drop=True)
        ], axis=1)
        
        # Ensure all required features exist
        for feat in self.feature_names:
            if feat not in X_combined.columns:
                X_combined[feat] = 0
        
        tabular_features = X_combined[self.feature_names].values
        
        # 6. Extract text
        texts = df['description'].fillna("").astype(str).tolist() if 'description' in df.columns else [""] * len(df)
        
        print(f"✅ Preprocessed {len(df)} samples")
        
        return texts, tabular_features
    
    @torch.no_grad()
    def predict(self, df, batch_size=16):
        """
        Pravi predikciju na DataFrame-u
        
        Args:
            df: pandas DataFrame sa CVE podacima
            batch_size: batch size za inference
            
        Returns:
            DataFrame sa predikcijama
        """
        print("\n🎯 Making predictions...")
        
        # Preprocess
        texts, tabular_features = self.preprocess_input(df)
        
        all_clf_probs = []
        all_reg_values = []
        
        # Batch processing
        n_samples = len(texts)
        for i in range(0, n_samples, batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_tabular = tabular_features[i:i+batch_size]
            
            # Tokenize
            encoding = self.tokenizer(
                batch_texts,
                add_special_tokens=True,
                max_length=Config.EMBEDDING_MAX_LENGTH,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            tabular_tensor = torch.tensor(batch_tabular, dtype=torch.float32).to(self.device)
            
            # Forward pass
            clf_logits, reg_output = self.model(input_ids, attention_mask, tabular_tensor)
            
            # Convert to probabilities
            clf_probs = torch.sigmoid(clf_logits).cpu().numpy()
            reg_values = reg_output.cpu().numpy()
            
            all_clf_probs.extend(clf_probs)
            all_reg_values.extend(reg_values)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'CVE_ID': df['CVE_ID'].values if 'CVE_ID' in df.columns else [f"CVE-{i}" for i in range(len(df))],
            'vulnerable_probability': all_clf_probs,
            'vulnerable_prediction': (np.array(all_clf_probs) > 0.5).astype(int),
            'exploit_probability': all_reg_values
        })
        
        # Add risk score (combined metric)
        results['risk_score'] = (results['vulnerable_probability'] * 0.5 + 
                                results['exploit_probability'] * 0.5)
        
        # Add risk category
        results['risk_category'] = pd.cut(
            results['risk_score'],
            bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        print(f"✅ Generated predictions for {len(results)} samples")
        
        return results
    
    def predict_single(self, cve_data):
        """
        Predviđanje za jedan CVE zapis
        
        Args:
            cve_data: dictionary sa CVE atributima
            
        Returns:
            dictionary sa predikcijama
        """
        df = pd.DataFrame([cve_data])
        results = self.predict(df)
        return results.iloc[0].to_dict()

# ============================================================================
# BATCH INFERENCE FUNCTION
# ============================================================================

def batch_inference(input_csv, model_path, output_csv, batch_size=16):
    """
    Batch inference na CSV fajlu
    
    Args:
        input_csv: putanja do input CSV fajla
        model_path: putanja do istreniranog modela
        output_csv: putanja za čuvanje rezultata
        batch_size: batch size za inference
    """
    print("\n" + "="*70)
    print("🔮 CVE VULNERABILITY BATCH INFERENCE")
    print("="*70)
    
    # Load data
    print(f"\n📂 Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"✅ Loaded {len(df)} CVE records")
    
    # Initialize predictor
    predictor = CVEVulnerabilityPredictor(model_path)
    
    # Make predictions
    results = predictor.predict(df, batch_size=batch_size)
    
    # Merge with original data
    df_output = pd.concat([df, results], axis=1)
    
    # Save results
    df_output.to_csv(output_csv, index=False)
    print(f"\n💾 Results saved to {output_csv}")
    
    # Print summary statistics
    print("\n📊 PREDICTION SUMMARY")
    print("="*70)
    print(f"Total CVEs analyzed: {len(results)}")
    print(f"\nVulnerable predictions:")
    print(results['vulnerable_prediction'].value_counts())
    print(f"\nRisk categories:")
    print(results['risk_category'].value_counts().sort_index())
    print(f"\nTop 10 highest risk CVEs:")
    print(results.nlargest(10, 'risk_score')[['CVE_ID', 'risk_score', 'risk_category']])
    
    return df_output

# ============================================================================
# INTERACTIVE SINGLE PREDICTION
# ============================================================================

def predict_new_cve():
    """
    Interaktivna funkcija za predviđanje jednog CVE-a
    """
    print("\n" + "="*70)
    print("🔍 SINGLE CVE VULNERABILITY PREDICTION")
    print("="*70)
    
    # Gather input
    cve_data = {}
    
    print("\nEnter CVE information (press Enter for default values):\n")
    
    cve_data['CVE_ID'] = input("CVE ID (e.g., CVE-2024-1234): ") or "CVE-2024-XXXX"
    cve_data['description'] = input("Description: ") or "No description available"
    
    # CVSS scores
    cve_data['baseScore'] = float(input("Base Score (0-10) [7.5]: ") or 7.5)
    cve_data['exploitabilityScore'] = float(input("Exploitability Score (0-10) [3.9]: ") or 3.9)
    cve_data['impactScore'] = float(input("Impact Score (0-10) [5.9]: ") or 5.9)
    
    # Attack vector
    print("\nAttack Vector: 1=NETWORK, 2=ADJACENT_NETWORK, 3=LOCAL, 4=PHYSICAL")
    av_choice = input("Choice [1]: ") or "1"
    av_map = {"1": "NETWORK", "2": "ADJACENT_NETWORK", "3": "LOCAL", "4": "PHYSICAL"}
    cve_data['attackVector'] = av_map.get(av_choice, "NETWORK")
    
    # Attack complexity
    print("\nAttack Complexity: 1=LOW, 2=HIGH")
    ac_choice = input("Choice [1]: ") or "1"
    cve_data['attackComplexity'] = "LOW" if ac_choice == "1" else "HIGH"
    
    # Privileges required
    print("\nPrivileges Required: 1=NONE, 2=LOW, 3=HIGH")
    pr_choice = input("Choice [1]: ") or "1"
    pr_map = {"1": "NONE", "2": "LOW", "3": "HIGH"}
    cve_data['privilegesRequired'] = pr_map.get(pr_choice, "NONE")
    
    # User interaction
    print("\nUser Interaction: 1=NONE, 2=REQUIRED")
    ui_choice = input("Choice [1]: ") or "1"
    cve_data['userInteraction'] = "NONE" if ui_choice == "1" else "REQUIRED"
    
    # Impact scores
    print("\nImpact (1=NONE, 2=LOW, 3=HIGH):")
    conf = input("Confidentiality Impact [3]: ") or "3"
    integ = input("Integrity Impact [3]: ") or "3"
    avail = input("Availability Impact [3]: ") or "3"
    
    impact_map = {"1": "NONE", "2": "LOW", "3": "HIGH"}
    cve_data['confidentialityImpact'] = impact_map.get(conf, "HIGH")
    cve_data['integrityImpact'] = impact_map.get(integ, "HIGH")
    cve_data['availabilityImpact'] = impact_map.get(avail, "HIGH")
    
    # Scope
    print("\nScope: 1=UNCHANGED, 2=CHANGED")
    scope_choice = input("Choice [1]: ") or "1"
    cve_data['scope'] = "UNCHANGED" if scope_choice == "1" else "CHANGED"
    
    # Additional fields
    cve_data['num_references'] = int(input("\nNumber of references [5]: ") or 5)
    cve_data['num_exploits'] = int(input("Number of known exploits [0]: ") or 0)
    cve_data['days_since_published'] = int(input("Days since published [30]: ") or 30)
    
    # Derived features
    cve_data['description_length'] = len(cve_data['description'])
    cve_data['description_words'] = len(cve_data['description'].split())
    cve_data['is_remote'] = 1 if cve_data['attackVector'] == 'NETWORK' else 0
    cve_data['requires_auth'] = 0 if cve_data['privilegesRequired'] == 'NONE' else 1
    cve_data['requires_user_interaction'] = 1 if cve_data['userInteraction'] == 'REQUIRED' else 0
    cve_data['high_severity'] = 1 if cve_data['baseScore'] >= 7.0 else 0
    cve_data['critical_severity'] = 1 if cve_data['baseScore'] >= 9.0 else 0
    cve_data['is_critical_cwe'] = 0
    cve_data['cvss_missing'] = 0
    cve_data['year_published'] = 2024
    cve_data['month_published'] = 1
    cve_data['quarter_published'] = 1
    
    # Load model and predict
    model_path = os.path.join(Config.OUTPUT_DIR, "multitask_model.pt")
    predictor = CVEVulnerabilityPredictor(model_path)
    
    result = predictor.predict_single(cve_data)
    
    # Display results
    print("\n" + "="*70)
    print("📊 PREDICTION RESULTS")
    print("="*70)
    print(f"\nCVE ID: {result['CVE_ID']}")
    print(f"\n🎯 Vulnerability Classification:")
    print(f"   Prediction: {'VULNERABLE' if result['vulnerable_prediction'] == 1 else 'NOT VULNERABLE'}")
    print(f"   Confidence: {result['vulnerable_probability']:.2%}")
    
    print(f"\n📈 Exploit Probability:")
    print(f"   Score: {result['exploit_probability']:.2%}")
    
    print(f"\n⚠️  Overall Risk Assessment:")
    print(f"   Risk Score: {result['risk_score']:.3f}")
    print(f"   Risk Category: {result['risk_category']}")
    
    # Risk interpretation
    if result['risk_category'] == 'Critical':
        print("\n🚨 CRITICAL: Immediate action required! High likelihood of exploitation.")
    elif result['risk_category'] == 'High':
        print("\n⚠️  HIGH RISK: Prioritize patching this vulnerability.")
    elif result['risk_category'] == 'Medium':
        print("\n⚡ MEDIUM RISK: Schedule patching in upcoming maintenance window.")
    else:
        print("\n✅ LOW RISK: Monitor but not urgent.")
    
    return result

# ============================================================================
# MAIN - EXAMPLE USAGE
# ============================================================================

def main():
    """
    Primeri korišćenja inference funkcija
    """
    import sys
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python inference.py batch <input.csv> <output.csv>")
        print("  python inference.py single")
        print("  python inference.py test")
        return
    
    mode = sys.argv[1]
    
    if mode == "batch":
        if len(sys.argv) < 4:
            print("Error: batch mode requires input and output CSV paths")
            return
        
        input_csv = sys.argv[2]
        output_csv = sys.argv[3]
        model_path = os.path.join(Config.OUTPUT_DIR, "multitask_model.pt")
        
        batch_inference(input_csv, model_path, output_csv)
    
    elif mode == "single":
        predict_new_cve()
    
    elif mode == "test":
        # Test na test setu
        print("\n🧪 Testing on test set...")
        test_df = pd.read_csv(os.path.join(Config.OUTPUT_DIR, "X_test_full.csv"))
        
        # Take sample for quick test
        test_sample = test_df.head(100)
        
        model_path = os.path.join(Config.OUTPUT_DIR, "multitask_model.pt")
        predictor = CVEVulnerabilityPredictor(model_path)
        
        results = predictor.predict(test_sample)
        
        # Show results
        print("\n📊 Test Results Sample:")
        print(results.head(10))
        
        # If we have ground truth
        if 'vulnerable' in test_sample.columns:
            from sklearn.metrics import accuracy_score, classification_report
            
            y_true = test_sample['vulnerable'].values
            y_pred = results['vulnerable_prediction'].values
            
            print("\n🎯 Classification Performance:")
            print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, 
                                       target_names=['Not Vulnerable', 'Vulnerable']))

if __name__ == "__main__":
    main()