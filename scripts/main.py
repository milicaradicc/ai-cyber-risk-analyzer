import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import pickle
from utils import ensure_dir, save_pickle, load_pickle
from config import *
# ML imports
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# Deep learning imports
import torch
from transformers import AutoTokenizer, AutoModel

# Warnings
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# DATA LOADING
# ============================================================================

def parse_cvss(metrics_dict):
    """Parsira CVSS metrike iz metrics dictionary-ja"""
    if not metrics_dict:
        return {
            "baseScore": None,
            "attackVector": None,
            "attackComplexity": None,
            "privilegesRequired": None,
            "userInteraction": None,
            "scope": None,
            "confidentialityImpact": None,
            "integrityImpact": None,
            "availabilityImpact": None,
            "exploitabilityScore": None,
            "impactScore": None,
        }
    
    # Pokušaj sa v31, pa v30, pa v2
    cvss_metric = (metrics_dict.get("cvssMetricV31") or 
                   metrics_dict.get("cvssMetricV30") or 
                   metrics_dict.get("cvssMetricV2") or [])
    
    if not cvss_metric:
        return {
            "baseScore": None,
            "attackVector": None,
            "attackComplexity": None,
            "privilegesRequired": None,
            "userInteraction": None,
            "scope": None,
            "confidentialityImpact": None,
            "integrityImpact": None,
            "availabilityImpact": None,
            "exploitabilityScore": None,
            "impactScore": None,
        }
    
    cvss = cvss_metric[0].get("cvssData", {})
    
    # Handle CVSS v2 differences
    return {
        "baseScore": cvss.get("baseScore"),
        "attackVector": cvss.get("attackVector") or cvss.get("accessVector"),
        "attackComplexity": cvss.get("attackComplexity") or cvss.get("accessComplexity"),
        "privilegesRequired": cvss.get("privilegesRequired") or cvss.get("authentication"),
        "userInteraction": cvss.get("userInteraction"),
        "scope": cvss.get("scope"),
        "confidentialityImpact": cvss.get("confidentialityImpact"),
        "integrityImpact": cvss.get("integrityImpact"),
        "availabilityImpact": cvss.get("availabilityImpact"),
        "exploitabilityScore": cvss_metric[0].get("exploitabilityScore"),
        "impactScore": cvss_metric[0].get("impactScore"),
    }

def load_nvd_data():
    """Učitava sve NVD JSON fajlove"""
    print("\n" + "="*70)
    print("📂 UČITAVANJE NVD PODATAKA")
    print("="*70)
    
    # Proveri da li data folder postoji
    if not os.path.exists(Config.NVD_BASE_FOLDER):
        print(f"❌ GREŠKA: Folder '{Config.NVD_BASE_FOLDER}' ne postoji!")
        print(f"   Trenutni direktorijum: {os.getcwd()}")
        print(f"   Sadržaj direktorijuma:")
        for item in os.listdir('.'):
            print(f"     - {item}")
        raise FileNotFoundError(f"Folder {Config.NVD_BASE_FOLDER} ne postoji")
    
    print(f"✅ Folder '{Config.NVD_BASE_FOLDER}' pronađen")
    print(f"\n📋 Sadržaj '{Config.NVD_BASE_FOLDER}' foldera:")
    for item in os.listdir(Config.NVD_BASE_FOLDER):
        item_path = os.path.join(Config.NVD_BASE_FOLDER, item)
        if os.path.isdir(item_path):
            print(f"   📁 {item}/")
        else:
            print(f"   📄 {item}")
    
    all_cves = []
    parsed_files = 0
    failed_files = 0
    
    for year in range(Config.START_YEAR, Config.END_YEAR):
        year_folder = os.path.join(Config.NVD_BASE_FOLDER, f"CVE-{year}")
        if not os.path.exists(year_folder):
            continue
        
        json_files = []
        # Rekurzivno prolazi kroz sve subfoldere (npr. CVE-2022-00xx)
        for root, dirs, files in os.walk(year_folder):
            for f in files:
                if f.endswith(".json"):
                    json_files.append(os.path.join(root, f))
        
        if len(json_files) == 0:
            print(f"\n⚠️  CVE-{year}: Folder postoji ali nema JSON fajlova")
            print(f"   Struktura foldera:")
            for root, dirs, files in os.walk(year_folder):
                level = root.replace(year_folder, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files[:5]:  # Prikaži prvih 5 fajlova
                    print(f"{subindent}{file}")
                if len(files) > 5:
                    print(f"{subindent}... i još {len(files)-5} fajlova")
            continue
        
        print(f"\n📁 CVE-{year}: Pronađeno {len(json_files)} JSON fajlova")
        if json_files:
            print(f"   Primer putanje: {json_files[0]}")
        
        for file_path in tqdm(json_files, desc=f"  Parsiranje {year}"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    cve_id = data.get("id")
                    if not cve_id:
                        continue
                    
                    descriptions = data.get("descriptions", [])
                    description_en = next((d["value"] for d in descriptions if d["lang"] == "en"), "")
                    
                    published = data.get("published")
                    lastModified = data.get("lastModified")
                    
                    # Parsiranje CVSS metrika
                    metrics = parse_cvss(data.get("metrics", {}))
                    
                    # Ekstrakcija CWE
                    weaknesses = data.get("weaknesses", [])
                    cwe = None
                    if weaknesses:
                        cwe_desc = weaknesses[0].get("description", [])
                        if cwe_desc:
                            cwe = cwe_desc[0].get("value")
                    
                    # Broj referenci
                    references = data.get("references", [])
                    num_references = len(references)
                    
                    all_cves.append({
                        "CVE_ID": cve_id,
                        "description": description_en,
                        "published": published,
                        "lastModified": lastModified,
                        "CWE": cwe,
                        "num_references": num_references,
                        **metrics
                    })
                    parsed_files += 1
                    
            except Exception as e:
                failed_files += 1
                print(f"\n⚠️  Greška u fajlu {file_path}: {str(e)}")
                continue
    
    print(f"\n✅ Uspešno parsiranih fajlova: {parsed_files}")
    print(f"❌ Neuspešnih fajlova: {failed_files}")
    
    if parsed_files == 0:
        print("\n❌ KRITIČNA GREŠKA: Nijedan JSON fajl nije uspešno parsiran!")
        print("\n💡 Moguća rešenja:")
        print("   1. Proverite strukturu 'data' foldera")
        print("   2. Proverite da li JSON fajlovi imaju ispravan format")
        print("   3. Pokrenite download skriptu ponovo")
        raise ValueError("Nema parsiranih CVE podataka")
    
    df_cve = pd.DataFrame(all_cves)
    print(f"📊 Ukupno CVE zapisa: {len(df_cve)}")
    
    return df_cve

def load_exploit_data():
    """Učitava ExploitDB podatke"""
    print("\n" + "="*70)
    print("📂 UČITAVANJE EXPLOITDB PODATAKA")
    print("="*70)
    
    if not os.path.exists(Config.EXPLOIT_CSV_PATH):
        raise FileNotFoundError(f"❌ Fajl ne postoji: {Config.EXPLOIT_CSV_PATH}")
    
    df_exploit = pd.read_csv(Config.EXPLOIT_CSV_PATH)
    print(f"✅ Učitano {len(df_exploit)} exploit zapisa")
    
    # Provera kolona
    if 'codes' not in df_exploit.columns:
        if 'aliases' in df_exploit.columns:
            df_exploit['codes'] = df_exploit['aliases']
        else:
            raise KeyError("❌ Nema ni 'codes' ni 'aliases' kolone!")
    
    def extract_cve_list(codes):
        if pd.isna(codes):
            return []
        if isinstance(codes, str):
            codes = codes.replace(',', ';')
            return [c.strip() for c in codes.split(";") if c.strip().startswith("CVE-")]
        return []
    
    df_exploit["CVE_list"] = df_exploit["codes"].apply(extract_cve_list)
    df_exploit = df_exploit.explode("CVE_list")
    df_exploit = df_exploit[df_exploit["CVE_list"].notna()]
    
    # Dodaj datum exploita
    if 'date_published' in df_exploit.columns:
        df_exploit['exploit_date'] = pd.to_datetime(df_exploit['date_published'], errors='coerce')
    
    print(f"📊 Ukupno CVE-exploit mapiranja: {len(df_exploit)}")
    
    return df_exploit

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def merge_and_engineer_features(df_cve, df_exploit):
    """Spaja podatke i kreira nove feature-e"""
    print("\n" + "="*70)
    print("🔗 SPAJANJE I KREIRANJE FEATURE-A")
    print("="*70)
    
    # Brojanje exploita po CVE-u
    df_exploit_grouped = df_exploit.groupby("CVE_list").agg({
        'CVE_list': 'count',
        'exploit_date': 'min'  # Najraniji exploit
    }).rename(columns={'CVE_list': 'num_exploits', 'exploit_date': 'first_exploit_date'})
    
    # Merge
    df_merged = pd.merge(df_cve, df_exploit_grouped, 
                        how="left", left_on="CVE_ID", right_index=True)
    
    df_merged["num_exploits"] = df_merged["num_exploits"].fillna(0).astype(int)
    df_merged["vulnerable"] = (df_merged["num_exploits"] > 0).astype(int)
    
    print(f"✅ Spojeni podaci: {len(df_merged)} redova")
    print(f"\n📊 Distribucija 'vulnerable':")
    print(df_merged["vulnerable"].value_counts())
    
    # TEMPORAL FEATURES
    df_merged["published"] = pd.to_datetime(df_merged["published"], errors='coerce')
    df_merged["first_exploit_date"] = pd.to_datetime(df_merged["first_exploit_date"], errors='coerce')
    
    df_merged["year_published"] = df_merged["published"].dt.year
    df_merged["month_published"] = df_merged["published"].dt.month
    df_merged["quarter_published"] = df_merged["published"].dt.quarter
    df_merged["day_of_week"] = df_merged["published"].dt.dayofweek
    df_merged["days_since_published"] = (datetime.now() - df_merged["published"]).dt.days
    
    # Time to exploit (ako postoji)
    df_merged["days_to_exploit"] = (
        df_merged["first_exploit_date"] - df_merged["published"]
    ).dt.days
    
    # DERIVED FEATURES
    df_merged['description_length'] = df_merged['description'].str.len()
    df_merged['description_words'] = df_merged['description'].str.split().str.len()
    
    # Binary indicators
    df_merged['is_remote'] = (df_merged['attackVector'] == 'NETWORK').astype(int)
    df_merged['requires_auth'] = (df_merged['privilegesRequired'] != 'NONE').astype(int)
    df_merged['requires_user_interaction'] = (df_merged['userInteraction'] == 'REQUIRED').astype(int)
    
    # Severity categories
    df_merged['high_severity'] = (df_merged['baseScore'] >= 7.0).astype(int)
    df_merged['critical_severity'] = (df_merged['baseScore'] >= 9.0).astype(int)
    
    # CWE criticality
    df_merged['is_critical_cwe'] = df_merged['CWE'].isin(Config.CRITICAL_CWES).astype(int)
    
    # Missing data indicators
    df_merged['cvss_missing'] = df_merged['baseScore'].isna().astype(int)
    
    print("\n✅ Kreirano dodatnih feature-a")
    
    return df_merged

def calculate_exploit_probability(df):
    """Sofisticiranija formula za exploit probability"""
    print("\n📊 Računanje exploit_probability...")
    
    def calc_prob(row):
        prob = 0.0
        
        # 1. Broj exploita (40%)
        if row['num_exploits'] > 0:
            prob += min(0.4, row['num_exploits'] * 0.1)
        
        # 2. CVSS base score (30%)
        if pd.notna(row['baseScore']):
            prob += (row['baseScore'] / 10) * 0.3
        
        # 3. Attack vector (15%)
        av_weights = {
            'NETWORK': 0.15,
            'ADJACENT_NETWORK': 0.10,
            'LOCAL': 0.05,
            'PHYSICAL': 0.02
        }
        prob += av_weights.get(row['attackVector'], 0.05)
        
        # 4. CWE severity (10%)
        if row['is_critical_cwe'] == 1:
            prob += 0.10
        
        # 5. Starost (5%)
        if pd.notna(row['days_since_published']) and row['days_since_published'] > 0:
            age_factor = min(row['days_since_published'] / 365, 5) / 5
            prob += age_factor * 0.05
        
        return min(prob, 1.0)
    
    df['exploit_probability'] = df.apply(calc_prob, axis=1)
    
    print(f"✅ exploit_probability statistics:")
    print(df['exploit_probability'].describe())
    
    return df

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def handle_missing_values(df):
    """Obrada nedostajućih vrednosti"""
    print("\n" + "="*70)
    print("🔧 OBRADA NEDOSTAJUĆIH VREDNOSTI")
    print("="*70)
    
    # Numeričke kolone - median imputation
    num_cols = ["baseScore", "exploitabilityScore", "impactScore", 
                "days_since_published", "description_length", "description_words",
                "num_references"]
    
    imputer = SimpleImputer(strategy='median')
    df[num_cols] = imputer.fit_transform(df[num_cols])
    
    # Kategorijske - fill sa UNKNOWN
    cat_cols = ["attackVector", "attackComplexity", "privilegesRequired", 
                "userInteraction", "scope", "confidentialityImpact", 
                "integrityImpact", "availabilityImpact", "CWE"]
    
    for col in cat_cols:
        df[col] = df[col].fillna("UNKNOWN")
    
    print("✅ Nedostajuće vrednosti obrađene")
    
    return df, imputer

def encode_categorical_features(df_train, df_test=None, encoder=None):
    """One-hot encoding kategorijskih feature-a"""
    print("\n🔢 One-hot encoding kategorijskih feature-a...")
    
    cat_cols = ["attackVector", "attackComplexity", "privilegesRequired", 
                "userInteraction", "scope", "confidentialityImpact", 
                "integrityImpact", "availabilityImpact"]
    
    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        cat_encoded_train = encoder.fit_transform(df_train[cat_cols])
    else:
        cat_encoded_train = encoder.transform(df_train[cat_cols])
    
    feature_names = encoder.get_feature_names_out(cat_cols)
    df_cat_train = pd.DataFrame(cat_encoded_train, columns=feature_names, index=df_train.index)
    
    if df_test is not None:
        cat_encoded_test = encoder.transform(df_test[cat_cols])
        df_cat_test = pd.DataFrame(cat_encoded_test, columns=feature_names, index=df_test.index)
        return df_cat_train, df_cat_test, encoder
    
    return df_cat_train, encoder

def scale_numerical_features(df_train, df_test=None, scaler=None):
    """Standardizacija numeričkih feature-a"""
    print("\n📏 Standardizacija numeričkih feature-a...")
    
    num_cols = ["baseScore", "exploitabilityScore", "impactScore",
                "days_since_published", "description_length", "description_words",
                "num_references", "num_exploits"]
    
    if scaler is None:
        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(df_train[num_cols])
    else:
        scaled_train = scaler.transform(df_train[num_cols])
    
    df_scaled_train = pd.DataFrame(
        scaled_train, 
        columns=[f"{col}_scaled" for col in num_cols],
        index=df_train.index
    )
    
    if df_test is not None:
        scaled_test = scaler.transform(df_test[num_cols])
        df_scaled_test = pd.DataFrame(
            scaled_test,
            columns=[f"{col}_scaled" for col in num_cols],
            index=df_test.index
        )
        return df_scaled_train, df_scaled_test, scaler
    
    return df_scaled_train, scaler

# ============================================================================
# TEXT EMBEDDINGS
# ============================================================================

def generate_embeddings_batch(texts, model, tokenizer):
    """Batch processing za brže generisanje embeddinga"""
    print("\n🤖 Generisanje text embeddinga...")
    print(f"  Model: {Config.EMBEDDING_MODEL}")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Batch size: {Config.EMBEDDING_BATCH_SIZE}")
    
    model.eval()
    model.to(Config.DEVICE)
    
    all_embeddings = []
    texts_list = texts.tolist() if isinstance(texts, pd.Series) else texts
    
    for i in tqdm(range(0, len(texts_list), Config.EMBEDDING_BATCH_SIZE), 
                  desc="  Generisanje embeddinga"):
        batch = texts_list[i:i+Config.EMBEDDING_BATCH_SIZE]
        
        # Handle empty or None texts
        batch = [str(text) if text and not pd.isna(text) else "" for text in batch]
        
        with torch.no_grad():
            inputs = tokenizer(batch, return_tensors="pt", 
                             truncation=True, 
                             max_length=Config.EMBEDDING_MAX_LENGTH,
                             padding=True)
            
            inputs = {k: v.to(Config.DEVICE) for k, v in inputs.items()}
            outputs = model(**inputs)
            
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            all_embeddings.append(embeddings)
    
    final_embeddings = np.vstack(all_embeddings)
    print(f"✅ Generisano {final_embeddings.shape[0]} embeddinga dimenzije {final_embeddings.shape[1]}")
    
    return final_embeddings

def reduce_embedding_dimensions(embeddings_train, embeddings_test=None, pca=None):
    """PCA za smanjenje dimenzionalnosti embeddinga"""
    if not Config.USE_PCA:
        return embeddings_train, embeddings_test, None
    
    print(f"\n📉 PCA redukcija dimenzija: {embeddings_train.shape[1]} → {Config.PCA_COMPONENTS}")
    
    if pca is None:
        pca = PCA(n_components=Config.PCA_COMPONENTS, random_state=Config.RANDOM_STATE)
        embeddings_reduced_train = pca.fit_transform(embeddings_train)
        print(f"  Objašnjeno {pca.explained_variance_ratio_.sum():.2%} varijanse")
    else:
        embeddings_reduced_train = pca.transform(embeddings_train)
    
    if embeddings_test is not None:
        embeddings_reduced_test = pca.transform(embeddings_test)
        return embeddings_reduced_train, embeddings_reduced_test, pca
    
    return embeddings_reduced_train, pca

# ============================================================================
# CLASS BALANCING
# ============================================================================

def balance_classes(X_train, y_train):
    """SMOTE za balansiranje klasa"""
    if not Config.USE_SMOTE:
        return X_train, y_train
    
    print("\n⚖️  Balansiranje klasa pomoću SMOTE...")
    print(f"  Pre balansiranja: {pd.Series(y_train).value_counts().to_dict()}")
    
    smote = SMOTE(sampling_strategy=Config.SMOTE_SAMPLING_STRATEGY, 
                  random_state=Config.RANDOM_STATE)
    
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"  Nakon balansiranja: {pd.Series(y_balanced).value_counts().to_dict()}")
    
    return X_balanced, y_balanced

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("\n" + "="*70)
    print("🚀 CVE VULNERABILITY PREPROCESSING PIPELINE")
    print("="*70)
    
    # Kreiraj output direktorijum
    ensure_dir(Config.OUTPUT_DIR)
    
    # 1. LOAD DATA
    df_cve = load_nvd_data()
    df_exploit = load_exploit_data()
    
    # 2. MERGE & FEATURE ENGINEERING
    df_merged = merge_and_engineer_features(df_cve, df_exploit)
    df_merged = calculate_exploit_probability(df_merged)
    
    # 3. HANDLE MISSING VALUES
    df_merged, imputer = handle_missing_values(df_merged)
    
    # Sačuvaj raw merged data
    df_merged.to_csv(os.path.join(Config.OUTPUT_DIR, "merged_data_raw.csv"), index=False)
    print(f"\n💾 Sačuvan raw merged data")
    
    # 4. TRAIN/VALIDATION/TEST SPLIT
    print("\n" + "="*70)
    print("✂️  PODELA PODATAKA")
    print("="*70)
    
    # Prvo izdvoj test set
    df_train_val, df_test = train_test_split(
        df_merged,
        test_size=Config.TEST_SIZE,
        stratify=df_merged['vulnerable'],
        random_state=Config.RANDOM_STATE
    )
    
    # Zatim podeli train_val na train i validation
    val_size_adjusted = Config.VALIDATION_SIZE / (1 - Config.TEST_SIZE)
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_size_adjusted,
        stratify=df_train_val['vulnerable'],
        random_state=Config.RANDOM_STATE
    )
    
    print(f"  Train set: {len(df_train)} ({len(df_train)/len(df_merged)*100:.1f}%)")
    print(f"  Validation set: {len(df_val)} ({len(df_val)/len(df_merged)*100:.1f}%)")
    print(f"  Test set: {len(df_test)} ({len(df_test)/len(df_merged)*100:.1f}%)")
    
    # 5. ENCODE CATEGORICAL FEATURES
    df_cat_train, df_cat_val, encoder = encode_categorical_features(df_train, df_val)
    df_cat_test, _ = encode_categorical_features(df_test, encoder=encoder) 

    # 6. SCALE NUMERICAL FEATURES  
    df_num_train, df_num_val, scaler = scale_numerical_features(df_train, df_val)
    df_num_test, _ = scale_numerical_features(df_test, scaler=scaler)  

    # 7. GENERATE TEXT EMBEDDINGS
    print("\n" + "="*70)
    print("📝 TEXT EMBEDDINGS")
    print("="*70)
    
    tokenizer = AutoTokenizer.from_pretrained(Config.EMBEDDING_MODEL)
    model = AutoModel.from_pretrained(Config.EMBEDDING_MODEL)
    
    embeddings_train = generate_embeddings_batch(df_train['description'], model, tokenizer)
    embeddings_val = generate_embeddings_batch(df_val['description'], model, tokenizer)
    embeddings_test = generate_embeddings_batch(df_test['description'], model, tokenizer)
    
    # 8. DIMENSIONALITY REDUCTION
    embeddings_train, embeddings_val, pca = reduce_embedding_dimensions(
        embeddings_train, embeddings_val
    )
    embeddings_test, _, _ = reduce_embedding_dimensions(embeddings_test, pca=pca)

    # 💾 DODAJ OVO - Sačuvaj embeddings
    save_pickle(embeddings_train, os.path.join(Config.OUTPUT_DIR, "embeddings_train.pkl"))
    save_pickle(embeddings_val, os.path.join(Config.OUTPUT_DIR, "embeddings_val.pkl"))
    save_pickle(embeddings_test, os.path.join(Config.OUTPUT_DIR, "embeddings_test.pkl"))
    print(f"\n💾 Sačuvani text embeddings")

    # Convert to DataFrame
    embed_cols = [f"embed_{i}" for i in range(embeddings_train.shape[1])]
    df_embed_train = pd.DataFrame(embeddings_train, columns=embed_cols, index=df_train.index)
    df_embed_val = pd.DataFrame(embeddings_val, columns=embed_cols, index=df_val.index)
    df_embed_test = pd.DataFrame(embeddings_test, columns=embed_cols, index=df_test.index)
    
    # 9. COMBINE ALL FEATURES
    print("\n🔗 Kombinovanje svih feature-a...")
    
    # Select relevant original features
    keep_cols = ['CVE_ID', 'vulnerable', 'exploit_probability', 'num_exploits',
                 'is_remote', 'requires_auth', 'requires_user_interaction',
                 'high_severity', 'critical_severity', 'is_critical_cwe',
                 'cvss_missing', 'year_published', 'month_published', 'quarter_published']
    
    X_train_final = pd.concat([
        df_train[keep_cols].reset_index(drop=True),
        df_cat_train.reset_index(drop=True),
        df_num_train.reset_index(drop=True),
        df_embed_train.reset_index(drop=True)
    ], axis=1)
    
    X_val_final = pd.concat([
        df_val[keep_cols].reset_index(drop=True),
        df_cat_val.reset_index(drop=True),
        df_num_val.reset_index(drop=True),
        df_embed_val.reset_index(drop=True)
    ], axis=1)
    
    X_test_final = pd.concat([
        df_test[keep_cols].reset_index(drop=True),
        df_cat_test.reset_index(drop=True),
        df_num_test.reset_index(drop=True),
        df_embed_test.reset_index(drop=True)
    ], axis=1)
    
    print(f"✅ Train set: {X_train_final.shape}")
    print(f"✅ Validation set: {X_val_final.shape}")
    print(f"✅ Test set: {X_test_final.shape}")

    # ========================================================================
    # 10. PREPARE TARGETS
    # ========================================================================
    y_train_clf = X_train_final['vulnerable'].values
    y_train_reg = X_train_final['exploit_probability'].values
    
    y_val_clf = X_val_final['vulnerable'].values
    y_val_reg = X_val_final['exploit_probability'].values
    
    y_test_clf = X_test_final['vulnerable'].values
    y_test_reg = X_test_final['exploit_probability'].values
    
    # Remove target columns from features
    feature_cols = [col for col in X_train_final.columns 
                   if col not in ['CVE_ID', 'vulnerable', 'exploit_probability']]
    
    X_train = X_train_final[feature_cols].values
    X_val = X_val_final[feature_cols].values
    X_test = X_test_final[feature_cols].values
    
    # ========================================================================
    #
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Prekinuto od strane korisnika (KeyboardInterrupt).")
    except Exception as e:
        print(f"\n❌ Došlo je do greške tokom izvođenja: {e}")
        raise
