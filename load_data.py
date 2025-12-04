import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from config import *
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import torch
import warnings

warnings.filterwarnings('ignore')

def parse_cvss(metrics_dict):
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
    if not os.path.exists(Config.NVD_BASE_FOLDER):
        raise FileNotFoundError(f"Folder {Config.NVD_BASE_FOLDER} doesnt exist")

    all_cves = []
    parsed_files = 0
    failed_files = 0

    for year in range(Config.START_YEAR, Config.END_YEAR):
        year_folder = os.path.join(Config.NVD_BASE_FOLDER, f"CVE-{year}")
        if not os.path.exists(year_folder):
            continue

        json_files = []
        for root, dirs, files in os.walk(year_folder):
            for f in files:
                if f.endswith(".json"):
                    json_files.append(os.path.join(root, f))

        if len(json_files) == 0:
            for root, dirs, files in os.walk(year_folder):
                level = root.replace(year_folder, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
            continue

        for file_path in tqdm(json_files, desc=f"  Parsing {year}"):
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

                    metrics = parse_cvss(data.get("metrics", {}))

                    weaknesses = data.get("weaknesses", [])
                    cwe = None
                    if weaknesses:
                        cwe_desc = weaknesses[0].get("description", [])
                        if cwe_desc:
                            cwe = cwe_desc[0].get("value")

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
                print(f"\n Error in file {file_path}: {str(e)}")
                continue

    if parsed_files == 0:
        raise ValueError("No parsed data")

    df_cve = pd.DataFrame(all_cves)
    return df_cve

def load_exploit_data():
    if not os.path.exists(Config.EXPLOIT_CSV_PATH):
        raise FileNotFoundError(f"File doesnt exist: {Config.EXPLOIT_CSV_PATH}")

    df_exploit = pd.read_csv(Config.EXPLOIT_CSV_PATH)

    if 'codes' not in df_exploit.columns:
        if 'aliases' in df_exploit.columns:
            df_exploit['codes'] = df_exploit['aliases']
        else:
            raise KeyError("Missing 'codes' or 'aliases' column!")

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

    if 'date_published' in df_exploit.columns:
        df_exploit['exploit_date'] = pd.to_datetime(df_exploit['date_published'], errors='coerce')

    return df_exploit

def merge_and_engineer_features(df_cve, df_exploit):
    df_exploit_grouped = df_exploit.groupby("CVE_list").agg({
        'CVE_list': 'count',
        'exploit_date': 'min'  # Najraniji exploit
    }).rename(columns={'CVE_list': 'num_exploits', 'exploit_date': 'first_exploit_date'})

    df_merged = pd.merge(df_cve, df_exploit_grouped,
                        how="left", left_on="CVE_ID", right_index=True)

    df_merged["num_exploits"] = df_merged["num_exploits"].fillna(0).astype(int)
    df_merged["vulnerable"] = (df_merged["num_exploits"] > 0).astype(int)

    df_merged["published"] = pd.to_datetime(df_merged["published"], errors='coerce')
    df_merged["first_exploit_date"] = pd.to_datetime(df_merged["first_exploit_date"], errors='coerce')

    df_merged["year_published"] = df_merged["published"].dt.year
    df_merged["month_published"] = df_merged["published"].dt.month
    df_merged["quarter_published"] = df_merged["published"].dt.quarter
    df_merged["day_of_week"] = df_merged["published"].dt.dayofweek
    df_merged["days_since_published"] = (datetime.now() - df_merged["published"]).dt.days

    df_merged["days_to_exploit"] = (
        df_merged["first_exploit_date"] - df_merged["published"]
    ).dt.days

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

    return df_merged

def calculate_exploit_probability(df):
    def calc_prob(row):
        prob = 0.0

        if row['num_exploits'] > 0:
            prob += min(0.4, row['num_exploits'] * 0.1)

        if pd.notna(row['baseScore']):
            prob += (row['baseScore'] / 10) * 0.3

        av_weights = {
            'NETWORK': 0.15,
            'ADJACENT_NETWORK': 0.10,
            'LOCAL': 0.05,
            'PHYSICAL': 0.02
        }
        prob += av_weights.get(row['attackVector'], 0.05)

        if row['is_critical_cwe'] == 1:
            prob += 0.10

        if pd.notna(row['days_since_published']) and row['days_since_published'] > 0:
            age_factor = min(row['days_since_published'] / 365, 5) / 5
            prob += age_factor * 0.05

        return min(prob, 1.0)

    df['exploit_probability'] = df.apply(calc_prob, axis=1)

    return df

def handle_missing_values(df):
    num_cols = ["baseScore", "exploitabilityScore", "impactScore",
                "days_since_published", "description_length", "description_words",
                "num_references"]

    imputer = SimpleImputer(strategy='median')
    df[num_cols] = imputer.fit_transform(df[num_cols])

    cat_cols = ["attackVector", "attackComplexity", "privilegesRequired",
                "userInteraction", "scope", "confidentialityImpact",
                "integrityImpact", "availabilityImpact", "CWE"]

    for col in cat_cols:
        df[col] = df[col].fillna("UNKNOWN")

    return df, imputer

def encode_categorical_features(df_train, df_test=None, encoder=None):
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


def generate_embeddings_batch(texts, model, tokenizer):
    model.eval()
    model.to(Config.DEVICE)

    all_embeddings = []
    texts_list = texts.tolist() if isinstance(texts, pd.Series) else texts

    for i in tqdm(range(0, len(texts_list), Config.EMBEDDING_BATCH_SIZE),
                  desc="  Generisanje embeddinga"):
        batch = texts_list[i:i+Config.EMBEDDING_BATCH_SIZE]

        batch = [str(text) if text and not pd.isna(text) else "" for text in batch]

        with torch.no_grad():
            inputs = tokenizer(batch, return_tensors="pt",
                             truncation=True,
                             max_length=Config.EMBEDDING_MAX_LENGTH,
                             padding=True)

            inputs = {k: v.to(Config.DEVICE) for k, v in inputs.items()}
            outputs = model(**inputs)

            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            all_embeddings.append(embeddings)

    final_embeddings = np.vstack(all_embeddings)

    return final_embeddings

def reduce_embedding_dimensions(embeddings_train, embeddings_test=None, pca=None):
    if not Config.USE_PCA:
        return embeddings_train, embeddings_test, None

    if pca is None:
        pca = PCA(n_components=Config.PCA_COMPONENTS, random_state=Config.RANDOM_STATE)
        embeddings_reduced_train = pca.fit_transform(embeddings_train)
    else:
        embeddings_reduced_train = pca.transform(embeddings_train)

    if embeddings_test is not None:
        embeddings_reduced_test = pca.transform(embeddings_test)
        return embeddings_reduced_train, embeddings_reduced_test, pca

    return embeddings_reduced_train, pca
