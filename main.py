import os
import json
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

# --- Folderi ---
nvd_base_folder = "data"  # CVE-2005 -> CVE-2025
exploit_csv_path = "data/files_exploits.csv"

all_cves = []

def parse_cvss(metrics_dict):
    """Parsira CVSS metrike iz metrics dictionary-ja"""
    if not metrics_dict:
        return {}
    
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
    return {
        "baseScore": cvss.get("baseScore"),
        "attackVector": cvss.get("attackVector"),
        "attackComplexity": cvss.get("attackComplexity"),
        "privilegesRequired": cvss.get("privilegesRequired"),
        "userInteraction": cvss.get("userInteraction"),
        "scope": cvss.get("scope"),
        "confidentialityImpact": cvss.get("confidentialityImpact"),
        "integrityImpact": cvss.get("integrityImpact"),
        "availabilityImpact": cvss.get("availabilityImpact"),
        "exploitabilityScore": cvss_metric[0].get("exploitabilityScore"),
        "impactScore": cvss_metric[0].get("impactScore"),
    }

# --- Parsiranje NVD JSON ---
print("Parsiranje NVD JSON fajlova...")
parsed_files = 0
failed_files = 0

for year in range(2005, 2026):
    year_folder = os.path.join(nvd_base_folder, f"CVE-{year}")
    if not os.path.exists(year_folder):
        print(f"⚠️  Folder ne postoji: {year_folder}")
        continue

    # Rekurzivno prolazi kroz sve poddirektorijume
    json_count = 0
    for root, dirs, files in os.walk(year_folder):
        json_files = [f for f in files if f.endswith(".json")]
        json_count += len(json_files)
        
        for file_name in json_files:
            file_path = os.path.join(root, file_name)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    cve_id = data.get("id")
                    if not cve_id:
                        print(f"⚠️  Nema CVE ID u {file_name}")
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
                    
                    all_cves.append({
                        "CVE_ID": cve_id,
                        "description": description_en,
                        "published": published,
                        "lastModified": lastModified,
                        "CWE": cwe,
                        **metrics
                    })
                    parsed_files += 1
                    
            except json.JSONDecodeError as e:
                print(f"❌ JSON greška u {file_path}: {e}")
                failed_files += 1
            except Exception as e:
                print(f"❌ Greška pri parsiranju {file_path}: {e}")
                failed_files += 1
    
    print(f"📁 CVE-{year}: Pronađeno {json_count} JSON fajlova")

print(f"\n✅ Uspešno parsiranih fajlova: {parsed_files}")
print(f"❌ Neuspešnih fajlova: {failed_files}")
print(f"📊 Ukupno CVE zapisa: {len(all_cves)}")

# Provera da li ima podataka
if len(all_cves) == 0:
    raise ValueError("❌ Nema parsiranih CVE podataka! Proverite putanju do NVD foldera.")

df_cve = pd.DataFrame(all_cves)
print(f"\n📋 Kreirani df_cve sa {len(df_cve)} redova i kolonama: {list(df_cve.columns)}")

# --- Učitavanje ExploitDB ---
print(f"\n📂 Učitavanje ExploitDB iz: {exploit_csv_path}")
if not os.path.exists(exploit_csv_path):
    raise FileNotFoundError(f"❌ Fajl ne postoji: {exploit_csv_path}")

df_exploit = pd.read_csv(exploit_csv_path)
print(f"✅ Učitano {len(df_exploit)} exploit zapisa")
print(f"📋 Kolone: {list(df_exploit.columns)}")

# Provera da li postoji 'codes' kolona
if 'codes' not in df_exploit.columns:
    print("⚠️  'codes' kolona ne postoji, pokušavam sa 'aliases'...")
    if 'aliases' in df_exploit.columns:
        df_exploit['codes'] = df_exploit['aliases']
    else:
        raise KeyError("❌ Nema ni 'codes' ni 'aliases' kolone u ExploitDB!")

def extract_cve_list(codes):
    """Izvlači CVE ID-ove iz codes/aliases kolone"""
    if pd.isna(codes):
        return []
    if isinstance(codes, str):
        # Podržava i ';' i ',' kao separator
        codes = codes.replace(',', ';')
        return [c.strip() for c in codes.split(";") if c.strip().startswith("CVE-")]
    return []

df_exploit["CVE_list"] = df_exploit["codes"].apply(extract_cve_list)
print(f"\n🔍 Primer ekstrakcije CVE-ova:")
print(df_exploit[["codes", "CVE_list"]].head())

# Eksplodovanje liste CVE-ova u odvojene redove
df_exploit = df_exploit.explode("CVE_list")
df_exploit = df_exploit[df_exploit["CVE_list"].notna()]  # Ukloni prazne

print(f"✅ Nakon explode: {len(df_exploit)} redova sa CVE mapiranjem")

# Brojanje exploit-a po CVE-u
df_exploit_grouped = df_exploit.groupby("CVE_list").size().reset_index(name="num_exploits")
print(f"📊 Ukupno CVE-ova sa exploitima: {len(df_exploit_grouped)}")
print(f"📈 Primer brojanja:\n{df_exploit_grouped.head()}")

# --- Spajanje ---
print(f"\n🔗 Spajanje NVD i ExploitDB podataka...")
df_merged = pd.merge(df_cve, df_exploit_grouped, how="left", left_on="CVE_ID", right_on="CVE_list")
df_merged["num_exploits"] = df_merged["num_exploits"].fillna(0).astype(int)
df_merged["vulnerable"] = (df_merged["num_exploits"] > 0).astype(int)

print(f"✅ Spojeni podaci: {len(df_merged)} redova")
print(f"📊 Distribucija vulnerable:")
print(df_merged["vulnerable"].value_counts())
print(f"📊 Statistika num_exploits:\n{df_merged['num_exploits'].describe()}")

# --- Pretvaranje datuma u feature ---
df_merged["published"] = pd.to_datetime(df_merged["published"], errors='coerce')
df_merged["year_published"] = df_merged["published"].dt.year
df_merged["month_published"] = df_merged["published"].dt.month
df_merged["days_since_published"] = (datetime.now() - df_merged["published"]).dt.days

# --- One-hot encoding za kategorijske CVSS feature-e ---
cat_cols = ["attackVector", "attackComplexity", "privilegesRequired", "userInteraction",
            "scope", "confidentialityImpact", "integrityImpact", "availabilityImpact"]

df_cat = df_merged[cat_cols].fillna("UNKNOWN")
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
cat_encoded = ohe.fit_transform(df_cat)
df_cat_encoded = pd.DataFrame(cat_encoded, columns=ohe.get_feature_names_out(cat_cols))

print(f"\n🔢 One-hot encoded features: {df_cat_encoded.shape[1]} kolona")

# --- Standardizacija numeričkih feature-a ---
num_cols = ["baseScore", "exploitabilityScore", "impactScore"]
scaler = StandardScaler()
df_num_scaled = pd.DataFrame(
    scaler.fit_transform(df_merged[num_cols].fillna(0)), 
    columns=[f"{col}_scaled" for col in num_cols]
)

print(f"📏 Standardizovani numerički features: {df_num_scaled.shape[1]} kolona")

# --- Tekstualni embeddingi pomoću CodeBERT ---
print(f"\n🤖 Generisanje text embeddinga pomoću CodeBERT...")
print("⏳ Ovo može potrajati nekoliko minuta...")

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
model.eval()

def get_text_embedding(text):
    """Generiše embedding za dati tekst"""
    if not text or pd.isna(text):
        return np.zeros(model.config.hidden_size)
    
    try:
        with torch.no_grad():
            inputs = tokenizer(str(text), return_tensors="pt", truncation=True, max_length=128, padding=True)
            outputs = model(**inputs)
            # Mean pooling preko svih tokena
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embedding
    except Exception as e:
        print(f"⚠️  Greška pri embedovanju: {e}")
        return np.zeros(model.config.hidden_size)

# Generisanje embeddinga za sve CVE-ove (može se limitirati za brži test)
LIMIT = 1000  # Postavite na None za sve CVE-ove
sample_size = min(LIMIT, len(df_merged)) if LIMIT else len(df_merged)

print(f"📝 Procesiranje {sample_size} opisa ranjivosti...")
embeddings = []
for idx, desc in enumerate(df_merged["description"].iloc[:sample_size]):
    if idx % 100 == 0:
        print(f"  Progress: {idx}/{sample_size}")
    emb = get_text_embedding(desc)
    embeddings.append(emb)

df_embed = pd.DataFrame(embeddings, columns=[f"embed_{i}" for i in range(len(embeddings[0]))])
print(f"✅ Kreirano {df_embed.shape[1]} embedding dimenzija")

# --- Spajanje svih feature-a ---
print(f"\n🔗 Spajanje svih feature-a u finalni dataset...")
df_final = pd.concat([
    df_merged.iloc[:sample_size].reset_index(drop=True),
    df_cat_encoded.iloc[:sample_size].reset_index(drop=True),
    df_num_scaled.iloc[:sample_size].reset_index(drop=True),
    df_embed.reset_index(drop=True)
], axis=1)

print(f"✅ Finalni dataset: {df_final.shape[0]} redova × {df_final.shape[1]} kolona")

# --- Čuvanje u CSV ---
output_csv_path = "processed_vulnerabilities_features.csv"
df_final.to_csv(output_csv_path, index=False)
print(f"\n💾 Sačuvano u: {output_csv_path}")

# --- Dodatne informacije ---
print(f"\n📊 SAŽETAK:")
print(f"  • Ukupno CVE zapisa: {len(df_final)}")
print(f"  • Ranjivi (sa exploitima): {df_final['vulnerable'].sum()}")
print(f"  • Neranjivi: {len(df_final) - df_final['vulnerable'].sum()}")
print(f"  • Prosečan broj exploita: {df_final['num_exploits'].mean():.2f}")
print(f"  • Max broj exploita: {df_final['num_exploits'].max()}")
print(f"  • Feature-i: {df_final.shape[1]} kolona")

print("\n✅ Preprocessing završen!")