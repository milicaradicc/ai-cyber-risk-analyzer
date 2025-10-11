import os
import json
import pandas as pd
from tqdm import tqdm

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

def load_single_year_nvd_data(year=2024, base_folder="data", max_files=None):
    """
    Učitava NVD podatke samo za jednu godinu
    
    Args:
        year: Godina za učitavanje (npr. 2024)
        base_folder: Root folder sa podacima
        max_files: Maksimalan broj fajlova za učitavanje (None = svi)
    """
    print("\n" + "="*70)
    print(f"📂 UČITAVANJE NVD PODATAKA ZA {year}")
    print("="*70)
    
    year_folder = os.path.join(base_folder, f"CVE-{year}")
    
    if not os.path.exists(year_folder):
        print(f"❌ Folder '{year_folder}' ne postoji!")
        return None
    
    print(f"✅ Folder '{year_folder}' pronađen")
    
    # Pronađi sve JSON fajlove
    json_files = []
    for root, dirs, files in os.walk(year_folder):
        for f in files:
            if f.endswith(".json"):
                json_files.append(os.path.join(root, f))
    
    if len(json_files) == 0:
        print(f"❌ Nema JSON fajlova u folderu")
        return None
    
    print(f"📁 Pronađeno {len(json_files)} JSON fajlova")
    
    # Ograniči broj fajlova ako je zadato
    if max_files is not None and max_files < len(json_files):
        json_files = json_files[:max_files]
        print(f"⚠️  Učitavam samo prvih {max_files} fajlova")
    
    print(f"   Prvi fajl: {json_files[0]}")
    if len(json_files) > 1:
        print(f"   Zadnji fajl: {json_files[-1]}")
    
    all_cves = []
    parsed_files = 0
    failed_files = 0
    
    for file_path in tqdm(json_files, desc=f"  Parsiranje"):
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
            if failed_files <= 5:  # Prikaži prvih 5 grešaka
                print(f"\n⚠️  Greška u fajlu {os.path.basename(file_path)}: {str(e)}")
            continue
    
    print(f"\n✅ Uspešno parsiranih fajlova: {parsed_files}")
    print(f"❌ Neuspešnih fajlova: {failed_files}")
    
    if parsed_files == 0:
        print("\n❌ Nijedan JSON fajl nije uspešno parsiran!")
        return None
    
    df_cve = pd.DataFrame(all_cves)
    print(f"\n📊 Ukupno CVE zapisa: {len(df_cve)}")
    print(f"\n📋 Pregled podataka:")
    print(df_cve.head())
    print(f"\n📋 Info o kolonama:")
    print(df_cve.info())
    
    return df_cve


# ============================================================================
# TEST FUNKCIJA
# ============================================================================

def test_load_data():
    """Test funkcija koja učitava samo jednu godinu"""
    
    # PRIMER 1: Učitaj sve fajlove za 2024
    print("\n" + "🔵"*35)
    print("TEST 1: Učitavanje svih fajlova za 2024")
    print("🔵"*35)
    df_2024 = load_single_year_nvd_data(year=2024, base_folder="data")
    
    if df_2024 is not None:
        # Sačuvaj u CSV
        output_file = "test_output_2024_full.csv"
        df_2024.to_csv(output_file, index=False)
        print(f"\n💾 Podaci sačuvani u: {output_file}")
    
    print("\n" + "="*70 + "\n")
    
    # PRIMER 2: Učitaj samo prvih 100 fajlova za brži test
    print("\n" + "🟢"*35)
    print("TEST 2: Učitavanje prvih 100 fajlova za 2025 (brzi test)")
    print("🟢"*35)
    df_2025_sample = load_single_year_nvd_data(
        year=2025, 
        base_folder="data", 
        max_files=100
    )
    
    if df_2025_sample is not None:
        output_file = "test_output_2025_sample.csv"
        df_2025_sample.to_csv(output_file, index=False)
        print(f"\n💾 Podaci sačuvani u: {output_file}")
        
        # Prikaži neke statistike
        print("\n📊 STATISTIKE:")
        print(f"   - Ukupno CVE-ova: {len(df_2025_sample)}")
        print(f"   - CVE sa CVSS score: {df_2025_sample['baseScore'].notna().sum()}")
        print(f"   - CVE sa CWE: {df_2025_sample['CWE'].notna().sum()}")
        print(f"\n   Distribucija Attack Vector:")
        print(df_2025_sample['attackVector'].value_counts())


if __name__ == "__main__":
    test_load_data()