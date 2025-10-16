"""
Preprocesiranje CVE podataka za vulnerability prediction
Kombinuje BERT embeddings sa tabularnim features
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm
import warnings
import random


class CVEPreprocessor:
    """Klasa za kompletno preprocesiranje CVE podataka"""

    def __init__(self, bert_model='bert-base-uncased'):
        self.bert_model_name = bert_model
        self.tokenizer = None
        self.bert_model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.cwe_top_list = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def set_seeds(self, seed: int = 42):
        """Postavljanje random seed-ova za reproduktivnost"""
        print(f"\n=== POSTAVLJANJE SEED-A: {seed} ===")
        np.random.seed(seed)
        random.seed(seed)
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            # Determinističko ponašanje (sporije ali reproduktivnije)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception as e:
            warnings.warn(f"Neuspešno postavljanje torch seed-a: {e}")

    def validate_schema(self, df, require_targets: bool = True):
        """Osnovna validacija ulaznog DataFrame-a.
        Ako require_targets=False, proverava samo prisustvo 'description' i dozvoljava da se mete
        generišu u feature_engineering koraku.
        """
        base_required = ['description']
        target_required = ['vulnerable', 'exploit_probability'] if require_targets else []
        required_cols = base_required + target_required
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Nedostaju obavezne kolone: {missing}. "
                f"Dodajte ih u CSV ili omogućite kreiranje u feature_engineering."
            )

    def load_data(self, filepath):
        """Učitavanje CSV podataka"""
        print(f"Učitavanje podataka iz {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Učitano {len(df)} redova i {len(df.columns)} kolona")
        return df

    def clean_data(self, df):
        """Čišćenje podataka"""
        print("\n=== ČIŠĆENJE PODATAKA ===")

        # Uklanjanje duplikata
        original_len = len(df)
        df = df.drop_duplicates(subset=['CVE_ID'])
        print(f"Uklonjeno {original_len - len(df)} duplikata")

        # Obrada missing vrednosti - numeričke kolone
        numeric_cols = ['baseScore', 'exploitabilityScore', 'impactScore',
                       'num_references', 'num_exploits', 'days_since_published']

        for col in numeric_cols:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    if col == 'num_exploits':
                        df[col].fillna(0, inplace=True)
                    else:
                        df[col].fillna(df[col].median(), inplace=True)
                    print(f"{col}: popunjeno {missing_count} missing vrednosti")

        # Obrada missing vrednosti - kategoričke kolone
        categorical_cols = ['attackVector', 'attackComplexity', 'privilegesRequired',
                           'userInteraction', 'scope', 'confidentialityImpact',
                           'integrityImpact', 'availabilityImpact', 'CWE']

        for col in categorical_cols:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    df[col].fillna('UNKNOWN', inplace=True)
                    print(f"{col}: popunjeno {missing_count} missing vrednosti")

        return df

    def feature_engineering(self, df, auto_create_targets: bool = True, top_n_cwe: int = 20,
                            high_score_threshold: float = 7.0, old_days_threshold: int = 365):
        """Kreiranje novih features usklađeno sa specifikacijom."""
        print("\n=== FEATURE ENGINEERING ===")
        df = df.copy()

        # ✅ MOVE THIS UP: Create is_critical_cwe FIRST (needed for exploit_probability)
        # Define critical CWEs
        CRITICAL_CWES = ['CWE-787', 'CWE-79', 'CWE-89', 'CWE-416', 'CWE-78',
                         'CWE-20', 'CWE-125', 'CWE-22', 'CWE-352', 'CWE-434',
                         'CWE-862', 'CWE-476', 'CWE-287', 'CWE-190', 'CWE-502']

        if 'CWE' in df.columns:
            df['is_critical_cwe'] = df['CWE'].isin(CRITICAL_CWES).astype(int)

        # 1) Ciljne promenljive
        if auto_create_targets:
            if 'vulnerable' not in df.columns:
                if 'num_exploits' in df.columns:
                    df['vulnerable'] = (df['num_exploits'].fillna(0) > 0).astype(int)
                else:
                    warnings.warn("'num_exploits' nije prisutan; 'vulnerable' neće biti kreiran.")

            # Create exploit_probability WITHOUT num_exploits
            if 'exploit_probability' not in df.columns:
                print("⚠️  Kreiranje exploit_probability BEZ korišćenja num_exploits...")

                def calc_prob(row):
                    prob = 0.0

                    # 1. CVSS base score (40%)
                    if pd.notna(row.get('baseScore')):
                        prob += (float(row['baseScore']) / 10.0) * 0.40

                    # 2. Exploitability score (30%)
                    if pd.notna(row.get('exploitabilityScore')):
                        prob += (float(row['exploitabilityScore']) / 10.0) * 0.30

                    # 3. Attack vector (15%)
                    av_weights = {
                        'NETWORK': 0.15,
                        'ADJACENT_NETWORK': 0.10,
                        'LOCAL': 0.05,
                        'PHYSICAL': 0.02
                    }
                    prob += av_weights.get(row.get('attackVector'), 0.05)

                    # 4. Impact score (10%)
                    if pd.notna(row.get('impactScore')):
                        prob += (float(row['impactScore']) / 10.0) * 0.10

                    # 5. Critical CWE (5%)
                    if row.get('is_critical_cwe') == 1:
                        prob += 0.05

                    return min(prob, 1.0)

                df['exploit_probability'] = df.apply(calc_prob, axis=1)

        # 2) Tekstualne karakteristike
        if 'description' in df.columns:
            desc = df['description'].astype(str).fillna("")
            df['description_length'] = desc.str.len()
            df['description_words'] = desc.str.split().apply(len)

        # 3) Broj uticaja (CIA)
        impact_cols = [c for c in ['confidentialityImpact', 'integrityImpact', 'availabilityImpact'] if c in df.columns]
        if impact_cols:
            df['num_impacts'] = sum(
                [(df[c].fillna('NONE').astype(str).str.upper() != 'NONE').astype(int) for c in impact_cols])

        # 4) Starost i visoki skor
        if 'days_since_published' in df.columns:
            df['is_old'] = (df['days_since_published'].fillna(0) > int(old_days_threshold)).astype(int)
        if 'baseScore' in df.columns:
            df['has_high_score'] = (df['baseScore'].astype(float) >= float(high_score_threshold)).astype(int)

        # 5) Grupisanje CWE (top-N + OTHER)
        if 'CWE' in df.columns:
            vc = df['CWE'].fillna('UNKNOWN').astype(str).value_counts()
            top_list = list(vc.head(int(top_n_cwe)).index)
            self.cwe_top_list = top_list
            df['CWE_grouped'] = df['CWE'].fillna('UNKNOWN').astype(str).apply(lambda x: x if x in top_list else 'OTHER')

        if 'vulnerable' in df.columns:
            print(f"Distribucija 'vulnerable': {df['vulnerable'].value_counts().to_dict()}")
        created_cols = [c for c in
                        ['vulnerable', 'exploit_probability', 'description_length', 'description_words', 'num_impacts',
                         'is_old', 'has_high_score', 'CWE_grouped'] if c in df.columns]
        print(f"Kreirano/azurirano features: {len(created_cols)} -> {created_cols}")

        return df

    def load_bert_model(self):
        """Učitavanje BERT modela"""
        print(f"\n=== UČITAVANJE BERT MODELA ({self.bert_model_name}) ===")
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)
        self.bert_model = BertModel.from_pretrained(self.bert_model_name)
        self.bert_model.to(self.device)
        self.bert_model.eval()
        print(f"BERT model učitan na device: {self.device}")

    def get_bert_embedding(self, text, max_length=512):
        """Generisanje BERT embedding vektora za jedan tekst"""
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding='max_length'
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.bert_model(**inputs)

        # Koristimo [CLS] token embedding
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embedding.flatten()

    def generate_bert_embeddings(self, df, batch_size=32, max_length=512):
        """Generisanje BERT embeddings za sve opise (batched)"""
        print(f"\n=== GENERISANJE BERT EMBEDDINGS (batched) ===")
        print(f"Procesiranje {len(df)} opisa sa batch_size={batch_size}...")

        if self.bert_model is None or self.tokenizer is None:
            self.load_bert_model()

        all_embeddings = []
        descriptions = df['description'].astype(str).tolist()

        for i in tqdm(range(0, len(descriptions), batch_size)):
            batch_texts = descriptions[i:i+batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors='pt',
                truncation=True,
                max_length=max_length,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(batch_embeddings)

        embeddings_array = np.vstack(all_embeddings) if all_embeddings else np.empty((0, self.bert_model.config.hidden_size))
        print(f"Generisano embeddings shape: {embeddings_array.shape}")

        return embeddings_array

    def normalize_numerical_features(self, df):
        """Normalizacija numeričkih features"""
        print("\n=== NORMALIZACIJA NUMERIČKIH FEATURES ===")

        numerical_features = [
            'baseScore', 'exploitabilityScore', 'impactScore',
            'num_references', 'days_since_published',
            'description_length', 'description_words', 'num_impacts'
        ]

        # Filtriranje samo postojećih kolona
        numerical_features = [col for col in numerical_features if col in df.columns]

        df_norm = df.copy()
        df_norm[numerical_features] = self.scaler.fit_transform(df[numerical_features])

        print(f"Normalizovano {len(numerical_features)} numeričkih features")

        return df_norm, numerical_features

    def encode_categorical_features(self, df):
        """One-hot encoding kategoričkih features"""
        print("\n=== ONE-HOT ENCODING KATEGORIČKIH FEATURES ===")

        categorical_features = [
            'attackVector', 'attackComplexity', 'privilegesRequired',
            'userInteraction', 'scope', 'confidentialityImpact',
            'integrityImpact', 'availabilityImpact', 'CWE_grouped'
        ]

        # Filtriranje samo postojećih kolona
        categorical_features = [col for col in categorical_features if col in df.columns]

        df_encoded = pd.get_dummies(
            df,
            columns=categorical_features,
            prefix=categorical_features,
            drop_first=True
        )

        print(f"Enkodovano {len(categorical_features)} kategoričkih features")
        print(f"Ukupno kolona nakon encoding-a: {len(df_encoded.columns)}")

        return df_encoded

    def prepare_features(self, df, bert_embeddings):
        """Priprema finalnih feature matrica"""
        print("\n=== PRIPREMA FINALNIH FEATURES ===")

        # Normalizacija i encoding
        df_norm, numerical_cols = self.normalize_numerical_features(df)
        df_encoded = self.encode_categorical_features(df_norm)

        # Binary features
        binary_features = ['is_remote', 'requires_auth', 'requires_user_interaction',
                          'high_severity', 'critical_severity', 'is_old', 'has_high_score']
        binary_features = [col for col in binary_features if col in df_encoded.columns]

        # Selekcija features za tabularni tok
        feature_cols = numerical_cols + binary_features

        # Dodavanje one-hot encoded kolona
        encoded_cols = [col for col in df_encoded.columns
                       if any(col.startswith(prefix + '_') for prefix in
                             ['attackVector', 'attackComplexity', 'privilegesRequired',
                              'userInteraction', 'scope', 'confidentialityImpact',
                              'integrityImpact', 'availabilityImpact', 'CWE_grouped'])]

        feature_cols.extend(encoded_cols)
        self.feature_columns = feature_cols

        # Kreiranje tabularnih features
        X_tabular = df_encoded[feature_cols].values

        # Target varijable
        y_classification = df['vulnerable'].values
        y_regression = df['exploit_probability'].values

        print(f"BERT embeddings shape: {bert_embeddings.shape}")
        print(f"Tabular features shape: {X_tabular.shape}")
        print(f"Classification target shape: {y_classification.shape}")
        print(f"Regression target shape: {y_regression.shape}")

        return bert_embeddings, X_tabular, y_classification, y_regression

    def split_data(self, X_bert, X_tabular, y_clf, y_reg, test_size=0.15, val_size=0.15, random_state: int = 42):
        """Podela na train/val/test setove"""
        print("\n=== PODELA PODATAKA ===")

        # Prvo odvajamo test set
        X_bert_temp, X_bert_test, X_tab_temp, X_tab_test, y_clf_temp, y_clf_test, y_reg_temp, y_reg_test = \
            train_test_split(X_bert, X_tabular, y_clf, y_reg,
                           test_size=test_size, random_state=random_state, stratify=y_clf)

        # Zatim train i validation
        val_ratio = val_size / (1 - test_size)
        X_bert_train, X_bert_val, X_tab_train, X_tab_val, y_clf_train, y_clf_val, y_reg_train, y_reg_val = \
            train_test_split(X_bert_temp, X_tab_temp, y_clf_temp, y_reg_temp,
                           test_size=val_ratio, random_state=random_state, stratify=y_clf_temp)

        print(f"Train set: {X_bert_train.shape[0]} samples")
        print(f"Validation set: {X_bert_val.shape[0]} samples")
        print(f"Test set: {X_bert_test.shape[0]} samples")

        return {
            'train': (X_bert_train, X_tab_train, y_clf_train, y_reg_train),
            'val': (X_bert_val, X_tab_val, y_clf_val, y_reg_val),
            'test': (X_bert_test, X_tab_test, y_clf_test, y_reg_test)
        }

    def apply_smote(self, X_bert, X_tabular, y_clf, mode: str = 'tabular_oversample', seed: int = 42):
        """Balansiranje dataseta.
        mode opcije:
        - 'none': bez balansiranja
        - 'tabular_oversample': bezbednije – duplira manjinsku klasu nasumično (bez sintetike); poravnava i BERT granu
        - 'combined_smote': stari pristup – SMOTE nad [BERT|tabular] (može biti skupo i semantički sporno)
        """
        if mode == 'none':
            return X_bert, X_tabular, y_clf

        print(f"\n=== BALANSIRANJE ({mode}) ===")
        print(f"Distribucija pre: {np.bincount(y_clf)}")

        if mode == 'combined_smote':
            X_combined = np.hstack([X_bert, X_tabular])
            smote = SMOTE(random_state=seed)
            X_resampled, y_resampled = smote.fit_resample(X_combined, y_clf)
            X_bert_resampled = X_resampled[:, :X_bert.shape[1]]
            X_tab_resampled = X_resampled[:, X_bert.shape[1]:]
        elif mode == 'tabular_oversample':
            ros = RandomOverSampler(random_state=seed)
            X_tab_resampled, y_resampled = ros.fit_resample(X_tabular, y_clf)
            # Primeni isti ROS na BERT granu kako bi indeksi bili poravnati
            X_bert_resampled, _ = ros.fit_resample(X_bert, y_clf)
        else:
            warnings.warn(f"Nepoznat mode '{mode}'. Preskačem balansiranje.")
            return X_bert, X_tabular, y_clf

        print(f"Distribucija posle: {np.bincount(y_resampled)}")
        return X_bert_resampled, X_tab_resampled, y_resampled

    def save_preprocessed_data(self, data_dict, save_dir='processed_data/processed'):
        """Čuvanje preprocesiranih podataka"""
        print(f"\n=== ČUVANJE PODATAKA U {save_dir} ===")

        import os
        os.makedirs(save_dir, exist_ok=True)

        # Čuvanje train/val/test setova
        for split_name, (X_bert, X_tab, y_clf, y_reg) in data_dict.items():
            print(f"Processing {split_name} split...")

            # ✅ Ensure proper dtypes before saving
            X_bert = np.asarray(X_bert, dtype=np.float32)
            X_tab = np.asarray(X_tab, dtype=np.float32)
            y_clf = np.asarray(y_clf, dtype=np.int32)
            y_reg = np.asarray(y_reg, dtype=np.float32)

            # Handle NaN/inf values
            if np.isnan(X_tab).any() or np.isinf(X_tab).any():
                print(f"⚠️  Warning: X_tab contains NaN/inf values, replacing with 0")
                X_tab = np.nan_to_num(X_tab, nan=0.0, posinf=0.0, neginf=0.0)

            np.save(f'{save_dir}/X_bert_{split_name}.npy', X_bert)
            np.save(f'{save_dir}/X_tabular_{split_name}.npy', X_tab)
            np.save(f'{save_dir}/y_classification_{split_name}.npy', y_clf)
            np.save(f'{save_dir}/y_regression_{split_name}.npy', y_reg)

            print(f"✅ Sačuvano: {split_name} set")
            print(f"   X_bert: {X_bert.shape}, dtype={X_bert.dtype}")
            print(f"   X_tab: {X_tab.shape}, dtype={X_tab.dtype}")

        # Čuvanje scaler-a, feature lista i CWE top liste
        with open(f'{save_dir}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)

        with open(f'{save_dir}/feature_columns.pkl', 'wb') as f:
            pickle.dump(self.feature_columns, f)

        if self.cwe_top_list is not None:
            with open(f'{save_dir}/cwe_top_list.pkl', 'wb') as f:
                pickle.dump(self.cwe_top_list, f)

        print("\n✅ Sačuvani: scaler, feature_columns i (opciono) cwe_top_list")

    def run_full_pipeline(self, filepath, save_dir='processed_data/processed', resample_mode: str = 'none', seed: int = 42, batch_size: int = 32,
                          auto_create_targets: bool = True, top_n_cwe: int = 20,
                          high_score_threshold: float = 7.0, old_days_threshold: int = 365):
        """Izvršavanje kompletnog pipeline-a"""
        print("="*60)
        print("POKRETANJE KOMPLETNOG PREPROCESIRANJA")
        print("="*60)

        # 0. Seed-ovi
        self.set_seeds(seed)

        # 1. Učitavanje
        df = self.load_data(filepath)

        # 2. Čišćenje
        df = self.clean_data(df)

        # 3.0 Validacija pre FE (ako ne kreiramo mete, moraju postojati)
        self.validate_schema(df, require_targets=not auto_create_targets)

        # 3. Feature engineering
        df = self.feature_engineering(
            df,
            auto_create_targets=auto_create_targets,
            top_n_cwe=top_n_cwe,
            high_score_threshold=high_score_threshold,
            old_days_threshold=old_days_threshold,
        )

        # 3.5 Validacija šeme (targets i opis) – nakon FE mete treba da postoje
        self.validate_schema(df, require_targets=True)

        # 4. Generisanje BERT embeddings
        bert_embeddings = self.generate_bert_embeddings(df, batch_size=batch_size)

        # 5. Priprema features
        X_bert, X_tabular, y_clf, y_reg = self.prepare_features(df, bert_embeddings)

        # 6. Podela podataka
        data_splits = self.split_data(X_bert, X_tabular, y_clf, y_reg, random_state=seed)

        # 7. Opciono: balansiranje train seta
        if resample_mode and resample_mode != 'none':
            X_bert_train, X_tab_train, y_clf_train, y_reg_train = data_splits['train']
            X_bert_train, X_tab_train, y_clf_train = self.apply_smote(
                X_bert_train, X_tab_train, y_clf_train, mode=resample_mode, seed=seed
            )
            # Poravnavanje y_reg sa novim brojem uzoraka (dupliranje samo radi poravnanja)
            if len(y_reg_train) != len(y_clf_train):
                y_reg_train = np.tile(y_reg_train, len(y_clf_train) // len(y_reg_train) + 1)[:len(y_clf_train)]
            data_splits['train'] = (X_bert_train, X_tab_train, y_clf_train, y_reg_train)

        # 8. Čuvanje
        self.save_preprocessed_data(data_splits, save_dir)

        print("\n" + "="*60)
        print("PREPROCESIRANJE ZAVRŠENO!")
        print("="*60)

        return data_splits


if __name__ == "__main__":
    # Primer korišćenja
    preprocessor = CVEPreprocessor(bert_model='bert-base-uncased')

    # Izvršavanje kompletnog pipeline-a
    data_splits = preprocessor.run_full_pipeline(
        filepath='../processed_data/merged_data_raw.csv',
        save_dir='processed_data/processed',
        resample_mode='tabular_oversample',  # opcije: 'none', 'tabular_oversample', 'combined_smote'
        seed=42,
        batch_size=8,
        auto_create_targets=True,
        top_n_cwe=20,
        high_score_threshold=7.0,
        old_days_threshold=365
    )

    print("\nPreprocesiranje uspešno završeno!")
    print("Podaci su spremni za trening modela.")