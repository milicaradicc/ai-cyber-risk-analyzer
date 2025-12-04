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
import gc


class CVEPreprocessor:
    def __init__(self, bert_model='bert-base-uncased'):
        self.bert_model_name = bert_model
        self.tokenizer = None
        self.bert_model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.cwe_top_list = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def set_seeds(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception as e:
            warnings.warn(f"{e}")

    def validate_schema(self, df, require_targets: bool = True):
        base_required = ['description']
        target_required = ['vulnerable', 'exploit_probability'] if require_targets else []
        required_cols = base_required + target_required
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing: {missing}. "
            )

    def load_data(self, filepath):
        df = pd.read_csv(filepath)
        return df

    def clean_data(self, df):
        original_len = len(df)
        df = df.drop_duplicates(subset=['CVE_ID'])

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

        categorical_cols = ['attackVector', 'attackComplexity', 'privilegesRequired',
                            'userInteraction', 'scope', 'confidentialityImpact',
                            'integrityImpact', 'availabilityImpact', 'CWE']

        for col in categorical_cols:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    df[col].fillna('UNKNOWN', inplace=True)

        return df

    def feature_engineering(self, df, auto_create_targets: bool = True, top_n_cwe: int = 20,
                            high_score_threshold: float = 7.0, old_days_threshold: int = 365):
        df = df.copy()

        CRITICAL_CWES = ['CWE-787', 'CWE-79', 'CWE-89', 'CWE-416', 'CWE-78',
                         'CWE-20', 'CWE-125', 'CWE-22', 'CWE-352', 'CWE-434',
                         'CWE-862', 'CWE-476', 'CWE-287', 'CWE-190', 'CWE-502']

        if 'CWE' in df.columns:
            df['is_critical_cwe'] = df['CWE'].isin(CRITICAL_CWES).astype(int)

        if auto_create_targets:
            if 'vulnerable' not in df.columns:
                if 'num_exploits' in df.columns:
                    df['vulnerable'] = (df['num_exploits'].fillna(0) > 0).astype(int)
                else:
                    warnings.warn("num_exploits missing")

            if 'exploit_probability' not in df.columns:

                def calc_prob(row):
                    prob = 0.0

                    if pd.notna(row.get('baseScore')):
                        prob += (float(row['baseScore']) / 10.0) * 0.40

                    if pd.notna(row.get('exploitabilityScore')):
                        prob += (float(row['exploitabilityScore']) / 10.0) * 0.30

                    av_weights = {
                        'NETWORK': 0.15,
                        'ADJACENT_NETWORK': 0.10,
                        'LOCAL': 0.05,
                        'PHYSICAL': 0.02
                    }
                    prob += av_weights.get(row.get('attackVector'), 0.05)

                    if pd.notna(row.get('impactScore')):
                        prob += (float(row['impactScore']) / 10.0) * 0.10

                    if row.get('is_critical_cwe') == 1:
                        prob += 0.05

                    return min(prob, 1.0)

                df['exploit_probability'] = df.apply(calc_prob, axis=1)

        if 'description' in df.columns:
            desc = df['description'].astype(str).fillna("")
            df['description_length'] = desc.str.len()
            df['description_words'] = desc.str.split().apply(len)

        impact_cols = [c for c in ['confidentialityImpact', 'integrityImpact', 'availabilityImpact'] if c in df.columns]
        if impact_cols:
            df['num_impacts'] = sum(
                [(df[c].fillna('NONE').astype(str).str.upper() != 'NONE').astype(int) for c in impact_cols])

        if 'days_since_published' in df.columns:
            df['is_old'] = (df['days_since_published'].fillna(0) > int(old_days_threshold)).astype(int)
        if 'baseScore' in df.columns:
            df['has_high_score'] = (df['baseScore'].astype(float) >= float(high_score_threshold)).astype(int)

        if 'CWE' in df.columns:
            vc = df['CWE'].fillna('UNKNOWN').astype(str).value_counts()
            top_list = list(vc.head(int(top_n_cwe)).index)
            self.cwe_top_list = top_list
            df['CWE_grouped'] = df['CWE'].fillna('UNKNOWN').astype(str).apply(lambda x: x if x in top_list else 'OTHER')

        return df

    def load_bert_model(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)
        self.bert_model = BertModel.from_pretrained(self.bert_model_name)
        self.bert_model.to(self.device)
        self.bert_model.eval()

    def get_bert_embedding(self, text, max_length=256):
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

        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embedding.flatten()

    def generate_bert_embeddings(self, df, batch_size=32, max_length=256):
        if self.bert_model is None or self.tokenizer is None:
            self.load_bert_model()

        all_embeddings = []
        descriptions = df['description'].astype(str).tolist()

        for i in tqdm(range(0, len(descriptions), batch_size)):
            batch_texts = descriptions[i:i + batch_size]
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

            del inputs, outputs, batch_embeddings

            if (i // batch_size) % 10 == 0:
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()

        embeddings_array = np.vstack(all_embeddings) if all_embeddings else np.empty(
            (0, self.bert_model.config.hidden_size))

        del self.bert_model, self.tokenizer, all_embeddings
        self.bert_model = None
        self.tokenizer = None

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

        return embeddings_array

    def normalize_numerical_features(self, df):
        numerical_features = [
            'baseScore', 'exploitabilityScore', 'impactScore',
            'num_references', 'days_since_published',
            'description_length', 'description_words', 'num_impacts'
        ]

        numerical_features = [col for col in numerical_features if col in df.columns]

        df[numerical_features] = self.scaler.fit_transform(df[numerical_features])

        return df, numerical_features

    def encode_categorical_features(self, df):
        categorical_features = [
            'attackVector', 'attackComplexity', 'privilegesRequired',
            'userInteraction', 'scope', 'confidentialityImpact',
            'integrityImpact', 'availabilityImpact', 'CWE_grouped'
        ]

        categorical_features = [col for col in categorical_features if col in df.columns]

        df_encoded = pd.get_dummies(
            df,
            columns=categorical_features,
            prefix=categorical_features,
            drop_first=True
        )

        del df
        gc.collect()

        return df_encoded

    def prepare_features(self, df, bert_embeddings):
        df, numerical_cols = self.normalize_numerical_features(df)
        df_encoded = self.encode_categorical_features(df)

        binary_features = ['is_remote', 'requires_auth', 'requires_user_interaction',
                           'high_severity', 'critical_severity', 'is_old', 'has_high_score',
                           'is_critical_cwe']
        binary_features = [col for col in binary_features if col in df_encoded.columns]

        feature_cols = numerical_cols + binary_features

        encoded_cols = [col for col in df_encoded.columns
                        if any(col.startswith(prefix + '_') for prefix in
                               ['attackVector', 'attackComplexity', 'privilegesRequired',
                                'userInteraction', 'scope', 'confidentialityImpact',
                                'integrityImpact', 'availabilityImpact', 'CWE_grouped'])]

        feature_cols.extend(encoded_cols)
        self.feature_columns = feature_cols

        y_classification = df_encoded['vulnerable'].values.astype(np.int32)
        y_regression = df_encoded['exploit_probability'].values.astype(np.float32)

        X_tabular = df_encoded[feature_cols].values.astype(np.float32)

        del df_encoded
        gc.collect()

        return bert_embeddings, X_tabular, y_classification, y_regression

    def split_data(self, X_bert, X_tabular, y_clf, y_reg, test_size=0.15, val_size=0.15, random_state: int = 42):
        X_bert_temp, X_bert_test, X_tab_temp, X_tab_test, y_clf_temp, y_clf_test, y_reg_temp, y_reg_test = \
            train_test_split(X_bert, X_tabular, y_clf, y_reg,
                             test_size=test_size, random_state=random_state, stratify=y_clf)

        val_ratio = val_size / (1 - test_size)
        X_bert_train, X_bert_val, X_tab_train, X_tab_val, y_clf_train, y_clf_val, y_reg_train, y_reg_val = \
            train_test_split(X_bert_temp, X_tab_temp, y_clf_temp, y_reg_temp,
                             test_size=val_ratio, random_state=random_state, stratify=y_clf_temp)

        return {
            'train': (X_bert_train, X_tab_train, y_clf_train, y_reg_train),
            'val': (X_bert_val, X_tab_val, y_clf_val, y_reg_val),
            'test': (X_bert_test, X_tab_test, y_clf_test, y_reg_test)
        }

    def apply_smote(self, X_bert, X_tabular, y_clf, mode: str = 'tabular_oversample', seed: int = 42):
        if mode == 'none':
            return X_bert, X_tabular, y_clf

        if mode == 'combined_smote':
            X_combined = np.hstack([X_bert, X_tabular])
            smote = SMOTE(random_state=seed, n_jobs=1) 
            X_resampled, y_resampled = smote.fit_resample(X_combined, y_clf)

            del X_combined
            gc.collect()

            X_bert_resampled = X_resampled[:, :X_bert.shape[1]].astype(np.float32)
            X_tab_resampled = X_resampled[:, X_bert.shape[1]:].astype(np.float32)

            del X_resampled
            gc.collect()

        elif mode == 'tabular_oversample':
            ros = RandomOverSampler(random_state=seed)
            X_tab_resampled, y_resampled = ros.fit_resample(X_tabular, y_clf)
            X_bert_resampled, _ = ros.fit_resample(X_bert, y_clf)

            X_bert_resampled = X_bert_resampled.astype(np.float32)
            X_tab_resampled = X_tab_resampled.astype(np.float32)
        else:
            return X_bert, X_tabular, y_clf

        del X_bert, X_tabular, y_clf
        gc.collect()

        return X_bert_resampled, X_tab_resampled, y_resampled

    def save_preprocessed_data(self, data_dict, save_dir='data/processed'):
        import os
        os.makedirs(save_dir, exist_ok=True)

        for split_name, (X_bert, X_tab, y_clf, y_reg) in data_dict.items():
            X_bert = np.asarray(X_bert, dtype=np.float32)
            X_tab = np.asarray(X_tab, dtype=np.float32)
            y_clf = np.asarray(y_clf, dtype=np.int32)
            y_reg = np.asarray(y_reg, dtype=np.float32)

            if np.isnan(X_tab).any() or np.isinf(X_tab).any():
                X_tab = np.nan_to_num(X_tab, nan=0.0, posinf=0.0, neginf=0.0)

            np.save(f'{save_dir}/X_bert_{split_name}.npy', X_bert)
            np.save(f'{save_dir}/X_tabular_{split_name}.npy', X_tab)
            np.save(f'{save_dir}/y_classification_{split_name}.npy', y_clf)
            np.save(f'{save_dir}/y_regression_{split_name}.npy', y_reg)

        with open(f'{save_dir}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)

        with open(f'{save_dir}/feature_columns.pkl', 'wb') as f:
            pickle.dump(self.feature_columns, f)

        if self.cwe_top_list is not None:
            with open(f'{save_dir}/cwe_top_list.pkl', 'wb') as f:
                pickle.dump(self.cwe_top_list, f)

    def run_full_pipeline(self, filepath, save_dir='data/processed', resample_mode: str = 'none', seed: int = 42,
                          batch_size: int = 32,
                          auto_create_targets: bool = True, top_n_cwe: int = 20,
                          high_score_threshold: float = 7.0, old_days_threshold: int = 365):
        self.set_seeds(seed)

        df = self.load_data(filepath)

        df = self.clean_data(df)

        self.validate_schema(df, require_targets=not auto_create_targets)

        df = self.feature_engineering(
            df,
            auto_create_targets=auto_create_targets,
            top_n_cwe=top_n_cwe,
            high_score_threshold=high_score_threshold,
            old_days_threshold=old_days_threshold,
        )

        self.validate_schema(df, require_targets=True)

        bert_embeddings = self.generate_bert_embeddings(df, batch_size=batch_size)

        X_bert, X_tabular, y_clf, y_reg = self.prepare_features(df, bert_embeddings)

        del df, bert_embeddings
        gc.collect()

        data_splits = self.split_data(X_bert, X_tabular, y_clf, y_reg, random_state=seed)

        if resample_mode and resample_mode != 'none':
            X_bert_train, X_tab_train, y_clf_train, y_reg_train = data_splits['train']
            X_bert_train, X_tab_train, y_clf_train = self.apply_smote(
                X_bert_train, X_tab_train, y_clf_train, mode=resample_mode, seed=seed
            )
            if len(y_reg_train) != len(y_clf_train):
                repeats = (len(y_clf_train) + len(y_reg_train) - 1) // len(y_reg_train)
                y_reg_train = np.tile(y_reg_train, repeats)[:len(y_clf_train)]
            data_splits['train'] = (X_bert_train, X_tab_train, y_clf_train, y_reg_train)

        self.save_preprocessed_data(data_splits, save_dir)

        gc.collect()

        return data_splits

if __name__ == "__main__":
    preprocessor = CVEPreprocessor(bert_model='bert-base-uncased')

    data_splits = preprocessor.run_full_pipeline(
        filepath='../processed_data/merged_data_raw.csv',
        save_dir='processed_data/processed',
        resample_mode='tabular_oversample', 
        seed=42,
        batch_size=8,
        auto_create_targets=True,
        top_n_cwe=20,
        high_score_threshold=7.0,
        old_days_threshold=365
    )