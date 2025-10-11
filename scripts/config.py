import torch


class Config:
    # Paths
    NVD_BASE_FOLDER = "data"
    EXPLOIT_CSV_PATH = "data/files_exploits.csv"
    OUTPUT_DIR = "processed_data"
    
    # Processing
    START_YEAR = 2005
    END_YEAR = 2026
    EMBEDDING_MODEL = "microsoft/codebert-base"
    EMBEDDING_BATCH_SIZE = 32
    EMBEDDING_MAX_LENGTH = 128
    
    # Feature engineering
    CRITICAL_CWES = [
        'CWE-78', 'CWE-89', 'CWE-79', 'CWE-94', 'CWE-306', 
        'CWE-287', 'CWE-22', 'CWE-352', 'CWE-434', 'CWE-119'
    ]
    
    # Train/Test split
    TEST_SIZE = 0.15
    VALIDATION_SIZE = 0.15
    RANDOM_STATE = 42
    
    # Class balancing
    USE_SMOTE = True
    SMOTE_SAMPLING_STRATEGY = 0.5
    
    # PCA for dimensionality reduction
    USE_PCA = True
    PCA_COMPONENTS = 128
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
