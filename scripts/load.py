from utils import load_pickle
import os

OUTPUT_DIR = "processed_data"

# Učitaj feature-e i targete
X_train = load_pickle(os.path.join(OUTPUT_DIR, "X_train.pkl"))
y_train_clf = load_pickle(os.path.join(OUTPUT_DIR, "y_train_clf.pkl"))

# Ili balansirani set
X_train_balanced = load_pickle(os.path.join(OUTPUT_DIR, "X_train_balanced.pkl"))
y_train_clf_balanced = load_pickle(os.path.join(OUTPUT_DIR, "y_train_clf_balanced.pkl"))

# Učitaj metadata
metadata = load_pickle(os.path.join(OUTPUT_DIR, "metadata.pkl"))
print(f"Broj feature-a: {metadata['n_features']}")