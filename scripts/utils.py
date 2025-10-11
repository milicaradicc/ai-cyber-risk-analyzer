import os
import pickle


def ensure_dir(directory):
    """Kreira direktorijum ako ne postoji"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"✅ Kreiran direktorijum: {directory}")

def save_pickle(obj, filepath):
    """Čuva objekat u pickle format"""
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    print(f"💾 Sačuvano: {filepath}")

def load_pickle(filepath):
    """Učitava objekat iz pickle formata"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)