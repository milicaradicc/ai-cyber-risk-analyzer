import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

try:
    import umap  
except Exception:
    umap = None

def load_arrays(data_dir: str, split: str):
    Xb = np.load(os.path.join(data_dir, f"X_bert_{split}.npy"))
    y = np.load(os.path.join(data_dir, f"y_classification_{split}.npy"))
    return Xb, y

def subsample(X, y, k):
    if k is None or k <= 0 or k >= len(X):
        return X, y
    idx = np.random.choice(len(X), size=k, replace=False)
    return X[idx], y[idx]

def scatter_2d(Z, y, title, out_path):
    plt.figure(figsize=(7, 6))
    y = y.astype(int)
    plt.scatter(Z[:, 0], Z[:, 1], c=y, cmap="coolwarm", s=8, alpha=0.8)
    plt.title(title)
    plt.colorbar(label="vulnerable")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=os.path.join("data", "processed"))
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test"]) 
    ap.add_argument("--sample", type=int, default=1000, help="Subsample size (0 or negative for all)")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--n_neighbors", type=int, default=15)
    ap.add_argument("--min_dist", type=float, default=0.1)
    args = ap.parse_args()

    out_dir = os.path.join(args.data_dir, "visualizations")
    os.makedirs(out_dir, exist_ok=True)

    Xb, y = load_arrays(args.data_dir, args.split)
    Xb, y = subsample(Xb, y, args.sample)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=args.perplexity, init="pca", learning_rate="auto")
    Z_tsne = tsne.fit_transform(Xb)
    scatter_2d(Z_tsne, y, f"t-SNE ({args.split})", os.path.join(out_dir, f"tsne_{args.split}.png"))

    # UMAP
    if umap is not None:
        reducer = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist, n_components=2)
        Z_umap = reducer.fit_transform(Xb)
        scatter_2d(Z_umap, y, f"UMAP ({args.split})", os.path.join(out_dir, f"umap_{args.split}.png"))
    else:
        print("umap-learn is not installed; skipping UMAP plot.")

if __name__ == "__main__":
    main()
