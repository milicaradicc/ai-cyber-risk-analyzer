"""
Streamlit dashboard for ai-cyber-risk-analyzer

Run:
  streamlit run app.py

Expects data/processed to contain evaluation JSONs (eval_*.json) and optional plots from tsne_umap.py and shap_lime.py.
"""
import os
import glob
import json
import numpy as np
import streamlit as st

DATA_DIR = os.path.join("processed_data", "processed")


def find_eval_reports():
    pattern = os.path.join(DATA_DIR, "eval_*.json")
    files = sorted(glob.glob(pattern))
    return files


def load_report(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    st.set_page_config(page_title="AI Cyber Risk Analyzer", layout="wide")
    st.title("AI Cyber Risk Analyzer Dashboard")

    if not os.path.exists(DATA_DIR):
        st.warning("processed_data/processed not found. Run preprocessing and evaluation first.")
        return

    reports = find_eval_reports()
    sel = None
    if reports:
        sel = st.selectbox("Select evaluation report", reports, index=len(reports)-1)
    else:
        st.info("No evaluation reports found. Run evaluation.py with --save_json.")

    cols = st.columns(2)

    if sel is not None:
        rep = load_report(sel)
        st.subheader("Configuration")
        st.json(rep.get("config", {}))

        with cols[0]:
            st.subheader("Validation Metrics")
            st.json(rep.get("validation", {}))
        with cols[1]:
            st.subheader("Test Metrics")
            st.json(rep.get("test", {}))

    st.markdown("---")
    st.subheader("Visualizations")
    vis_dir = os.path.join(DATA_DIR, "visualizations")
    if os.path.isdir(vis_dir):
        imgs = sorted(glob.glob(os.path.join(vis_dir, "*.png")))
        for p in imgs:
            st.image(p, caption=os.path.basename(p), use_container_width=True)
    else:
        st.write("No visualizations found. Run tsne_umap.py to generate plots.")

    st.markdown("---")
    st.subheader("Interpretability")
    interp_dir = os.path.join(DATA_DIR, "interpret")
    if os.path.isdir(interp_dir):
        imgs = sorted(glob.glob(os.path.join(interp_dir, "*.png")))
        if imgs:
            cols = st.columns(2)
            for i, p in enumerate(imgs):
                with cols[i % 2]:
                    st.image(p, caption=os.path.basename(p), use_container_width=True)
        htmls = sorted(glob.glob(os.path.join(interp_dir, "*.html")))
        if htmls:
            st.write("LIME reports (open in a new tab):")
            for h in htmls:
                st.markdown(f"- [{os.path.basename(h)}]({h})")
    else:
        st.write("No interpretability artifacts yet. Run shap_lime.py.")

    st.markdown("---")
    st.subheader("Similarity search (BERT embeddings)")
    # Simple cosine similarity against train split
    Xb_train_path = os.path.join(DATA_DIR, "X_bert_train.npy")
    if os.path.exists(Xb_train_path):
        Xb_train = np.load(Xb_train_path)
        q_idx = st.number_input("Query index (row in train split)", min_value=0, max_value=max(0, len(Xb_train)-1), value=0)
        topk = st.slider("Top-K similar", 1, 20, 5)
        if st.button("Find similar"):
            q = Xb_train[int(q_idx)]
            qn = q / (np.linalg.norm(q) + 1e-12)
            Xn = Xb_train / (np.linalg.norm(Xb_train, axis=1, keepdims=True) + 1e-12)
            sims = Xn @ qn
            idxs = np.argsort(-sims)[:topk]
            st.write({int(i): float(sims[i]) for i in idxs})
    else:
        st.write("BERT embeddings not found.")


if __name__ == "__main__":
    main()
