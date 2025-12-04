import os
import json
import numpy as np
import streamlit as st

DATA_DIR = os.path.join("processed_data")
RAW_DATA_DIR = os.path.join("data", "processed")

import os
import glob
from datetime import datetime

def find_eval_reports():
    evaluation_pattern = os.path.join(DATA_DIR, "evaluation", "*.json")
    evaluation_files = glob.glob(evaluation_pattern)
    def extract_timestamp(path):
        filename = os.path.basename(path)
        try:
            timestamp_str = filename.split('_')[-2] + "_" + filename.split('_')[-1].split('.')[0]
            return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        except Exception:
            return datetime.min

    files = sorted(evaluation_files, key=extract_timestamp, reverse=True)[:6]
    multitask_pattern = os.path.join(DATA_DIR, "multitask", "*.json")
    multitask_files = glob.glob(multitask_pattern)
    files.append(sorted(multitask_files, key=extract_timestamp, reverse=True)[0])
    return files

def load_report(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    st.set_page_config(page_title="AI Cyber Risk Analyzer", layout="wide")
    st.title("AI Cyber Risk Analyzer Dashboard")

    if not os.path.exists(DATA_DIR):
        st.warning(f"{DATA_DIR} not found. Run preprocessing and evaluation first.")
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
    vis_dir = os.path.join(RAW_DATA_DIR, "visualizations")
    if os.path.isdir(vis_dir):
        def newest_by_prefix(prefix: str):
            files = glob.glob(os.path.join(vis_dir, f"{prefix}*.png"))
            if not files:
                return None
            return max(files, key=lambda p: os.path.getmtime(p))
        latest_imgs = []
        for pref in ["tsne_train", "tsne_val", "tsne_test", "umap_train", "umap_val", "umap_test"]:
            p = newest_by_prefix(pref)
            if p:
                latest_imgs.append(p)
        if not latest_imgs:
            all_imgs = glob.glob(os.path.join(vis_dir, "*.png"))
            if all_imgs:
                latest_imgs = [max(all_imgs, key=lambda p: os.path.getmtime(p))]
        for p in latest_imgs:
            st.image(p, caption=os.path.basename(p), use_container_width=True)
    else:
        st.write("No visualizations found. Run tsne_umap.py to generate plots.")

    st.markdown("---")
    st.subheader("Interpretability")
    interp_dir = os.path.join(RAW_DATA_DIR, "interpret")
    if os.path.isdir(interp_dir):
        def newest_file(pattern: str):
            files = glob.glob(os.path.join(interp_dir, pattern))
            if not files:
                return None
            return max(files, key=lambda p: os.path.getmtime(p))
        newest_bar = newest_file("*bar*.png")
        newest_summary = newest_file("*summary*.png")
        cols2 = st.columns(2)
        if newest_bar:
            with cols2[0]:
                st.image(newest_bar, caption=os.path.basename(newest_bar), use_container_width=True)
        if newest_summary:
            with cols2[1]:
                st.image(newest_summary, caption=os.path.basename(newest_summary), use_container_width=True)

        newest_html = newest_file("*.html")
        if newest_html:
            st.write("Newest LIME report:")
            st.markdown(f"- [{os.path.basename(newest_html)}]({newest_html})")
        else:
            htmls = glob.glob(os.path.join(interp_dir, "*.html"))
            if htmls:
                st.write("LIME reports (open in a new tab):")
                for h in sorted(htmls, key=lambda p: os.path.getmtime(p), reverse=True)[:3]:
                    st.markdown(f"- [{os.path.basename(h)}]({h})")
    else:
        st.write("No interpretability artifacts yet. Run shap_lime.py.")

    st.markdown("---")
    st.subheader("Similarity search (BERT embeddings)")

    Xb_train_path = os.path.join(RAW_DATA_DIR, "X_bert_train.npy")
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
