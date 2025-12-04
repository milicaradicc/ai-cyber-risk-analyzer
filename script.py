import subprocess
import sys
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_SCRIPT = os.path.join(THIS_DIR, "evaluation.py")
MULTITASK_SCRIPT = os.path.join(THIS_DIR, "multitask.py")

data_dir = os.path.join(THIS_DIR, "data", "processed")
features = "both" 
seed = "42"

clf_models = ["logistic", "random_forest", "xgboost"]

reg_models = ["linear", "random_forest", "xgboost"]
for i in range(len(clf_models)):
    cmd = [
        sys.executable,
        EVAL_SCRIPT,
        "--data_dir", data_dir,
        "--features", features,
        "--seed", seed,
        "--clf_model", clf_models[i],
        "--reg_model", reg_models[i],
        "--save_json",
    ]
    subprocess.run(cmd, check=True)

cmd = [
    sys.executable,
    MULTITASK_SCRIPT,
    "--data_dir", data_dir,
    "--features", features,
    "--epochs", "10",
    "--batch_size", "64",
    "--lr", "1e-3",
    "--weight_decay", "1e-4",
    "--warmup_ratio", "0.1",
    "--clf_weight", "1.0",
    "--reg_weight", "1.0",
    "--seed", seed,
]
subprocess.run(cmd, check=True)