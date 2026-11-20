# src/lab/train.py

import argparse
import yaml
import joblib
import os
import json

from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline

from src.lab.data import load_data, build_preprocessor
from src.lab.model import build_model, AVAILABLE_MODELS


# ---------------- Config Loader ----------------
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------- Load tuned params ----------------
def load_tuned_params(model_name):
    tune_path = Path(f"outputs/tuning/{model_name}_best_params.json")

    if tune_path.exists():
        with open(tune_path) as f:
            print(f"Using tuned params for {model_name}")
            return json.load(f)

    return None


# ---------------- Main Training ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fast_dev_run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)

    test_size = config["data"]["test_size"]
    seed = config["training"]["seed"]

    # ---------------- Load Data ----------------
    X, y = load_data()

    print("\n===== DATA TYPES =====")
    print(X.dtypes)
    print("======================\n")

    schema = {col: str(dtype) for col, dtype in X.dtypes.items()}

    os.makedirs("outputs/schema", exist_ok=True)

    with open("outputs/schema/feature_schema.json", "w") as f:
        json.dump(schema, f, indent=4)

    print("Feature schema saved to outputs/schema/feature_schema.json")


    if args.fast_dev_run:
        print("⚡ Fast dev run enabled")
        X = X.sample(2000, random_state=seed)
        y = y.loc[X.index]

    # ---------------- Train/Test Split ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    preprocessor = build_preprocessor(X_train)

    results = {}
    best_auc = 0
    best_pipeline = None
    best_model_name = None

    print("\n🚀 Training models...\n")

    # ---------------- Train Multiple Models ----------------
    for name in AVAILABLE_MODELS:

        print(f"Training {name}...")

        tuned_params = load_tuned_params(name)

        model = build_model(name, tuned_params)

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        # ---- Cross Validation ----
        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1
        )

        cv_auc = cv_scores.mean()

        # ---- Train final model ----
        pipeline.fit(X_train, y_train)

        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = pipeline.predict(X_test)

        test_auc = roc_auc_score(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)

        results[name] = {
            "cv_auc": float(cv_auc),
            "test_auc": float(test_auc),
            "accuracy": float(acc)
        }

        print(
            f"{name:6} | "
            f"CV-AUC: {cv_auc:.4f} | "
            f"TEST-AUC: {test_auc:.4f} | "
            f"ACC: {acc:.4f}"
        )

        if test_auc > best_auc:
            best_auc = test_auc
            best_pipeline = pipeline
            best_model_name = name

    # ---------------- Save Best Model ----------------
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    model_path = "outputs/models/best_model.pkl"
    joblib.dump(best_pipeline, model_path)

    metrics_path = "outputs/reports/metrics.json"

    with open(metrics_path, "w") as f:
        json.dump({
            "best_model": best_model_name,
            "best_roc_auc": best_auc,
            "all_results": results
        }, f, indent=4)

    # ---------------- Final Report ----------------
    print("\n==============================")
    print("🏆 BEST MODEL:", best_model_name)
    print(f"BEST ROC-AUC: {best_auc:.4f}")
    print("==============================\n")

    print(f"Model saved: {model_path}")
    print(f"Metrics saved: {metrics_path}")


if __name__ == "__main__":
    main()