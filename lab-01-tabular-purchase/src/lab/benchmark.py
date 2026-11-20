import time
import joblib
import onnxruntime as ort
import numpy as np

from src.lab.data import load_data

# Load model
pipeline = joblib.load("outputs/models/best_model.pkl")

# Load data
X, y = load_data()

# Sklearn benchmark
start = time.time()

pipeline.predict_proba(X)

print("Sklearn time:", time.time() - start)

# ---------------- ONNX ----------------

sess = ort.InferenceSession("outputs/export/model.onnx")

# Convert input đúng format ONNX
inputs = {}

for col in X.columns:
    if X[col].dtype == "object":
        inputs[col] = X[col].values.reshape(-1, 1).astype(object)
    else:
        inputs[col] = X[col].values.reshape(-1, 1).astype(np.float32)

# Benchmark
start = time.time()

sess.run(None, inputs)

print("ONNX time:", time.time() - start)