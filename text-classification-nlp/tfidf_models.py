# tfidf_models.py

import pickle
import torch
import torch.nn as nn


# ======================
# TF-IDF + Logistic
# ======================

class TfidfModel:
    def __init__(self, model_path):
        with open(model_path, "rb") as f:
            self.vectorizer, self.model = pickle.load(f)

    def predict(self, text):
        X = self.vectorizer.transform([text])
        pred = self.model.predict(X)[0]
        prob = self.model.predict_proba(X)[0]
        return int(pred), prob


# ======================
# (Giữ nguyên LSTM/BERT nếu đã có)
# ======================