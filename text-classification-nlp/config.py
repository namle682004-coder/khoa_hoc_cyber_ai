import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TFIDF_PATH = os.path.join(BASE_DIR, "checkpoints", "tfidf_model.pkl")
LSTM_MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "lstm_model.pth")