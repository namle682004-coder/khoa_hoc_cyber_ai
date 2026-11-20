import argparse
import os
import pickle
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from lstm_model import LSTMClassifier

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split

from config import TFIDF_PATH

def simple_tokenize(text):
    return text.lower().split()


def train_tfidf():
    start_time = time.time()

    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb")

    texts = dataset["train"]["text"]
    labels = dataset["train"]["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    print("Vectorizing...")
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        stop_words="english",
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    print("Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    print("Evaluating...")
    preds = model.predict(X_val_vec)

    acc = accuracy_score(y_val, preds)
    precision = precision_score(y_val, preds)
    recall = recall_score(y_val, preds)
    f1 = f1_score(y_val, preds)

    training_time = time.time() - start_time

    print("\n===== VALIDATION METRICS =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"Training time: {training_time:.2f}s")

    # ===== CREATE CHECKPOINT DIR FIRST =====
    os.makedirs("checkpoints", exist_ok=True)

    # ===== SAVE MODEL =====
    with open(TFIDF_PATH, "wb") as f:
        pickle.dump((vectorizer, model), f)

    print(f"\nModel saved to {TFIDF_PATH}")

    # ===== SAVE METRICS =====
    metrics = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "training_time_seconds": round(training_time, 2),
    }

    with open("checkpoints/tfidf_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Metrics saved to checkpoints/tfidf_metrics.json")
    print("\nTF-IDF training completed successfully!")

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=300):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def encode(self, text):
        tokens = simple_tokenize(text)
        ids = [self.vocab.get(t, self.vocab["<unk>"]) for t in tokens]
        ids = ids[: self.max_len]
        ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.encode(self.texts[idx]), torch.tensor(self.labels[idx])

def train_lstm():
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb")

    texts = dataset["train"]["text"]
    labels = dataset["train"]["label"]

    # ===== GIỐNG TF-IDF =====
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    print("Building vocabulary...")
    counter = Counter()
    for text in X_train:
        counter.update(simple_tokenize(text))

    vocab = {"<pad>": 0, "<unk>": 1}
    for word, _ in counter.most_common(20000):
        vocab[word] = len(vocab)

    print("Vocab size:", len(vocab))

    train_dataset = TextDataset(X_train, y_train, vocab, max_len=300)
    val_dataset = TextDataset(X_val, y_val, vocab, max_len=300)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)

    model = LSTMClassifier(len(vocab)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    epochs = 6
    best_acc = 0
    best_metrics = {}

    for epoch in range(epochs):
        epoch_start = time.time()

        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        # ===== Validation =====
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                outputs = model(x)
                preds = outputs.argmax(dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(y.numpy())

        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)

        epoch_time = time.time() - epoch_start

        print(
            f"\nEpoch {epoch+1}/{epochs}"
            f"\nAcc: {acc:.4f}"
            f"\nPrecision: {precision:.4f}"
            f"\nRecall: {recall:.4f}"
            f"\nF1: {f1:.4f}"
            f"\nEpoch time: {epoch_time:.2f}s"
        )

        if acc > best_acc:
            best_acc = acc
            best_metrics = {
                "accuracy": float(acc),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
            }

            os.makedirs("checkpoints", exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab,
                },
                "checkpoints/lstm_model.pth",
            )

    training_time = time.time() - start_time
    best_metrics["training_time_seconds"] = round(training_time, 2)

    with open("checkpoints/lstm_metrics.json", "w") as f:
        json.dump(best_metrics, f, indent=4)

    print("\nLSTM training completed.")
    print("Best Accuracy:", round(best_acc, 4))
    print("Training time:", round(training_time, 2), "seconds")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="tfidf",
        choices=["tfidf", "lstm"],
    )
    args = parser.parse_args()

    if args.model == "tfidf":
        train_tfidf()
    elif args.model == "lstm":
        train_lstm()


if __name__ == "__main__":
    main()