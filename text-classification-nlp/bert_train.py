import torch
import numpy as np
import json
import time
import os

from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# =============================
# METRICS
# =============================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# =============================
# TRAIN FUNCTION
# =============================
def train_model():
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load dataset
    dataset = load_dataset("imdb")

    # Load tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,
        )

    print("Tokenizing dataset...")
    tokenized = dataset.map(tokenize_function, batched=True)

    tokenized = tokenized.remove_columns(["text"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    # Dynamic padding (NHANH HƠN)
    data_collator = DataCollatorWithPadding(tokenizer)

    # Load model
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
    ).to(device)

    training_args = TrainingArguments(
        output_dir="./checkpoints/distilbert",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,  # tăng batch
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),  # dùng mixed precision
        load_best_model_at_end=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    # Save final clean model
    trainer.save_model("./checkpoints/distilbert-final")
    tokenizer.save_pretrained("./checkpoints/distilbert-final")

    print("Evaluating best model...")
    metrics = trainer.evaluate()
    print(metrics)

    os.makedirs("checkpoints", exist_ok=True)

    with open("checkpoints/distilbert_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    training_time = time.time() - start_time

    print("\nTraining completed.")
    print("Final Accuracy:", round(metrics["eval_accuracy"], 4))
    print("Training time:", round(training_time, 2), "seconds")


if __name__ == "__main__":
    train_model()