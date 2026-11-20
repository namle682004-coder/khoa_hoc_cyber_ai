import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments
)

model_name = "vinai/bartpho-syllable"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model.to(device)

# load dataset
df = pd.read_csv("data/raw/train_dataset.csv")

dataset = Dataset.from_pandas(df)


def preprocess(example):
    inputs = tokenizer(
        example["input"],
        max_length=64,
        truncation=True,
        padding="max_length"
    )

    targets = tokenizer(
        example["target"],
        max_length=64,
        truncation=True,
        padding="max_length"
    )

    inputs["labels"] = targets["input_ids"]
    return inputs


dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

training_args = TrainingArguments(
    output_dir="models/linguafix",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=3e-5,
    fp16=True,   # dùng GPU FP16
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

trainer.train()

trainer.save_model("models/linguafix")