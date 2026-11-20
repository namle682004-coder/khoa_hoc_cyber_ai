from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch
from .vocab import Vocab


def load_imdb():
    dataset = load_dataset("stanfordnlp/imdb")
    return dataset["train"], dataset["test"]


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_ids = self.vocab.encode(self.texts[idx], self.max_len)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return input_ids, label


def get_dataloaders(config, fast_dev_run=False):
    train_data, test_data = load_imdb()

    train_texts = train_data["text"]
    train_labels = train_data["label"]

    test_texts = test_data["text"]
    test_labels = test_data["label"]

    if fast_dev_run:
        train_texts = train_texts[:500]
        train_labels = train_labels[:500]
        test_texts = test_texts[:500]
        test_labels = test_labels[:500]

    vocab = Vocab.build(train_texts, config["data"]["max_vocab"])

    train_dataset = SentimentDataset(
        train_texts,
        train_labels,
        vocab,
        config["data"]["max_len"],
    )

    test_dataset = SentimentDataset(
        test_texts,
        test_labels,
        vocab,
        config["data"]["max_len"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
    )

    return train_loader, test_loader, vocab