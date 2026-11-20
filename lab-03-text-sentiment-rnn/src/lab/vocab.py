from collections import Counter
from typing import List


class Vocab:
    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = itos

    @classmethod
    def build(cls, texts: List[str], max_vocab: int):
        counter = Counter()

        for text in texts:
            tokens = text.lower().split()
            counter.update(tokens)

        most_common = counter.most_common(max_vocab)

        stoi = {"<PAD>": 0, "<UNK>": 1}
        itos = ["<PAD>", "<UNK>"]

        for token, _ in most_common:
            if token not in stoi:
                stoi[token] = len(stoi)
                itos.append(token)

        return cls(stoi, itos)

    def encode(self, text: str, max_len: int):
        tokens = text.lower().split()
        ids = []

        for token in tokens:
            ids.append(self.stoi.get(token, self.stoi["<UNK>"]))

        if len(ids) < max_len:
            ids += [self.stoi["<PAD>"]] * (max_len - len(ids))
        else:
            ids = ids[:max_len]

        return ids