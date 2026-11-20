import random
import pandas as pd

def add_typo(sentence):
    if len(sentence) < 4:
        return sentence

    i = random.randint(0, len(sentence)-2)
    return sentence[:i] + sentence[i+1] + sentence[i] + sentence[i+2:]

clean = []

with open("data/raw/clean_sentences.txt", "r", encoding="utf-8") as f:
    for line in f:
        line=line.strip()
        if line:
            clean.append(line)

data = []

for s in clean:
    for _ in range(5):   # tạo 5 lỗi / câu
        typo = add_typo(s.lower())
        data.append((typo, s))

df = pd.DataFrame(data, columns=["input","target"])

df.to_csv("data/raw/train_dataset.csv", index=False)

print("Dataset size:", len(df))