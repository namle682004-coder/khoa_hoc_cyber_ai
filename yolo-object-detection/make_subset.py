from pathlib import Path
import random
import shutil

SRC = Path("data")        # 👈 folder dataset hiện tại
DST = Path("data_small") # dataset nhỏ mới

N_TRAIN = 5000
N_VAL = 1000
N_TEST = 80

random.seed(42)

def sample_split(split, n):
    imgs = list((SRC / split / "images").glob("*"))
    sampled = random.sample(imgs, min(n, len(imgs)))

    for img in sampled:
        label = SRC / split / "labels" / f"{img.stem}.txt"

        out_img = DST / split / "images"
        out_lbl = DST / split / "labels"

        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        shutil.copy(img, out_img / img.name)
        if label.exists():
            shutil.copy(label, out_lbl / label.name)

for s, n in [("train.sh", N_TRAIN), ("valid", N_VAL), ("test", N_TEST)]:
    sample_split(s, n)

print("✅ Subset created!")
