import os
from collections import Counter
import yaml

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

def load_class_names(base_dir):
    yaml_path = os.path.join(base_dir, "data.yaml")

    if not os.path.exists(yaml_path):
        print("⚠ Không tìm thấy data.yaml – chỉ in class_id.")
        return None

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data.get("names")


def count_classes(base_dir):
    splits = ["train", "valid", "test"]

    class_counter = Counter()

    for split in splits:
        label_dir = os.path.join(base_dir, split, "labels")

        if not os.path.exists(label_dir):
            continue

        for file in os.listdir(label_dir):
            if not file.endswith(".txt"):
                continue

            path = os.path.join(label_dir, file)

            if os.path.getsize(path) == 0:
                continue

            with open(path) as f:
                for line in f:
                    cls_id = int(line.strip().split()[0])
                    class_counter[cls_id] += 1

    return class_counter


if __name__ == "__main__":
    base_dir = "data"

    names = load_class_names(base_dir)
    counts = count_classes(base_dir)

    print("\n📊 CLASS STATISTICS")

    for cls_id, num in sorted(counts.items()):
        if names:
            print(f"  {cls_id}: {names[cls_id]} → {num}")
        else:
            print(f"  {cls_id}: {num}")
