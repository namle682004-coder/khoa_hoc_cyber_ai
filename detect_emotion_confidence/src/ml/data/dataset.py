from pathlib import Path
from typing import Callable, List, Tuple
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FER2013Dataset(Dataset):
    """
    Dataset loader for FER2013-style directory dataset.

    Directory structure expected:
        root/
            angry/
            disgust/
            fear/
            happy/
            sad/
            surprise/
            neutral/
    """

    EMOTION_MAP = {
        "angry": 0,
        "disgust": 1,
        "fear": 2,
        "happy": 3,
        "sad": 4,
        "surprise": 5,
        "neutral": 6
    }

    def __init__(
        self,
        root_dir: str,
        transform: Callable | None = None
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.image_paths: List[Path] = []
        self.labels: List[int] = []

        self._load_dataset()

    def _load_dataset(self):

        for class_dir in sorted(self.root_dir.iterdir()):

            if not class_dir.is_dir():
                continue

            emotion = class_dir.name.lower()

            if emotion not in self.EMOTION_MAP:
                continue

            label = self.EMOTION_MAP[emotion]

            for img_path in class_dir.glob("*"):

                if img_path.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
                    continue

                self.image_paths.append(img_path)
                self.labels.append(label)

        print(f"Loaded {len(self.image_paths)} images from {self.root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple:

        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# -------------------------
# Transforms
# -------------------------

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),

    transforms.ToTensor(),
])


val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])