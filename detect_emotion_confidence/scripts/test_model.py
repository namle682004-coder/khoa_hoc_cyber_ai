import torch
from src.ml.models.model import EmotionResNet18


def main():

    model = EmotionResNet18()

    x = torch.randn(8, 3, 224, 224)

    y = model(x)

    print("Output shape:", y.shape)


if __name__ == "__main__":
    main()