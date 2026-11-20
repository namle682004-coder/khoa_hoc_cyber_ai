from src.ml.data.loader import create_dataloaders


def main():

    train_loader, test_loader = create_dataloaders(
        train_dir="data/raw/fer2013/train",
        test_dir="data/raw/fer2013/test",
        batch_size=32
    )

    for images, labels in train_loader:

        print("Images shape:", images.shape)
        print("Labels shape:", labels.shape)

        break


if __name__ == "__main__":
    main()