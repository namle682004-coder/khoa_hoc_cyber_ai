import torch
from pathlib import Path
from .model import SentimentRNN


def main():
    checkpoint_path = "outputs/checkpoints/best.pt"
    export_path = "outputs/export/model.pt"

    Path("outputs/export").mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    config = checkpoint["config"]
    vocab_stoi = checkpoint["vocab_stoi"]
    vocab_itos = checkpoint["vocab_itos"]

    model = SentimentRNN(
        vocab_size=len(vocab_stoi),
        emb_dim=config["model"]["emb_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        bidirectional=config["model"]["bidirectional"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "vocab_stoi": vocab_stoi,
            "vocab_itos": vocab_itos,
        },
        export_path,
    )

    print(f"Exported model to {export_path}")


if __name__ == "__main__":
    main()