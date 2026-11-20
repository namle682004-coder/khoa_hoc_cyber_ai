import torch
import torch.nn.functional as F
import gradio as gr
from pathlib import Path

from src.lab.model import SentimentRNN


# ===== Load checkpoint =====
CHECKPOINT_PATH = "outputs/checkpoints/best.pt"

checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

config = checkpoint["config"]
stoi = checkpoint["vocab_stoi"]
itos = checkpoint["vocab_itos"]

device = torch.device("cpu")

model = SentimentRNN(
    vocab_size=len(stoi),
    emb_dim=config["model"]["emb_dim"],
    hidden_dim=config["model"]["hidden_dim"],
    num_layers=config["model"]["num_layers"],
    dropout=config["model"]["dropout"],
    bidirectional=config["model"]["bidirectional"],
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()


# ===== Text preprocessing =====
def tokenize(text):
    return text.lower().split()


def numericalize(tokens, max_len):
    ids = [stoi.get(token, stoi.get("<unk>", 0)) for token in tokens]

    if len(ids) < max_len:
        ids += [stoi.get("<pad>", 0)] * (max_len - len(ids))
    else:
        ids = ids[:max_len]

    return torch.tensor(ids).unsqueeze(0)


# ===== Prediction =====
def predict(text):
    tokens = tokenize(text)
    input_tensor = numericalize(tokens, config["data"]["max_len"])

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        confidence = torch.max(probs).item()
        pred = torch.argmax(probs, dim=1).item()

    label = "Positive 😊" if pred == 1 else "Negative 😡"

    return f"{label} (confidence: {confidence:.4f})"


# ===== Gradio UI =====
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Enter a movie review here...",
        label="Movie Review",
    ),
    outputs=gr.Textbox(label="Prediction"),
    title="🎬 IMDB Sentiment Analysis (RNN)",
    description="BiLSTM model trained on IMDB dataset.",
)

if __name__ == "__main__":
    demo.launch()