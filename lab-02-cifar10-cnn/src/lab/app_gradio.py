import argparse
import json
import torch
import gradio as gr
from torchvision import transforms
from PIL import Image


# =========================
# Load model + classes
# =========================
def load_model(config_path: str):
    model_path = "outputs/export/model.ts"
    classes_path = "outputs/export/classes.json"

    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()

    with open(classes_path, "r") as f:
        classes = json.load(f)

    return model, classes


# =========================
# Predict function
# =========================
def predict(image):
    if image is None:
        return {"Error": 1.0}

    transform = transforms.Compose([
        transforms.Resize(40),  # scale trước
        transforms.CenterCrop(32),  # crop giữa
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)[0]

    # Convert sang dict cho Gradio Label
    result = {
        classes[i]: float(probs[i])
        for i in range(len(classes))
    }

    return result


# =========================
# Build UI
# =========================
def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("## CIFAR-10 Image Classification")
        gr.Markdown("Upload an image (animal/vehicle) and see prediction.")

        with gr.Row():
            img_input = gr.Image(type="pil", label="Upload Image")
            output = gr.Label(num_top_classes=3, label="Prediction")

        btn = gr.Button("Predict")

        btn.click(
            fn=predict,
            inputs=img_input,
            outputs=output
        )

    return demo


# =========================
# Main
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, required=True)
    args = parser.parse_args()

    model, classes = load_model(args.configs)

    app = build_ui()
    app.launch()