import time
import pickle
import torch
import torch.nn.functional as F
import gradio as gr
import transformers

from transformers import BertForSequenceClassification, DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers.models.bert.tokenization_bert import BertTokenizerFast

from config import TFIDF_PATH, LSTM_MODEL_PATH
from lstm_model import LSTMClassifier


# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ================= LOAD TF-IDF =================
with open(TFIDF_PATH, "rb") as f:
    vectorizer, tfidf_model = pickle.load(f)


# ================= LOAD LSTM =================
checkpoint = torch.load(LSTM_MODEL_PATH, map_location=device)
vocab = checkpoint["vocab"]
lstm_model = LSTMClassifier(len(vocab)).to(device)
lstm_model.load_state_dict(checkpoint["model_state_dict"])
lstm_model.eval()


# ================= LOAD BERT =================

bert_path = "./checkpoints/distilbert-final"

bert_tokenizer = DistilBertTokenizerFast.from_pretrained(bert_path)
bert_model = DistilBertForSequenceClassification.from_pretrained(bert_path)
bert_model.to(device)
bert_model.eval()

# ================= ENCODE LSTM =================
def encode_lstm(text, max_len=300):
    tokens = text.lower().split()
    ids = [vocab.get(t, vocab["<unk>"]) for t in tokens]
    ids = ids[:max_len]

    if len(ids) < max_len:
        ids += [vocab["<pad>"]] * (max_len - len(ids))

    return torch.tensor(ids).unsqueeze(0).to(device)

def predict(text):
    if not text.strip():
        return [""] * 9

    # ================= TF-IDF =================
    start = time.time()
    X = vectorizer.transform([text])
    pred_tfidf = tfidf_model.predict(X)[0]
    prob_tfidf = tfidf_model.predict_proba(X)[0]
    tfidf_time = (time.time() - start) * 1000

    sentiment_tfidf = "Positive" if pred_tfidf == 1 else "Negative"
    conf_tfidf = float(max(prob_tfidf))

    # ================= LSTM =================
    start = time.time()
    input_tensor = encode_lstm(text).to(device)

    with torch.no_grad():
        outputs = lstm_model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        pred_lstm = torch.argmax(probs, dim=1).item()
        conf_lstm = probs[0][pred_lstm].item()

    lstm_time = (time.time() - start) * 1000
    sentiment_lstm = "Positive" if pred_lstm == 1 else "Negative"

    # ================= BERT =================
    start = time.time()
    inputs = bert_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256,
    ).to(device)

    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred_bert = torch.argmax(probs, dim=1).item()
        conf_bert = probs[0][pred_bert].item()

    bert_time = (time.time() - start) * 1000
    sentiment_bert = "Positive" if pred_bert == 1 else "Negative"

    # ================= HIGHLIGHT =================
    confidences = {
        "tfidf": conf_tfidf,
        "lstm": conf_lstm,
        "bert": conf_bert,
    }

    times = {
        "tfidf": tfidf_time,
        "lstm": lstm_time,
        "bert": bert_time,
    }

    max_conf_model = max(confidences, key=confidences.get)
    min_time_model = min(times, key=times.get)

    def format_conf(model, value):
        value = round(value, 4)
        if model == max_conf_model:
            return f"<span style='color:green;font-weight:bold'>{value} ⭐</span>"
        return f"{value}"

    def format_time(model, value):
        value = round(value, 2)
        if model == min_time_model:
            return f"<span style='color:green;font-weight:bold'>{value} ms ⚡</span>"
        return f"{value} ms"

    return (
        sentiment_tfidf,
        format_conf("tfidf", conf_tfidf),
        format_time("tfidf", tfidf_time),

        sentiment_lstm,
        format_conf("lstm", conf_lstm),
        format_time("lstm", lstm_time),

        sentiment_bert,
        format_conf("bert", conf_bert),
        format_time("bert", bert_time),
    )


with gr.Blocks(title="IMDB Sentiment Analysis - 3 Model Comparison") as demo:

    gr.Markdown("# IMDB Sentiment Analysis - 3 Model Comparison")
    gr.Markdown("Comparison between TF-IDF, LSTM, and BERT")

    text_input = gr.Textbox(
        lines=6,
        placeholder="Enter movie review here..."
    )

    with gr.Row():
        with gr.Column():
            tfidf_label = gr.Label(label="TF-IDF Sentiment")
            tfidf_conf = gr.Markdown()
            tfidf_time = gr.Markdown()

        with gr.Column():
            lstm_label = gr.Label(label="LSTM Sentiment")
            lstm_conf = gr.Markdown()
            lstm_time = gr.Markdown()

        with gr.Column():
            bert_label = gr.Label(label="BERT Sentiment")
            bert_conf = gr.Markdown()
            bert_time = gr.Markdown()

    analyze_btn = gr.Button("Submit")

    analyze_btn.click(
        fn=predict,
        inputs=text_input,
        outputs=[
            tfidf_label, tfidf_conf, tfidf_time,
            lstm_label, lstm_conf, lstm_time,
            bert_label, bert_conf, bert_time,
        ]
    )

if __name__ == "__main__":
    demo.launch()