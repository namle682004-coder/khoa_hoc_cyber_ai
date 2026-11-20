import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_path = "models/linguafix"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def correct(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=64,
            num_beams=5,
            no_repeat_ngram_size=3
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

while True:
    text = input("Input: ")
    print("Corrected:", correct(text))