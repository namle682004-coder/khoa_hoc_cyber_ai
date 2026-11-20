import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "vinai/bartpho-syllable"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

text = "toi ko biet lam sao"

inputs = tokenizer(text, return_tensors="pt").to(device)

outputs = model.generate(
    **inputs,
    max_length=50,
    num_beams=5,
    repetition_penalty=2.0,
    no_repeat_ngram_size=3,
    early_stopping=True
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))