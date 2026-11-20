README.md
# lab-03-text-sentiment-rnn

**Lab 03 – Deep Learning Text: Sentiment Analysis with RNN/LSTM**

> ⚠️ Đây là bài tập về nhà. Các hàm được đánh dấu **TODO** cần bạn hoàn thiện.

---

## Overview

Bài toán phân loại cảm xúc nhị phân (positive / negative) trên dataset IMDb.

| | |
|---|---|
| **Dataset** | IMDb Large Movie Review Dataset (Hugging Face) |
| **Nguồn** | [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb) |
| **Bài toán** | Binary sentiment classification (0=negative, 1=positive) |
| **Mô hình** | SentimentRNN (LSTM, PyTorch) — bạn phải xây dựng |
| **Metric** | Accuracy |
| **Export** | TorchScript + vocab.json + label_map.json |
| **Demo** | Gradio |

---

## Cấu trúc

```
lab-03-text-sentiment-rnn/
  README.md
  pyproject.toml
  uv.lock
  config.yaml
  .gitignore
  .github/workflows/smoke.yml
  src/
    lab/
      __init__.py
      config.py          ← (cung cấp sẵn)
      vocab.py           ← TODO: Vocab.build(), encode()
      data.py            ← TODO: load_imdb(), SentimentDataset, get_dataloaders()
      model.py           ← TODO: SentimentRNN.__init__(), forward()
      utils.py           ← TODO: accuracy(), train_one_epoch(), evaluate()
      train.py           ← TODO: main() training loop
      export.py          ← TODO: export_torchscript(), main()
      app_gradio.py      ← TODO: predict(), build_ui(), load_model()
  outputs/
    checkpoints/         # best.pt
    export/              # model.ts + vocab.json + label_map.json
    reports/             # metrics.json
```

---

## Setup

```bash
uv python install 3.10
uv sync
```

---

## Danh sách TODO

### `src/lab/vocab.py`
- [ ] `Vocab.build(texts, max_vocab)` — đếm token, tạo ánh xạ token→idx
- [ ] `Vocab.encode(text, max_len)` — tokenize + map + pad/truncate

### `src/lab/data.py`
- [ ] `load_imdb()` — tải dataset từ Hugging Face
- [ ] `SentimentDataset.__len__()` và `__getitem__()`
- [ ] `get_dataloaders()` — xây vocab, tạo DataLoader, xử lý fast_dev_run

### `src/lab/model.py`
- [ ] `SentimentRNN.__init__()` — Embedding + LSTM + Linear
- [ ] `SentimentRNN.forward()` — forward pass lấy hidden state cuối

### `src/lab/utils.py`
- [ ] `accuracy()` — tính accuracy từ logits
- [ ] `train_one_epoch()` — 1 epoch training với text batch
- [ ] `evaluate()` — eval loop (no_grad)

### `src/lab/train.py`
- [ ] Load data, khởi tạo model, optimizer, criterion
- [ ] Training loop với checkpoint saving
- [ ] Lưu vocab.json, label_map.json, metrics.json

### `src/lab/export.py`
- [ ] `export_torchscript()` — trace LSTM model với dummy input (LongTensor)
- [ ] `main()` — load checkpoint + vocab, export, verify

### `src/lab/app_gradio.py`
- [ ] `load_model()` — load TorchScript
- [ ] `predict()` — encode text → inference → dict {label: prob}
- [ ] `build_ui()` — Gradio Blocks UI với ví dụ mẫu

---

## Train

```bash
# Full train (8 epochs, IMDb 25k)
uv run python -m src.lab.train --config config.yaml

# Fast dev run (1 epoch, 500 samples)
uv run python -m src.lab.train --config config.yaml --fast_dev_run
```

---

## Export

```bash
uv run python -m src.lab.export --config config.yaml
```

---

## Demo (Gradio)

```bash
uv run python -m src.lab.app_gradio --config config.yaml
```

---

## Artifacts

| File | Mô tả |
|------|-------|
| `outputs/checkpoints/best.pt` | Best model state_dict |
| `outputs/export/model.ts` | TorchScript model |
| `outputs/export/vocab.json` | Token → index mapping |
| `outputs/export/label_map.json` | Index → label name |
| `outputs/reports/metrics.json` | Train/val accuracy |

---

## CI (GitHub Actions)

Smoke test khi push/PR lên `main`:
1. `uv sync`
2. Smoke train (1 epoch, 500 samples, vocab 5000)
3. Smoke export (TorchScript)
4. Import check Gradio app

---

## Kiến trúc LSTM gợi ý

```
Input IDs (batch, seq_len)
     ↓
Embedding (vocab_size, emb_dim)        # (batch, seq_len, emb_dim)
     ↓
LSTM (emb_dim → hidden_dim)            # h_n: (num_layers, batch, hidden_dim)
     ↓
h_n[-1]   # last hidden state          # (batch, hidden_dim)
     ↓
Dropout
     ↓
Linear(hidden_dim → 2)
     ↓
Logits (batch, 2)
```

---

## Gợi ý nâng cao

- Thử `arch: gru` (GRU thay vì LSTM).
- Thêm Bidirectional LSTM (`bidirectional=True`).
- Dùng pre-trained word embeddings (GloVe, FastText).
- So sánh với SST-2 dataset (câu ngắn hơn, train nhanh hơn).
- Vẽ confusion matrix và ROC curve.