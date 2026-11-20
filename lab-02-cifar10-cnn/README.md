README.md
# lab-02-cifar10-cnn

**Lab 02 – Deep Learning Image: CIFAR-10 CNN Classification**

> Đây là bài tập về nhà. Các hàm được đánh dấu **TODO** cần bạn hoàn thiện.

---

## Overview

Bài toán phân loại ảnh 10 lớp với dataset CIFAR-10 (60k ảnh 32×32).

| | |
|---|---|
| **Dataset** | CIFAR-10 (via torchvision) |
| **Bài toán** | Multi-class image classification (10 classes) |
| **Mô hình** | SimpleCNN (PyTorch) — bạn phải xây dựng |
| **Metric** | Accuracy |
| **Export** | TorchScript (`.ts`) |
| **Demo** | Gradio |

### CIFAR-10 Classes

`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`

---

## Cấu trúc

```
lab-02-cifar10-cnn/
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
      data.py            ← TODO: get_transforms(), get_dataloaders()
      model.py           ← TODO: SimpleCNN.__init__(), forward()
      utils.py           ← TODO: accuracy(), train_one_epoch(), evaluate()
      train.py           ← TODO: main() training loop
      export.py          ← TODO: export_torchscript(), main()
      app_gradio.py      ← TODO: predict(), build_ui(), load_model()
  outputs/
    checkpoints/         # best.pt
    export/              # model.ts + classes.json
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

### `src/lab/data.py`
- [ ] `get_transforms()` — train augmentation (RandomCrop, RandomHorizontalFlip, Normalize) + val transform
- [ ] `get_dataloaders()` — tải CIFAR-10, tạo DataLoader, xử lý fast_dev_run subset

### `src/lab/model.py`
- [ ] `SimpleCNN.__init__()` — xây dựng Conv blocks + FC layers
- [ ] `SimpleCNN.forward()` — forward pass

### `src/lab/utils.py`
- [ ] `accuracy()` — tính accuracy từ logits
- [ ] `train_one_epoch()` — 1 epoch train loop
- [ ] `evaluate()` — eval loop (no_grad)

### `src/lab/train.py`
- [ ] Load data, khởi tạo model, optimizer, criterion
- [ ] Training loop với checkpoint saving
- [ ] Lưu metrics.json

### `src/lab/export.py`
- [ ] `export_torchscript()` — trace model và save
- [ ] `main()` — load checkpoint, export, save classes.json

### `src/lab/app_gradio.py`
- [ ] `load_model()` — load TorchScript
- [ ] `predict()` — preprocess ảnh + inference + trả về dict
- [ ] `build_ui()` — Gradio Blocks UI

---

## Train

```bash
# Full train
uv run python -m src.lab.train --configs configs.yaml

# Fast dev run (1 epoch, 256 samples)
uv run python -m src.lab.train --configs configs.yaml --fast_dev_run
```

---

## Export

```bash
uv run python -m src.lab.export --configs configs.yaml
```

---

## Demo (Gradio)

```bash
uv run python -m src.lab.app_gradio --configs configs.yaml
```

---

## Artifacts

| File | Mô tả |
|------|-------|
| `outputs/checkpoints/best.pt` | Best model state_dict |
| `outputs/export/model.ts` | TorchScript model |
| `outputs/export/classes.json` | Danh sách 10 class |
| `outputs/reports/metrics.json` | Train/val accuracy |

---

## CI (GitHub Actions)

Smoke test chạy khi push/PR lên `main`:
1. `uv sync`
2. Smoke train (1 epoch, 256 samples)
3. Smoke export (TorchScript)
4. Import check Gradio app

---

## Gợi ý nâng cao

- Thêm Learning Rate Scheduler (CosineAnnealingLR, StepLR).
- Thêm Residual connections (ResBlock).
- Thử Data Augmentation mạnh hơn (CutOut, MixUp).
- Log training curves bằng matplotlib.