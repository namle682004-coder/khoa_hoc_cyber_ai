# Chapter 4: vLLM Docker -> FastAPI -> Gradio

Repo nay duoc sap lai theo dung logic giang day:

- 4.1 chi self-host vLLM bang mot Dockerfile duy nhat.
- 4.2 moi them FastAPI de boc lai vLLM thanh backend co login, custom system prompt va chat API.
- 4.3 dung Gradio goi backend de demo voi nguoi dung cuoi.

## 1. Cau truc toi gian

```text
chapter4-unified-vllm-fastapi-gradio/
  Dockerfile
  .env.example
  pyproject.toml
  uv.lock
  config/
    model_config.yaml
  scripts/
    deploy_vllm.sh
    deploy_backend.sh
  src/
    __init__.py
    api.py
    auth.py
    config.py
    download_model.py
    gradio_ui.py
    schemas.py
    vllm_client.py
```

## 2. Flow bai giang

### 4.1 Docker cho vLLM

Chi can 1 Dockerfile de build image vLLM, sau do script `scripts/deploy_vllm.sh` se:

- tai bat ky model Hugging Face nao ve thu muc dung chung ngoai project,
- build image vLLM,
- chay container vLLM va mount model vao trong container.

Mac dinh model duoc luu vao shared checkpoint dir:

```bash
$HOME/.cache/huggingface/checkpoints
```

Neu ban muon dung root `/checkpoints`, chi can dat trong `.env`:

```bash
CHECKPOINT_ROOT=/checkpoints
```

Chay mac dinh voi model trong `.env`:

```bash
cd chapter4-unified-vllm-fastapi-gradio
bash scripts/deploy_vllm.sh
```

Chay voi model khac:

```bash
cd chapter4-unified-vllm-fastapi-gradio
bash scripts/deploy_vllm.sh Qwen/Qwen2.5-1.5B-Instruct
```

vLLM se mo tai `http://localhost:8000`.

### 4.2 FastAPI boc lai vLLM

Sau khi vLLM da len, script `scripts/deploy_backend.sh` se chay backend FastAPI local:

```bash
cd chapter4-unified-vllm-fastapi-gradio
bash scripts/deploy_backend.sh
```

Backend mo tai `http://localhost:8100/docs`.

Backend nay gom:

- `POST /auth/login`
- `POST /api/chat`
- `POST /api/summary`
- `POST /api/chat-with-system-prompt`

### 4.3 Gradio demo

Khi backend da chay, mo them terminal khac va chay:

```bash
cd chapter4-unified-vllm-fastapi-gradio
uv run python -m src.gradio_ui
```

Gradio se mo tai `http://localhost:7860`.

## 3. API demo nhanh

Login:

```bash
curl -X POST http://localhost:8100/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"demo123"}'
```

Chat:

```bash
curl -X POST http://localhost:8100/api/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer demo-secret-token" \
  -d '{"user_message":"Write quicksort in Python","temperature":0.2,"max_tokens":256}'
```

Custom system prompt:

```bash
curl -X POST http://localhost:8100/api/chat-with-system-prompt \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer demo-secret-token" \
  -d '{"system_prompt":"You are a Python tutor.","user_message":"Explain recursion with code.","temperature":0.2,"max_tokens":256}'
```

## 4. Luu y

- `HF_TOKEN` la tuy chon, nhung nen dat neu model can quyen truy cap.
- Script vLLM mac dinh dung `--gpus all`, nen may can Docker + NVIDIA runtime.
- `config/model_config.yaml` giu vai tro lam default teaching config, con `.env` la noi de doi nhanh khi demo.
- Mac dinh model cache va checkpoints duoc dat theo kieu shared de nhieu project co the dung chung, khong bi khoa vao folder rieng cua Chapter 4.
