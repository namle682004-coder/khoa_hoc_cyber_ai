Chuẩn rồi 👍 — bạn đang đi đúng hướng. Mình viết cho bạn **full flow chạy bằng Ubuntu (WSL)** từ đầu → chạy Qdrant luôn.

---

# 🚀 1. Mở Ubuntu trong WSL

Trong PowerShell:

```powershell
wsl -d Ubuntu-22.04
```

👉 Hoặc đơn giản:

```powershell
wsl
```

---

# 📂 2. Vào project

## ❗ Lưu ý: bạn gõ sai tên folder

Bạn viết:

```bash
cd ~/chapter5-deploy-qrant ❌
```

👉 đúng phải là:

```bash
cd ~/chapter5-deploy-qdrant ✅
```

---

## Nếu chưa copy vào Ubuntu thì làm:

```bash
cp -r /mnt/c/Users/ADMIN/Downloads/Deploy-vllm\&qdrant/chapter5-deploy-qdrant ~/
cd ~/chapter5-deploy-qdrant
```

---

# 🔧 3. Chuẩn bị môi trường (chỉ cần làm 1 lần)

## Cài Docker

```bash
sudo apt update
sudo apt install -y docker.io docker-compose
```

---

## Cho phép chạy docker không cần sudo

```bash
sudo usermod -aG docker $USER
```

👉 Sau đó:

```bash
exit
```

→ mở lại `wsl`

---

## Test docker

```bash
docker ps
```

---

# ⚙️ 4. Setup project

```bash
cd ~/chapter5-deploy-qdrant
cp .env.example .env
```

---

## (Optional) chỉnh API key

```bash
nano .env
```

Ví dụ:

```env
QDRANT_API_KEY=abc123xyz
```

---

# 🚀 5. Chạy Qdrant

```bash
bash scripts/up_qdrant.sh
```

---

# ✅ 6. Check health

```bash
curl http://localhost:6333/healthz
```

---

# 🧩 7. Tạo collection

```bash
bash scripts/create_collection.sh
```

---

# 🔍 8. Check collections

```bash
curl http://localhost:6333/collections -H "api-key: abc123xyz"
```

---

# 📊 Flow tổng (nhớ cái này là đủ)

```bash
wsl
cd ~/chapter5-deploy-qdrant
bash scripts/up_qdrant.sh
bash scripts/create_collection.sh
```

---

# 💥 Tip cực quan trọng (tránh lỗi ngu)

## ❌ Đừng chạy trong:

```bash
/mnt/c/...
```

## ✅ Luôn chạy trong:

```bash
~/...
```

---

# 🎯 Kết luận

👉 Bạn chỉ cần nhớ 4 lệnh chính:

```bash
wsl
cd ~/chapter5-deploy-qdrant
bash scripts/up_qdrant.sh
bash scripts/create_collection.sh
```

→ là chạy full Qdrant xong

---

Nếu bạn muốn đi tiếp đúng hướng:

👉 Mình sẽ guide bạn:

* chunk file PDF
* tạo embedding
* push vào Qdrant
* build RAG thật

Chỉ cần nói:
**“làm tiếp RAG”** 🚀
