# Chapter 5 Base Qdrant Deploy (Vector Store cho RAG)

Repo này chỉ tập trung một việc: self-host Qdrant để lưu trữ vector embeddings cho hệ thống RAG. Không bao gồm LLM serving, không bao gồm ứng dụng chat.

## 1) Mục tiêu của repo

- Dựng nhanh một Qdrant instance ổn định bằng Docker Compose.
- Có volume persistent để dữ liệu vector không mất khi restart container.
- Có script chuẩn cho vận hành: start, stop, logs, health-check, tạo collection.
- Sẵn sàng tích hợp với pipeline chunking + embedding từ Chapter 5.

## 2) Cấu trúc repo

```text
chapter5-deploy-qdrant/
  .env.example
  docker-compose.yml
  config/
    qdrant.yaml
  data/
    qdrant/
      storage/
      snapshots/
  scripts/
    up_qdrant.sh
    down_qdrant.sh
    logs_qdrant.sh
    health_qdrant.sh
    create_collection.sh
```

## 3) Kiến trúc triển khai

- Qdrant HTTP API: port 6333
- Qdrant gRPC: port 6334
- Dữ liệu chính: data/qdrant/storage
- Snapshot backup: data/qdrant/snapshots

Lưu ý: mỗi collection có cấu hình vector size riêng. Vector size phải khớp đúng với embedding model bạn dùng.

## 4) Bước triển khai

### Bước 1 - Chuẩn bị môi trường

Yêu cầu:

- Docker Engine + Docker Compose plugin
- Linux server hoặc máy local để thực hành

```bash
cd chapter5-deploy-qdrant
cp .env.example .env
```

### Bước 2 - Cấu hình .env

Các biến quan trọng:

```bash
QDRANT_HTTP_PORT=6333
QDRANT_GRPC_PORT=6334
QDRANT_BASE_URL=http://localhost:6333
QDRANT_API_KEY=

DEFAULT_COLLECTION=rag_documents
VECTOR_SIZE=384
DISTANCE=Cosine
```

Giải thích nhanh:

- VECTOR_SIZE: số chiều vector embedding.
- DISTANCE: metric tìm kiếm gần nhất (Cosine hoặc Dot hoặc Euclid).
- QDRANT_API_KEY: nên bật khi deploy ngoài localhost hoặc dùng server chung.

### Bước 3 - Khởi động Qdrant

```bash
bash scripts/up_qdrant.sh
```

Script sẽ tự:

1. Chạy Docker Compose up -d
2. Chờ endpoint /healthz sẵn sàng
3. In ra địa chỉ HTTP/gRPC

### Bước 4 - Kiểm tra health

```bash
bash scripts/health_qdrant.sh
```

Hoặc trực tiếp:

```bash
curl http://localhost:6333/healthz
```

### Bước 5 - Tạo collection cho RAG

```bash
# Dùng default trong .env
bash scripts/create_collection.sh

# Hoặc override
bash scripts/create_collection.sh rag_private_docs 768 Cosine
```

### Bước 6 - Theo dõi và dừng dịch vụ

```bash
bash scripts/logs_qdrant.sh
bash scripts/down_qdrant.sh
```

## 5) Checklist để tránh lỗi khi tích hợp RAG

- Embedding dimension phải khớp 100% với VECTOR_SIZE của collection.
- Nếu đổi embedding model (ví dụ 384 -> 768), nên tạo collection mới thay vì tái dùng collection cũ.
- Payload metadata nên lưu đầy đủ (source, chunk_id, year, section, doc_type) để filter tốt hơn.
- Khi chạy production, bật QDRANT_API_KEY và giới hạn network exposure.

## 6) Những yếu tố ảnh hưởng hiệu năng tìm kiếm

### 6.1 Số lượng điểm và payload

Collection càng lớn thì index càng tốn RAM/disk. Cần planning dung lượng sớm.

### 6.2 Metric distance

- Cosine: phổ biến cho hầu hết sentence embedding models.
- Dot: phù hợp khi vector đã được normalize và pipeline yêu cầu inner product.
- Euclid: dùng ít hơn trong semantic retrieval.

### 6.3 Vector dimension

Dimension cao hơn thường tăng chi phí lưu trữ và truy vấn. Chọn đúng theo model, không chọn “to hơn cho chắc”.

### 6.4 Filter complexity

Truy vấn có nhiều điều kiện filter payload sẽ giảm throughput. Cần benchmark theo workload thực tế.

## 7) Qdrant phù hợp để làm gì trong Chapter 5

- Lưu vector của chunk tài liệu công ty.
- Truy vấn top-k semantic search.
- Kết hợp metadata filter (year, segment, doc_type) cho advanced RAG.
- Làm backend vector store cho FastAPI/Gradio RAG app.

## 8) Khuyến nghị production cơ bản

- Dùng ổ đĩa SSD cho thư mục storage.
- Bật API key và reverse proxy (Nginx/Caddy) nếu mở truy cập ngoài.
- Thiết lập snapshot định kỳ để backup.
- Tách môi trường dev/prod bằng file .env riêng.

## 9) Ví dụ tích hợp nhanh từ Python (tham khảo)

```python
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

client = QdrantClient(url="http://localhost:6333")

client.upsert(
    collection_name="rag_documents",
    points=[
        PointStruct(
            id=1,
            vector=[0.1] * 384,
            payload={"source": "policy_2026.pdf", "chunk_id": "c1"},
        )
    ],
)
```

Ghi chú: ví dụ trên chỉ để minh họa. Repo này không ép buộc stack ứng dụng, chỉ phụ trách self-host Qdrant.
