from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Kết nối Qdrant
client = QdrantClient(
    url="http://localhost:6333",
    api_key="abc123xyz"
)

collection_name = "rag_documents"

# Model embed query
model = SentenceTransformer("all-MiniLM-L6-v2")

# Nhập câu hỏi
query_text = input("Enter your query: ")

# encode phải bọc trong list
query_vector = model.encode([query_text])[0].tolist()

# 🔹 DÙNG search_points() thay cho search()
results = client.search_points(
    collection_name=collection_name,
    query_vector=query_vector,
    limit=3
)

# In kết quả
print("\n=== Retrieved Chunks ===")
for i, r in enumerate(results, 1):
    print(f"{i}. {r.payload['text']}")