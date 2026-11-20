from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import uuid

# 1. Load model embedding
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Connect Qdrant
client = QdrantClient(
    url="http://localhost:6333",
    api_key="abc123xyz"
)

COLLECTION_NAME = "rag_documents"

# 3. Load file
with open("data/sample.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 4. Chunking đơn giản
def chunk_text(text, chunk_size=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

chunks = chunk_text(text)

# 5. Embedding
vectors = model.encode(chunks)

# 6. Insert vào Qdrant
points = []
for i, chunk in enumerate(chunks):
    points.append({
        "id": str(uuid.uuid4()),
        "vector": vectors[i].tolist(),
        "payload": {
            "text": chunk
        }
    })

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print(f"Inserted {len(points)} chunks!")