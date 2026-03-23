import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)

collection = client.get_collection("pptx_rag")

data = collection.get(include=["metadatas"], limit=20)

print("\n--- SAMPLE METADATA ---\n")

for i, meta in enumerate(data["metadatas"], 1):
    print(f"{i}.", meta)