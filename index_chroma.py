import json
import os
from typing import List, Dict, Any
import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "data/processed/chunks.jsonl"
CHROMA_DIR = "data/chroma"
COLLECTION_NAME = "opencv_docs"

# fast and good default for RAG
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_chunks(jsonl_path: str) -> List[Dict[str, Any]]:
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        chunks.append(json.loads(line))
    return chunks

def batched(iterable, batch_size: int):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def build_chroma_index():
    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(f"{CHUNKS_PATH} not found. Run build_docs.py")
   
    os.makedirs(CHROMA_DIR, exist_ok=True)   

    # init embedder
    model = SentenceTransformer(EMBED_MODEL_NAME)

    # init Chroma
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )

    # (re)create collection cleanly
    try:
       client.delete_collection(COLLECTION_NAME)
    except Exception:
       pass
    
    collection = client.create_collection(
       name=COLLECTION_NAME,
       metadata={"hnsw:space": "cosine"} # cosine distance
    )

    chunks = load_chunks(CHUNKS_PATH)

    ids = [c["id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [c.get("metadata", {}) for c in chunks]

    # embed + add in batches
    BATCH = 64
    for batch_idx, idxs in enumerate(batched(list(range(len(documents))), BATCH), start=1):
       docs_batch = [documents[i] for i in idxs]
       emb = model.encode(docs_batch, show_progress_bar=False)
       emb = np.asarray(emb, dtype=np.float32).tolist()

       collection.add(
          ids=[ids[i] for i in idxs],
          documents=docs_batch,
          metadatas=[metadatas[i] for i in idxs],
          embeddings=emb
       )

    print(f"Indexed {len(chunks)} chunks into Chroma: {CHROMA_DIR}/{COLLECTION_NAME}")

def retrieve_top_k(query: str, k: int = 5):
    model = SentenceTransformer(EMBED_MODEL_NAME)
    client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
    collection = client.get_collection(COLLECTION_NAME)

    q_emb = model.encode([query], normalize_embeddings=True)
    res = collection.query(query_embeddings=q_emb.tolist(), n_results=k)

    hits = []
    for i in range(len(res["ids"][0])):
        hits.append({
          "id": res["ids"][0][i],
          "text": res["documents"][0][i],
          "metadata": res["metadatas"][0][i],
          "distance": res["distances"][0][i]
        })
    return hits

if __name__ == "__main__":
    build_chroma_index()

    # quick test
    q = "When should I use NORM_HAMMING vs NORM_L2 in BFMatcher?"
    hits = retrieve_top_k(q, k=3)
    for h in hits:
        print("\n---")
        print(h["metadata"].get("doc_title", ""))
        print(h["metadata"].get("source_url", ""))
        print(h["text"][:400], "...")