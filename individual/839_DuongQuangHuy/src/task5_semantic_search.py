"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DB_DIR = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION_NAME = "DrugLawDocs"
EMBEDDING_MODEL = "text-embedding-3-small"

load_dotenv(PROJECT_ROOT / ".env")


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    if not query.strip() or top_k <= 0:
        return []

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Please set it in .env.")

    if not CHROMA_DB_DIR.exists():
        return []

    # Step 1: Embed query with the same model used in Task 4.
    client = OpenAI(api_key=api_key)
    embedding_response = client.embeddings.create(model=EMBEDDING_MODEL, input=query)
    query_embedding = embedding_response.data[0].embedding

    # Step 2: Query ChromaDB by vector similarity.
    import chromadb

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception:
        return []

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    # Step 3: Normalize output format and sort descending by score.
    outputs: list[dict] = []
    for doc, metadata, distance in zip(documents, metadatas, distances):
        similarity = 1.0 - float(distance)
        outputs.append(
            {
                "content": doc or "",
                "score": similarity,
                "metadata": metadata or {},
            }
        )

    outputs.sort(key=lambda item: item["score"], reverse=True)
    return outputs[:top_k]


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
