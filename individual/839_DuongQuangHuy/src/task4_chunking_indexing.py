"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

import os
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STANDARDIZED_DIR = PROJECT_ROOT / "data" / "standardized"
LOCAL_INDEX_SNAPSHOT = PROJECT_ROOT / "data" / "vector_index_snapshot.json"
CHROMA_DB_DIR = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION_NAME = "DrugLawDocs"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# TODO: Chọn chunking strategy và giải thích vì sao
CHUNK_SIZE = 500        # Vì sao chọn 500? ...
CHUNK_OVERLAP = 50      # Vì sao chọn 50? ...
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# TODO: Chọn embedding model và giải thích
# Dùng OpenAI text-embedding-3-small vì ổn định, nhanh và dễ tích hợp khi đã có API key.
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# TODO: Chọn vector store
VECTOR_STORE = "chromadb"  # "weaviate" | "chromadb" | "faiss"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents: list[dict] = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        doc_type = "legal" if "legal" in md_file.parts else "news"
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "type": doc_type,
                    "path": str(md_file.relative_to(PROJECT_ROOT)),
                },
            }
        )
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    if not documents:
        return []

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {**doc["metadata"], "chunk_index": i},
                }
            )
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    if not chunks:
        return []

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Please set it in .env.")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    texts = [c["content"] for c in chunks]

    def batched(items: list[str], size: int = 100) -> Iterator[list[str]]:
        for i in range(0, len(items), size):
            yield items[i : i + size]

    all_embeddings: list[list[float]] = []
    for batch in batched(texts, size=100):
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_embeddings.extend([item.embedding for item in response.data])

    for chunk, emb in zip(chunks, all_embeddings):
        chunk["embedding"] = emb
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    if not chunks:
        print("⚠ No chunks to index.")
        return

    if VECTOR_STORE == "chromadb":
        import chromadb

        CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        ids = []
        documents = []
        embeddings = []
        metadatas = []
        for idx, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            ids.append(
                f"{metadata.get('source', 'doc')}-{metadata.get('chunk_index', 0)}-{idx}"
            )
            documents.append(chunk["content"])
            embeddings.append(chunk["embedding"])
            metadatas.append(
                {
                    "source": metadata.get("source", ""),
                    "doc_type": metadata.get("type", "unknown"),
                    "chunk_index": int(metadata.get("chunk_index", 0)),
                    "path": metadata.get("path", ""),
                }
            )

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        print(f"✓ Indexed {len(ids)} chunks into ChromaDB at: {CHROMA_DB_DIR}")
        return

    if VECTOR_STORE == "weaviate":
        try:
            import weaviate
            from weaviate.classes.config import Configure, DataType, Property

            client = weaviate.connect_to_local()

            if not client.collections.exists(COLLECTION_NAME):
                client.collections.create(
                    name=COLLECTION_NAME,
                    vectorizer_config=Configure.Vectorizer.none(),
                    properties=[
                        Property(name="content", data_type=DataType.TEXT),
                        Property(name="source", data_type=DataType.TEXT),
                        Property(name="doc_type", data_type=DataType.TEXT),
                        Property(name="chunk_index", data_type=DataType.INT),
                        Property(name="path", data_type=DataType.TEXT),
                    ],
                )

            collection = client.collections.get(COLLECTION_NAME)
            with collection.batch.dynamic() as batch:
                for chunk in chunks:
                    metadata = chunk.get("metadata", {})
                    batch.add_object(
                        properties={
                            "content": chunk["content"],
                            "source": metadata.get("source", ""),
                            "doc_type": metadata.get("type", "unknown"),
                            "chunk_index": int(metadata.get("chunk_index", 0)),
                            "path": metadata.get("path", ""),
                        },
                        vector=chunk["embedding"],
                    )
            client.close()
            return
        except Exception as exc:
            print(f"⚠ Weaviate unavailable ({exc}). Saving local snapshot instead.")

    # Generic fallback snapshot keeps progress when selected DB is unavailable.
    import json

    LOCAL_INDEX_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_INDEX_SNAPSHOT.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    print(f"✓ Saved fallback index snapshot: {LOCAL_INDEX_SNAPSHOT}")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
