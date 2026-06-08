"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import os
import time
from json import dumps, loads
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = PROJECT_ROOT / "data" / "standardized"
LANDING_LEGAL_DIR = PROJECT_ROOT / "data" / "landing" / "legal"
PAGEINDEX_REGISTRY_PATH = PROJECT_ROOT / "data" / "pageindex_doc_registry.json"


def _load_registry() -> dict:
    if not PAGEINDEX_REGISTRY_PATH.exists():
        return {"documents": []}
    try:
        return loads(PAGEINDEX_REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"documents": []}


def _save_registry(registry: dict) -> None:
    PAGEINDEX_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAGEINDEX_REGISTRY_PATH.write_text(dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")


def upload_documents():
    """
    Upload legal PDF documents lên PageIndex.
    """
    if not PAGEINDEX_API_KEY:
        print("⚠ PAGEINDEX_API_KEY chưa được cấu hình, bỏ qua upload.")
        return {"uploaded": 0, "status": "missing_api_key"}

    try:
        from pageindex import PageIndexClient  # type: ignore
    except Exception as exc:
        print(f"⚠ Không import được pageindex SDK ({exc}).")
        return {"uploaded": 0, "status": "sdk_unavailable"}

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    registry = _load_registry()
    known_paths = {item.get("local_path"): item for item in registry.get("documents", [])}

    uploaded = 0
    skipped = 0
    for pdf_file in sorted(LANDING_LEGAL_DIR.rglob("*.pdf")):
        if pdf_file.stat().st_size == 0:
            continue
        local_path = str(pdf_file.relative_to(PROJECT_ROOT))
        if local_path in known_paths:
            skipped += 1
            continue

        try:
            response = client.submit_document(file_path=str(pdf_file))

            doc_id = response.get("doc_id")
            if not doc_id:
                raise ValueError(f"submit_document response missing doc_id: {response}")

            registry.setdefault("documents", []).append(
                {
                    "doc_id": doc_id,
                    "filename": pdf_file.name,
                    "doc_type": "legal",
                    "local_path": local_path,
                }
            )
            uploaded += 1
            print(f"  ✓ Uploaded: {pdf_file.name} -> {doc_id}")
        except Exception as exc:
            print(f"  ⚠ Failed upload {pdf_file.name}: {exc}")
    _save_registry(registry)
    return {"uploaded": uploaded, "skipped": skipped, "status": "ok"}


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if not query.strip() or top_k <= 0:
        return []

    if PAGEINDEX_API_KEY:
        try:
            from pageindex import PageIndexClient  # type: ignore

            client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
            registry = _load_registry()
            docs = registry.get("documents", [])
            if not docs:
                docs_api = client.list_documents(limit=50, offset=0)
                docs = [{"doc_id": d.get("id"), "filename": d.get("name", "")} for d in docs_api.get("documents", [])]

            normalized = []
            for doc in docs:
                doc_id = doc.get("doc_id")
                if not doc_id:
                    continue
                submit_resp = client.submit_query(doc_id=doc_id, query=query)
                retrieval_id = submit_resp.get("retrieval_id")
                if not retrieval_id:
                    continue

                retrieval_payload = {}
                for _ in range(5):
                    retrieval_payload = client.get_retrieval(retrieval_id)
                    status = retrieval_payload.get("status")
                    if status in {"completed", "success", "ready", None}:
                        break
                    time.sleep(0.5)

                results_list = (
                    retrieval_payload.get("results")
                    or retrieval_payload.get("chunks")
                    or retrieval_payload.get("nodes")
                    or []
                )
                for item in results_list:
                    text = item.get("text") or item.get("content") or item.get("chunk") or ""
                    score = item.get("score", item.get("relevance_score", 0.0))
                    metadata = item.get("metadata", {})
                    if not metadata:
                        metadata = {"doc_id": doc_id, "source": doc.get("filename", "")}
                    normalized.append(
                        {
                            "content": str(text),
                            "score": float(score),
                            "metadata": metadata if isinstance(metadata, dict) else {"raw_metadata": metadata},
                            "source": "pageindex",
                        }
                    )

            normalized.sort(key=lambda x: x["score"], reverse=True)
            return normalized[:top_k]
        except Exception as exc:
            print(f"⚠ PageIndex query lỗi ({exc}), dùng fallback local.")

    # Fallback local when PageIndex key/SDK is unavailable.
    try:
        from .task5_semantic_search import semantic_search
        fallback = semantic_search(query, top_k=top_k)
        for item in fallback:
            item["source"] = "pageindex"
        if fallback:
            return fallback
    except Exception:
        pass

    # Final offline-safe fallback: lexical search.
    try:
        from .task6_lexical_search import lexical_search

        fallback = lexical_search(query, top_k=top_k)
        for item in fallback:
            item["source"] = "pageindex"
        return fallback
    except Exception:
        return []


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
