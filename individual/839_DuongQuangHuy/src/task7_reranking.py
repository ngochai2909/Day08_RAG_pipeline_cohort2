"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.
"""

import os
import re
from math import sqrt

import requests


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def _cosine_sim(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sqrt(sum(a * a for a in vec_a))
    norm_b = sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _jaccard_sim(text_a: str, text_b: str) -> float:
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(tokens_a & tokens_b) / len(union)


def _simple_relevance(query: str, content: str, base_score: float) -> float:
    overlap = _jaccard_sim(query, content)
    return 0.6 * overlap + 0.4 * float(base_score)


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    if not candidates or top_k <= 0:
        return []

    jina_api_key = os.getenv("JINA_API_KEY", "").strip()
    if jina_api_key:
        try:
            response = requests.post(
                "https://api.jina.ai/v1/rerank",
                headers={"Authorization": f"Bearer {jina_api_key}"},
                json={
                    "model": "jina-reranker-v2-base-multilingual",
                    "query": query,
                    "documents": [c.get("content", "") for c in candidates],
                    "top_n": min(top_k, len(candidates)),
                },
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
            reranked = payload.get("results", [])
            outputs: list[dict] = []
            for item in reranked:
                idx = int(item["index"])
                base = candidates[idx].copy()
                base["score"] = float(item.get("relevance_score", base.get("score", 0.0)))
                outputs.append(base)
            if outputs:
                return outputs[:top_k]
        except Exception:
            # Fall back to local heuristic when API is unavailable.
            pass

    # Local fallback: combine lexical overlap with existing retrieval score.
    reranked = []
    for candidate in candidates:
        base_score = float(candidate.get("score", 0.0))
        combined = _simple_relevance(query, candidate.get("content", ""), base_score)
        item = candidate.copy()
        item["score"] = combined
        reranked.append(item)
    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    if not candidates or top_k <= 0:
        return []

    selected_indices: list[int] = []
    remaining = list(range(len(candidates)))

    while remaining and len(selected_indices) < top_k:
        best_idx = remaining[0]
        best_score = float("-inf")

        for idx in remaining:
            cand = candidates[idx]
            cand_embedding = cand.get("embedding")

            if isinstance(cand_embedding, list) and query_embedding:
                relevance = _cosine_sim(query_embedding, cand_embedding)
            else:
                relevance = float(cand.get("score", 0.0))

            max_sim_to_selected = 0.0
            for sel_idx in selected_indices:
                selected = candidates[sel_idx]
                selected_embedding = selected.get("embedding")
                if isinstance(cand_embedding, list) and isinstance(selected_embedding, list):
                    sim = _cosine_sim(cand_embedding, selected_embedding)
                else:
                    sim = _jaccard_sim(cand.get("content", ""), selected.get("content", ""))
                max_sim_to_selected = max(max_sim_to_selected, sim)

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    results = []
    for idx in selected_indices:
        item = candidates[idx].copy()
        item["score"] = float(item.get("score", 0.0))
        results.append(item)
    return results


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    if not ranked_lists or top_k <= 0:
        return []

    rrf_scores: dict[str, float] = {}
    item_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = str(item.get("content", ""))
            if not key:
                continue
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            item_map[key] = item

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    results: list[dict] = []
    for content, score in merged[:top_k]:
        candidate = item_map[content].copy()
        candidate["score"] = float(score)
        results.append(candidate)
    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if not candidates or top_k <= 0:
        return []

    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "mmr":
        # Without query embedding at this layer, MMR falls back to candidate scores.
        return rerank_mmr(query_embedding=[], candidates=candidates, top_k=top_k)
    if method == "rrf":
        # Single-list RRF fallback to keep interface simple.
        return rerank_rrf([candidates], top_k=top_k)
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
