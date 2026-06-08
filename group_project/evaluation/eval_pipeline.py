"""
RAG Evaluation Pipeline.

Sử dụng DeepEval / RAGAS / TruLens để đánh giá chất lượng RAG pipeline.
Chọn 1 framework và implement đầy đủ.

Yêu cầu:
    1. Load golden_dataset.json (≥15 Q&A pairs)
    2. Chạy RAG pipeline trên từng question
    3. Evaluate với 4 metrics: faithfulness, relevance, context_recall, context_precision
    4. So sánh A/B ít nhất 2 configs
    5. Export results ra results.md
"""

import json
import os
import sys
import traceback
from datetime import datetime
from importlib import util as importlib_util
from pathlib import Path
from typing import Any

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# =============================================================================
# Step 2 — RAG interface adapter for evaluation
# =============================================================================

# Two baseline configs for A/B evaluation.
CONFIG_A = {
    "name": "hybrid_rerank",
    "top_k": 5,
    "score_threshold": 0.3,
    "use_reranking": True,
}

CONFIG_B = {
    "name": "dense_only",
    "top_k": 5,
    "score_threshold": 0.3,
    "use_reranking": False,
}


def get_ab_configs() -> dict[str, dict]:
    """Return standardized A/B configs used by evaluation pipeline."""
    return {
        "config_a": CONFIG_A.copy(),
        "config_b": CONFIG_B.copy(),
    }


def _normalize_sources(raw_sources: Any) -> list[dict]:
    """
    Normalize retrieval sources to a unified schema:
    {'content': str, 'metadata': dict}
    """
    if not isinstance(raw_sources, list):
        return []

    normalized: list[dict] = []
    for item in raw_sources:
        if isinstance(item, dict):
            content = item.get("content", "")
            metadata = item.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {"raw_metadata": metadata}
            normalized.append({"content": str(content), "metadata": metadata})
        else:
            normalized.append({"content": str(item), "metadata": {}})
    return normalized


def normalize_rag_output(raw_output: Any, query: str, config: dict) -> dict:
    """
    Normalize different RAG output formats into a single contract:
    {
        'answer': str,
        'sources': list[dict],  # each has content + metadata
        'retrieval_source': str
    }
    """
    retrieval_source = str(config.get("name", "unknown"))

    if isinstance(raw_output, dict):
        return {
            "answer": str(raw_output.get("answer", "")),
            "sources": _normalize_sources(raw_output.get("sources", [])),
            "retrieval_source": str(raw_output.get("retrieval_source", retrieval_source)),
        }

    # Fallback for string-only generators
    if isinstance(raw_output, str):
        return {
            "answer": raw_output,
            "sources": [],
            "retrieval_source": retrieval_source,
        }

    # Conservative fallback to avoid crashing the eval loop.
    return {
        "answer": "",
        "sources": [],
        "retrieval_source": retrieval_source,
    }


def run_rag(question: str, rag_pipeline: Any, config: dict) -> dict:
    """
    Unified adapter used by evaluation scripts.

    Supported rag_pipeline forms:
    1) Callable: rag_pipeline(question, config=config) or rag_pipeline(question, **config)
    2) Object with generate_with_citation(question, ...)
    """
    if rag_pipeline is None:
        raise ValueError("rag_pipeline is required")

    raw_output = None

    # Case 1: function-style pipeline
    if callable(rag_pipeline):
        try:
            raw_output = rag_pipeline(question, config=config)
        except TypeError:
            try:
                raw_output = rag_pipeline(question, **config)
            except TypeError:
                raw_output = rag_pipeline(question)
    # Case 2: object-style pipeline
    elif hasattr(rag_pipeline, "generate_with_citation"):
        generator = rag_pipeline.generate_with_citation
        try:
            raw_output = generator(question, config=config)
        except TypeError:
            try:
                raw_output = generator(question, top_k=config.get("top_k", 5))
            except TypeError:
                raw_output = generator(question)
    else:
        raise TypeError(
            "rag_pipeline must be callable or expose generate_with_citation(question, ...)"
        )

    normalized = normalize_rag_output(raw_output, question, config)
    if "answer" not in normalized:
        normalized["answer"] = ""
    if "sources" not in normalized:
        normalized["sources"] = []
    if "retrieval_source" not in normalized:
        normalized["retrieval_source"] = str(config.get("name", "unknown"))
    return normalized


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Option 1: DeepEval
# =============================================================================

def evaluate_with_deepeval(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng DeepEval.

    pip install deepeval
    """
    # TODO: Implement
    #
     from deepeval import evaluate
     from deepeval.metrics import (
         FaithfulnessMetric,
         AnswerRelevancyMetric,
         ContextualRecallMetric,
         ContextualPrecisionMetric,
     )
     from deepeval.test_case import LLMTestCase
    
     test_cases = []
     for item in golden_dataset:
         result = rag_pipeline.generate_with_citation(item["question"])
         test_case = LLMTestCase(
             input=item["question"],
         actual_output=result["answer"],
             expected_output=item["expected_answer"],
             retrieval_context=[c["content"] for c in result["sources"]],
         )
         test_cases.append(test_case)
    
     metrics = [
         FaithfulnessMetric(threshold=0.7),
         AnswerRelevancyMetric(threshold=0.7),
         ContextualRecallMetric(threshold=0.7),
         ContextualPrecisionMetric(threshold=0.7),
     ]
    
     results = evaluate(test_cases, metrics)
     return results
    raise NotImplementedError("Implement evaluate_with_deepeval")


# =============================================================================
# Option 2: RAGAS
# =============================================================================

def evaluate_with_ragas(
    rag_pipeline, golden_dataset: list[dict], config: dict | None = None
) -> dict:
    """
    Evaluate RAG pipeline sử dụng RAGAS.

    pip install ragas
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing RAGAS dependencies. Run: "
            "pip install -r requirements.txt (or pip install ragas langchain-community)"
        ) from exc

    config = (config or CONFIG_A).copy()
    eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    run_details: list[dict] = []

    for item in golden_dataset:
        question = str(item.get("question", ""))
        expected_answer = str(item.get("expected_answer", ""))

        rag_output = run_rag(question, rag_pipeline, config=config)
        contexts = [
            str(source.get("content", ""))
            for source in rag_output.get("sources", [])
            if isinstance(source, dict) and str(source.get("content", "")).strip()
        ]

        # Keep schema stable for RAGAS even when retrieval returns empty.
        if not contexts:
            contexts = [""]

        answer = str(rag_output.get("answer", "")).strip()
        if not answer:
            answer = "Tôi không thể xác minh thông tin này từ nguồn hiện có."

        eval_data["question"].append(question)
        eval_data["answer"].append(answer)
        eval_data["contexts"].append(contexts)
        eval_data["ground_truth"].append(expected_answer)
        run_details.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "expected_answer": expected_answer,
                "retrieval_source": rag_output.get("retrieval_source", config["name"]),
            }
        )

    dataset = Dataset.from_dict(eval_data)
    ragas_result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )
    result_df = ragas_result.to_pandas()

    # Derive summary scores robustly across ragas versions.
    score_columns = [
        col
        for col in ("faithfulness", "answer_relevancy", "context_recall", "context_precision")
        if col in result_df.columns
    ]
    summary_scores = {
        col: float(result_df[col].mean()) for col in score_columns if not result_df[col].empty
    }
    if summary_scores:
        summary_scores["average"] = sum(summary_scores.values()) / len(summary_scores)

    return {
        "framework": "ragas",
        "config": config["name"],
        "num_samples": len(golden_dataset),
        "scores": summary_scores,
        "details": run_details,
        "result_table": result_df.to_dict(orient="records"),
    }


# =============================================================================
# Option 3: TruLens
# =============================================================================

def evaluate_with_trulens(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng TruLens.

    pip install trulens
    """
    # TODO: Implement
    #
    from trulens.apps.custom import TruCustomApp
     from trulens.core import Feedback
     from trulens.providers.openai import OpenAI as TruOpenAI
    
     provider = TruOpenAI()
    
     f_faithfulness = Feedback(provider.groundedness_measure_with_cot_reasons).on_output()
     f_relevance = Feedback(provider.relevance).on_input_output()
     f_context_relevance = Feedback(provider.context_relevance).on_input()
    
     tru_rag = TruCustomApp(
         rag_pipeline,
         app_name="DrugLaw_RAG",
         feedbacks=[f_faithfulness, f_relevance, f_context_relevance],
     )
    
     with tru_rag as recording:
         for item in golden_dataset:
             rag_pipeline.generate_with_citation(item["question"])
    
    # Dashboard: from trulens.dashboard import run_dashboard; run_dashboard()
    raise NotImplementedError("Implement evaluate_with_trulens")


# =============================================================================
# A/B Comparison
# =============================================================================

def compare_configs(rag_pipeline, golden_dataset: list[dict]):
    """
    So sánh A/B giữa ít nhất 2 configs.

    Gợi ý configs để so sánh:
    - Config A: hybrid search + reranking
    - Config B: dense-only (không reranking)
    - Config C: hybrid search + PageIndex fallback
    """
    configs = get_ab_configs()
    config_a = configs["config_a"]
    config_b = configs["config_b"]

    result_a = evaluate_with_ragas(rag_pipeline, golden_dataset, config=config_a)
    result_b = evaluate_with_ragas(rag_pipeline, golden_dataset, config=config_b)

    metric_names = ("faithfulness", "answer_relevancy", "context_recall", "context_precision")
    deltas: dict[str, float] = {}
    for metric in metric_names:
        a_score = float(result_a.get("scores", {}).get(metric, 0.0))
        b_score = float(result_b.get("scores", {}).get(metric, 0.0))
        deltas[metric] = a_score - b_score

    avg_a = float(result_a.get("scores", {}).get("average", 0.0))
    avg_b = float(result_b.get("scores", {}).get("average", 0.0))
    delta_average = avg_a - avg_b

    if delta_average > 0:
        winner = config_a["name"]
    elif delta_average < 0:
        winner = config_b["name"]
    else:
        winner = "tie"

    return {
        "framework": "ragas",
        "config_a": {
            "params": config_a,
            "scores": result_a.get("scores", {}),
            "num_samples": result_a.get("num_samples", len(golden_dataset)),
        },
        "config_b": {
            "params": config_b,
            "scores": result_b.get("scores", {}),
            "num_samples": result_b.get("num_samples", len(golden_dataset)),
        },
        "delta": {
            **deltas,
            "average": delta_average,
        },
        "winner": winner,
        "raw_results": {
            config_a["name"]: result_a,
            config_b["name"]: result_b,
        },
    }


# =============================================================================
# Export Results
# =============================================================================

def export_results(results: dict, comparison: dict):
    """Export evaluation results to results.md"""
    metric_rows = [
        ("Faithfulness", "faithfulness"),
        ("Answer Relevance", "answer_relevancy"),
        ("Context Recall", "context_recall"),
        ("Context Precision", "context_precision"),
        ("Average", "average"),
    ]
    config_a = comparison.get("config_a", {})
    config_b = comparison.get("config_b", {})
    delta = comparison.get("delta", {})

    lines: list[str] = []
    lines.append("# RAG Evaluation Results")
    lines.append("")
    lines.append(f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"- Framework: `{comparison.get('framework', 'ragas')}`")
    lines.append(f"- Samples: `{results.get('num_samples', 0)}`")
    lines.append("")
    lines.append("## Overall Scores")
    lines.append("")
    lines.append("| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ (A-B) |")
    lines.append("|--------|---------------------------:|----------------------:|--------:|")
    for label, key in metric_rows:
        a = float(config_a.get("scores", {}).get(key, 0.0))
        b = float(config_b.get("scores", {}).get(key, 0.0))
        d = float(delta.get(key, a - b))
        lines.append(f"| {label} | {a:.4f} | {b:.4f} | {d:.4f} |")

    lines.append("")
    lines.append("## A/B Comparison Analysis")
    lines.append("")
    lines.append(f"- Config A params: `{config_a.get('params', {})}`")
    lines.append(f"- Config B params: `{config_b.get('params', {})}`")
    lines.append(f"- Winner: `{comparison.get('winner', 'tie')}`")
    lines.append("")

    # Bottom 3 by mean metric score if available.
    result_table = results.get("result_table", []) or []
    metric_keys = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    worst_rows = []
    for idx, row in enumerate(result_table):
        values = [float(row.get(k, 0.0)) for k in metric_keys if row.get(k) is not None]
        if not values:
            continue
        worst_rows.append((idx, sum(values) / len(values), row))
    worst_rows.sort(key=lambda x: x[1])
    worst_rows = worst_rows[:3]

    lines.append("## Worst Performers (Bottom 3)")
    lines.append("")
    lines.append("| # | Question | Faithfulness | Relevance | Recall | Precision |")
    lines.append("|---|----------|-------------:|----------:|-------:|----------:|")
    details = results.get("details", []) or []
    for rank, (idx, _score, row) in enumerate(worst_rows, 1):
        question = ""
        if idx < len(details):
            question = str(details[idx].get("question", ""))
        question = question.replace("|", "\\|")
        lines.append(
            "| {rank} | {q} | {f:.4f} | {r:.4f} | {rc:.4f} | {p:.4f} |".format(
                rank=rank,
                q=question[:120] + ("..." if len(question) > 120 else ""),
                f=float(row.get("faithfulness", 0.0)),
                r=float(row.get("answer_relevancy", 0.0)),
                rc=float(row.get("context_recall", 0.0)),
                p=float(row.get("context_precision", 0.0)),
            )
        )

    lines.append("")
    lines.append("## Recommendations")
    lines.append("")
    lines.append("1. Tăng chất lượng retriever bằng tuning ngưỡng fallback và top_k theo từng loại câu hỏi.")
    lines.append("2. Cải thiện prompt generation để bắt buộc citation theo từng claim cụ thể.")
    lines.append("3. Phân tích các mẫu bottom-3 để bổ sung dữ liệu/normalize metadata nguồn.")
    lines.append("")

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")


def _load_default_rag_pipeline():
    """
    Load team pipeline function dynamically.
    Priority:
    1) env RAG_PIPELINE_FILE + RAG_PIPELINE_SYMBOL
    2) src.task10_generation.generate_with_citation (root)
    """
    pipeline_file = os.getenv("RAG_PIPELINE_FILE", "").strip()
    pipeline_symbol = os.getenv("RAG_PIPELINE_SYMBOL", "generate_with_citation").strip()

    if pipeline_file:
        module_path = Path(pipeline_file)
        if not module_path.is_absolute():
            module_path = PROJECT_ROOT / module_path
        spec = importlib_util.spec_from_file_location("rag_pipeline_dynamic", module_path)
        if spec and spec.loader:
            mod = importlib_util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, pipeline_symbol):
                return getattr(mod, pipeline_symbol)
            raise AttributeError(f"Symbol '{pipeline_symbol}' not found in {module_path}")

    # Prefer first individual pipeline found under individual/*/src/task10_generation.py
    individual_candidates = sorted((PROJECT_ROOT / "individual").glob("*/src/task10_generation.py"))
    if individual_candidates:
        module_path = individual_candidates[0]
        spec = importlib_util.spec_from_file_location("rag_pipeline_individual", module_path)
        if spec and spec.loader:
            module_dir = str(module_path.parent)
            if module_dir not in sys.path:
                sys.path.insert(0, module_dir)
            mod = importlib_util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "generate_with_citation"):
                return getattr(mod, "generate_with_citation")

    try:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from src.task10_generation import generate_with_citation

        return generate_with_citation
    except Exception as exc:
        raise RuntimeError(
            "Unable to load default RAG pipeline. "
            "Set RAG_PIPELINE_FILE and RAG_PIPELINE_SYMBOL."
        ) from exc


if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")
    print("A/B configs:", get_ab_configs())
    try:
        pipeline = _load_default_rag_pipeline()
        # Baseline run uses config A.
        results = evaluate_with_ragas(pipeline, golden_dataset, config=CONFIG_A)
        comparison = compare_configs(pipeline, golden_dataset)
        export_results(results, comparison)
        print(f"✓ Evaluation completed. Results written to: {RESULTS_PATH}")
    except Exception as exc:
        print(f"✗ Evaluation failed: {exc}")
        traceback.print_exc()
