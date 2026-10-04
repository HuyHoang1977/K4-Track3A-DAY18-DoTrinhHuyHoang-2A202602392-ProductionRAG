from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (TEST_SET_PATH, GEMINI_API_KEY, GEMINI_MODEL,
                    GEMINI_EMBEDDING_MODEL, GEMINI_OPENAI_BASE_URL)


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class _GeminiEmbeddings:
    """LangChain-compatible embeddings dùng Gemini native SDK.

    OpenAI-compat endpoint của Gemini KHÔNG implement /embeddings → 501
    UNIMPLEMENTED. Nên phải gọi SDK google-genai rồi đóng gói lại theo
    interface mà RAGAS cần (embed_query / embed_documents).
    """

    def __init__(self, api_key: str, model: str):
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def _embed(self, text: str) -> list[float]:
        # Embeddings cũng ăn quota → throttle + tôn trọng quota_exhausted,
        # nếu không RAGAS sẽ bắn hàng loạt embed call và dính 429.
        from src.m5_enrichment import _throttle, quota_was_exhausted
        if quota_was_exhausted():
            raise RuntimeError("Gemini quota exhausted")
        _throttle()
        response = self._client.models.embed_content(model=self._model, contents=text)
        return response.embeddings[0].values

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]


def _build_ragas_llms():
    """Tạo LLM + embedding cho RAGAS. Ưu tiên Gemini, fallback OpenAI."""
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    class SingleCandidateChatOpenAI(ChatOpenAI):
        """Ép n=1 + throttle khi gọi Gemini qua OpenAI-compat endpoint.

        - Gemini từ chối nhiều candidate: 400 "Multiple candidates is not enabled
          for this model". RAGAS 0.1.22 truyền n>1 → metric lỗi → về 0.
        - RAGAS bắn ~80 LLM call gần như song song, vượt quota 15 RPM của free
          tier. RunConfig.max_workers chỉ hạ concurrency chứ không pacing, nên
          phải chèn nghỉ tối thiểu giữa các request.
        """

        def _get_request_payload(self, input_, *, stop=None, **kwargs):
            from src.m5_enrichment import _throttle
            _throttle()
            payload = super()._get_request_payload(input_, stop=stop, **kwargs)
            payload["n"] = 1
            return payload

    if GEMINI_API_KEY:
        client = lambda **kw: SingleCandidateChatOpenAI(
            model=GEMINI_MODEL, api_key=GEMINI_API_KEY,
            base_url=GEMINI_OPENAI_BASE_URL, **kw)
        embeddings = _GeminiEmbeddings(GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL)
    else:
        from config import OPENAI_API_KEY
        if not OPENAI_API_KEY:
            raise RuntimeError("Cần GEMINI_API_KEY hoặc OPENAI_API_KEY để chạy RAGAS")
        client = lambda **kw: ChatOpenAI(model="gpt-4o-mini",
                                         api_key=OPENAI_API_KEY, **kw)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small",
                                      api_key=OPENAI_API_KEY)

    return (LangchainLLMWrapper(client(temperature=0)),
            LangchainEmbeddingsWrapper(embeddings))


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    zeros = {"faithfulness": 0.0, "answer_relevancy": 0.0,
             "context_precision": 0.0, "context_recall": 0.0, "per_question": []}

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (answer_relevancy, context_precision,
                                   context_recall, faithfulness)

        # RAGAS bắn ~80 job cùng lúc → vượt quota 15 RPM free tier.
        # max_workers=2 giảm burst; pacing thật do _throttle trong
        # SingleCandidateChatOpenAI._get_request_payload đảm nhiệm.
        run_config = None
        try:
            from ragas.executor import RunConfig
            run_config = RunConfig(max_workers=2, max_retries=5, max_wait=30)
        except Exception as e:
            print(f"  ⚠️  Không tạo được RunConfig: {e}")

        llm, embed_llm = _build_ragas_llms()

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })

        extra = {"run_config": run_config} if run_config else {}
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                            context_precision, context_recall],
                          llm=llm, embeddings=embed_llm, **extra)
        df = result.to_pandas()

        def _score(row, name: str) -> float:
            """RAGAS có thể trả None/NaN cho câu hỏi lỗi → coi như 0.0."""
            value = row.get(name, 0.0)
            try:
                number = float(value)
            except (TypeError, ValueError):
                return 0.0
            return 0.0 if number != number else number  # NaN check

        per_question = [
            EvalResult(
                question=row["question"], answer=row["answer"],
                contexts=row["contexts"], ground_truth=row["ground_truth"],
                faithfulness=_score(row, "faithfulness"),
                answer_relevancy=_score(row, "answer_relevancy"),
                context_precision=_score(row, "context_precision"),
                context_recall=_score(row, "context_recall"),
            )
            for _, row in df.iterrows()
        ]

        def _mean(name: str) -> float:
            scores = [getattr(r, name) for r in per_question]
            return sum(scores) / len(scores) if scores else 0.0

        return {"faithfulness": _mean("faithfulness"),
                "answer_relevancy": _mean("answer_relevancy"),
                "context_precision": _mean("context_precision"),
                "context_recall": _mean("context_recall"),
                "per_question": per_question}

    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return zeros


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }

    rows: list[dict] = []
    for r in eval_results:
        scores = {
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "context_precision": r.context_precision,
            "context_recall": r.context_recall,
        }
        avg_score = sum(scores.values()) / len(scores)
        # min() trả về key đầu tiên khi hòa → deterministic theo thứ tự khai báo.
        worst_metric = min(scores, key=lambda m: scores[m])
        diagnosis, suggested_fix = diagnostic_tree.get(
            worst_metric, ("Unknown issue", "Inspect manually"))

        rows.append({
            "question": r.question,
            "worst_metric": worst_metric,
            "score": round(avg_score, 4),
            "metric_scores": {k: round(v, 4) for k, v in scores.items()},
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    rows.sort(key=lambda x: x["score"])   # tệ nhất lên đầu
    return rows[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
