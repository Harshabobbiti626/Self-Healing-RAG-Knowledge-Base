import logging
from typing import Any, Dict, List
from src.evaluation.metrics import RAGEvaluator, RAGTriadMetrics
from src.fast_loop.workflow import FastLoopRAGWorkflow

logger = logging.getLogger(__name__)


class RegressionTestRunner:
    """CI/CD test harness running evaluation benchmarks against the RAG system."""

    def __init__(self, workflow: FastLoopRAGWorkflow):
        self.workflow = workflow
        self.evaluator = RAGEvaluator()

    def run_benchmark(
        self,
        test_dataset: List[Dict[str, str]],
        min_pass_ratio: float = 0.80,
    ) -> Dict[str, Any]:
        """Runs full test suite, computing aggregate Triad metrics and pass/fail gate."""
        results = []
        passed_count = 0

        for item in test_dataset:
            question = item["question"]
            state = self.workflow.query(question)

            retrieved = [c.get("content", "") for c in state.get("graded_chunks", [])]
            answer = state.get("generation", "")

            metrics: RAGTriadMetrics = self.evaluator.evaluate_triad(
                query=question,
                retrieved_contexts=retrieved,
                generated_answer=answer,
            )

            if metrics.passed:
                passed_count += 1

            results.append({
                "id": item.get("id"),
                "question": question,
                "metrics": metrics.model_dump(),
                "answer": answer,
                "retrieved_count": len(retrieved),
            })

        total = max(1, len(test_dataset))
        pass_ratio = passed_count / total
        overall_passed = pass_ratio >= min_pass_ratio

        avg_ctx_rel = sum(r["metrics"]["context_relevance"] for r in results) / total
        avg_grounded = sum(r["metrics"]["groundedness"] for r in results) / total
        avg_ans_rel = sum(r["metrics"]["answer_relevance"] for r in results) / total

        report = {
            "total_tests": total,
            "passed_tests": passed_count,
            "pass_ratio": round(pass_ratio, 3),
            "gate_passed": overall_passed,
            "averages": {
                "context_relevance": round(avg_ctx_rel, 3),
                "groundedness": round(avg_grounded, 3),
                "answer_relevance": round(avg_ans_rel, 3),
            },
            "details": results,
        }

        logger.info(
            "Regression benchmark completed: %d/%d passed (ratio: %.2f, gate: %s)",
            passed_count,
            total,
            pass_ratio,
            "PASSED" if overall_passed else "FAILED",
        )
        return report
