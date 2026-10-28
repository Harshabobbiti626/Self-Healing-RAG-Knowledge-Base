import re
from typing import Dict, List
from pydantic import BaseModel, Field


class RAGTriadMetrics(BaseModel):
    """Core RAG Triad quality metrics for continuous evaluation."""
    context_relevance: float = Field(ge=0.0, le=1.0)
    groundedness: float = Field(ge=0.0, le=1.0)
    answer_relevance: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)
    passed: bool = True


class RAGEvaluator:
    """Programmatic evaluator for measuring RAG Triad scores."""

    @staticmethod
    def _words(text: str) -> set:
        return set(re.findall(r"\w{3,}", text.lower()))

    def evaluate_triad(
        self,
        query: str,
        retrieved_contexts: List[str],
        generated_answer: str,
        min_context_rel: float = 0.40,
        min_groundedness: float = 0.70,
        min_answer_rel: float = 0.50,
    ) -> RAGTriadMetrics:
        """Calculates Context Relevance, Groundedness, and Answer Relevance."""
        q_words = self._words(query)
        ans_words = self._words(generated_answer)
        joined_context = " ".join(retrieved_contexts)
        ctx_words = self._words(joined_context)

        stopwords = {
            "what", "are", "the", "for", "regarding", "policy", "guidelines",
            "how", "does", "can", "with", "from", "and", "under"
        }
        q_keywords = {w for w in q_words if w not in stopwords} or q_words

        # 1. Context Relevance: Proportion of query terms present in context
        if not q_keywords or not ctx_words:
            c_rel = 0.0
        else:
            overlap = q_keywords.intersection(ctx_words)
            c_rel = min(1.0, len(overlap) / len(q_keywords))

        # 2. Groundedness (Faithfulness): Proportion of answer content supported by context
        if not ans_words or not ctx_words:
            grounded = 0.0
        else:
            supported = ans_words.intersection(ctx_words)
            grounded = min(1.0, len(supported) / max(1, len(ans_words)))

        # 3. Answer Relevance: Answer addresses query words
        if not q_words or not ans_words:
            a_rel = 0.0
        else:
            overlap_qa = q_words.intersection(ans_words)
            a_rel = min(1.0, (len(overlap_qa) + 1) / max(1, len(q_words)))

        overall = (c_rel + grounded + a_rel) / 3.0
        passed = (
            c_rel >= min_context_rel
            and grounded >= min_groundedness
            and a_rel >= min_answer_rel
        )

        return RAGTriadMetrics(
            context_relevance=round(c_rel, 3),
            groundedness=round(grounded, 3),
            answer_relevance=round(a_rel, 3),
            overall_score=round(overall, 3),
            passed=passed,
        )
