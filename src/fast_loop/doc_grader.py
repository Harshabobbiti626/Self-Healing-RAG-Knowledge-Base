import logging
from typing import Any, Dict, List, Tuple
from config.settings import settings
from src.core.llm_client import BaseLLMClient
from src.core.models import Chunk, DocumentRelevance, RelevanceStatus

logger = logging.getLogger(__name__)


class DocumentGrader:
    """Evaluates factual relevance of retrieved chunks to filter out noise."""

    def __init__(self, llm_client: BaseLLMClient, threshold: float = 0.60):
        self.llm_client = llm_client
        self.threshold = threshold

    def grade_candidate_chunks(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[DocumentRelevance], bool]:
        """Grades all candidates, returning filtered relevant chunks, all grading records, and fallback flag."""
        relevant_chunks: List[Dict[str, Any]] = []
        gradings: List[DocumentRelevance] = []

        for cand in candidates:
            c_id = cand["chunk_id"]
            content = cand["content"]
            grading = self.llm_client.grade_relevance(
                query=query,
                chunk_content=content,
                chunk_id=c_id,
            )
            gradings.append(grading)

            if grading.score >= self.threshold or grading.status == RelevanceStatus.RELEVANT:
                cand["grade_score"] = grading.score
                cand["grade_status"] = grading.status.value
                relevant_chunks.append(cand)

        avg_score = (
            sum(g.score for g in gradings) / max(1, len(gradings))
            if gradings
            else 0.0
        )
        needs_fallback = len(relevant_chunks) == 0 or avg_score < settings.LOW_CONFIDENCE_LOG_THRESHOLD

        logger.info(
            "Graded %d candidate chunks: %d relevant (avg score: %.2f, fallback needed: %s)",
            len(candidates),
            len(relevant_chunks),
            avg_score,
            needs_fallback,
        )

        return relevant_chunks, gradings, needs_fallback
