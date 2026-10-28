from typing import Any, Dict, List
from src.core.llm_client import BaseLLMClient
from src.core.models import Chunk, RelevanceStatus


class Reranker:
    """Precision reranking stage taking top RRF candidates to select top-N precision chunks."""

    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """Reranks candidate chunks by precision semantic alignment with query."""
        if not candidates:
            return []

        scored_candidates = []
        for cand in candidates:
            c_id = cand["chunk_id"]
            content = cand["content"]
            grading = self.llm_client.grade_relevance(
                query=query,
                chunk_content=content,
                chunk_id=c_id,
            )
            cand["rerank_score"] = grading.score
            cand["relevance_status"] = grading.status.value
            cand["relevance_reasoning"] = grading.reasoning
            scored_candidates.append(cand)

        # Sort descending by rerank score
        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_candidates[:top_n]
