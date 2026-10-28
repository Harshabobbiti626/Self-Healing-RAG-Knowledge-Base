import logging
from typing import Any, Dict, List
from src.core.llm_client import BaseLLMClient
from src.core.models import GroundingVerdict

logger = logging.getLogger(__name__)


class GroundingGuard:
    """Post-generation hallucination and claim verification guard."""

    def __init__(self, llm_client: BaseLLMClient, max_hallucination_score: float = 0.35):
        self.llm_client = llm_client
        self.max_hallucination_score = max_hallucination_score

    def audit_generation(
        self,
        query: str,
        answer: str,
        context_chunks: List[Dict[str, Any]],
    ) -> GroundingVerdict:
        """Verifies that the generated response is strictly supported by context chunks."""
        verdict = self.llm_client.verify_grounding(
            query=query,
            answer=answer,
            context_chunks=context_chunks,
        )

        # Flag if hallucination exceeds threshold
        if verdict.hallucination_score > self.max_hallucination_score:
            verdict.is_faithful = False
            if not verdict.critique_feedback:
                verdict.critique_feedback = (
                    f"Draft answer contains ungrounded claims with hallucination score "
                    f"{verdict.hallucination_score:.2f} > {self.max_hallucination_score:.2f}."
                )

        logger.info(
            "Grounding audit: faithful=%s, hallucination_score=%.2f",
            verdict.is_faithful,
            verdict.hallucination_score,
        )
        return verdict
