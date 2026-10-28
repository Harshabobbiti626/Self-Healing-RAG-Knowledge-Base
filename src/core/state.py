from typing import List, Optional
from typing_extensions import TypedDict
from src.core.models import Chunk, GroundingVerdict, TelemetrySignal


class RAGState(TypedDict, total=False):
    """LangGraph cyclic state schema for Fast-Loop Self-Healing RAG."""
    # User Inputs
    question: str
    rewritten_query: Optional[str]
    sub_queries: List[str]

    # Retrieval State
    candidate_chunks: List[Chunk]
    graded_chunks: List[Chunk]
    grader_scores: List[float]
    needs_fallback: bool

    # Generation State
    generation: str
    citations: List[str]
    grounding_verdict: Optional[GroundingVerdict]

    # Iteration & Safety
    retry_count: int
    critique_prompt: Optional[str]
    fallback_used: bool

    # Observability
    telemetry_signal: Optional[TelemetrySignal]
    status: str
