import logging
import time
from typing import Any, Dict, List
import uuid

from langgraph.graph import END, StateGraph
from config.settings import settings
from src.core.llm_client import BaseLLMClient
from src.core.models import Chunk, TelemetrySignal
from src.core.state import RAGState
from src.fast_loop.doc_grader import DocumentGrader
from src.fast_loop.generator import GroundedGenerator
from src.fast_loop.grounding_guard import GroundingGuard
from src.fast_loop.query_rewriter import QueryRewriter
from src.storage.hybrid_retriever import HybridRetriever
from src.storage.reranker import Reranker

logger = logging.getLogger(__name__)


class FastLoopRAGWorkflow:
    """Cyclic LangGraph state machine orchestrating runtime self-healing RAG."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker,
        llm_client: BaseLLMClient,
        max_retries: int = 2,
    ):
        self.retriever = hybrid_retriever
        self.reranker = reranker
        self.llm_client = llm_client
        self.max_retries = max_retries

        self.rewriter = QueryRewriter(llm_client=llm_client)
        self.grader = DocumentGrader(
            llm_client=llm_client, threshold=settings.RELEVANCE_THRESHOLD
        )
        self.generator = GroundedGenerator(llm_client=llm_client)
        self.guard = GroundingGuard(llm_client=llm_client)

        self.app = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(RAGState)

        # Define graph nodes
        workflow.add_node("retrieve", self._node_retrieve)
        workflow.add_node("grade_documents", self._node_grade_documents)
        workflow.add_node("transform_query", self._node_transform_query)
        workflow.add_node("generate", self._node_generate)
        workflow.add_node("verify_grounding", self._node_verify_grounding)
        workflow.add_node("self_critique", self._node_self_critique)

        # Set entry point
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")

        # Conditional edge after grading
        def decide_after_grading(state: RAGState) -> str:
            retry_count = state.get("retry_count", 0)
            graded = state.get("graded_chunks", [])
            needs_fallback = state.get("needs_fallback", False)

            if (len(graded) == 0 or needs_fallback) and retry_count < self.max_retries:
                logger.info("Triggering query rewrite loop (retry %d/%d)", retry_count + 1, self.max_retries)
                return "transform_query"
            return "generate"

        workflow.add_conditional_edges(
            "grade_documents",
            decide_after_grading,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )

        workflow.add_edge("transform_query", "retrieve")
        workflow.add_edge("generate", "verify_grounding")

        # Conditional edge after grounding check
        def decide_after_verification(state: RAGState) -> str:
            retry_count = state.get("retry_count", 0)
            verdict = state.get("grounding_verdict")

            if verdict and not verdict.is_faithful and retry_count < self.max_retries:
                logger.warning("Hallucination flagged. Triggering self-critique loop.")
                return "self_critique"
            return END

        workflow.add_conditional_edges(
            "verify_grounding",
            decide_after_verification,
            {
                "self_critique": "self_critique",
                END: END,
            },
        )

        workflow.add_edge("self_critique", "generate")

        return workflow.compile()

    # --- Node Implementations ---

    def _node_retrieve(self, state: RAGState) -> Dict[str, Any]:
        target_query = state.get("rewritten_query") or state["question"]
        logger.info("Executing retrieval for query: '%s'", target_query)
        raw_candidates = self.retriever.retrieve(
            query=target_query, top_k=settings.TOP_K_RETRIEVAL
        )
        # Apply precision reranker
        reranked = self.reranker.rerank(
            query=target_query, candidates=raw_candidates, top_n=settings.TOP_K_RERANK
        )
        return {"candidate_chunks": reranked}

    def _node_grade_documents(self, state: RAGState) -> Dict[str, Any]:
        target_query = state.get("rewritten_query") or state["question"]
        candidates = state.get("candidate_chunks", [])
        relevant_chunks, gradings, needs_fallback = self.grader.grade_candidate_chunks(
            query=target_query, candidates=candidates
        )
        return {
            "graded_chunks": relevant_chunks,
            "grader_scores": [g.score for g in gradings],
            "needs_fallback": needs_fallback,
        }

    def _node_transform_query(self, state: RAGState) -> Dict[str, Any]:
        current_query = state.get("rewritten_query") or state["question"]
        retry = state.get("retry_count", 0) + 1
        rewritten = self.rewriter.rewrite_for_retrieval(
            query=current_query, failure_reason="Insufficient relevance in previous retrieval."
        )
        return {
            "rewritten_query": rewritten,
            "retry_count": retry,
            "fallback_used": True,
        }

    def _node_generate(self, state: RAGState) -> Dict[str, Any]:
        query = state["question"]
        context = state.get("graded_chunks", [])
        critique = state.get("critique_prompt")
        answer = self.generator.generate_response(
            query=query, context_chunks=context, critique_feedback=critique
        )
        citations = [c.get("chunk_id", "") for c in context]
        return {
            "generation": answer,
            "citations": citations,
            "critique_prompt": None,  # Reset critique once consumed
        }

    def _node_verify_grounding(self, state: RAGState) -> Dict[str, Any]:
        query = state["question"]
        answer = state.get("generation", "")
        context = state.get("graded_chunks", [])
        verdict = self.guard.audit_generation(
            query=query, answer=answer, context_chunks=context
        )
        return {"grounding_verdict": verdict}

    def _node_self_critique(self, state: RAGState) -> Dict[str, Any]:
        verdict = state.get("grounding_verdict")
        retry = state.get("retry_count", 0) + 1
        critique_msg = (
            verdict.critique_feedback
            if verdict and verdict.critique_feedback
            else "Eliminate ungrounded statements and stick strictly to retrieved facts."
        )
        return {
            "critique_prompt": critique_msg,
            "retry_count": retry,
        }

    def query(self, question: str) -> Dict[str, Any]:
        """User-facing execution entrypoint."""
        start_time = time.perf_counter()
        initial_state: RAGState = {
            "question": question,
            "rewritten_query": None,
            "sub_queries": [],
            "candidate_chunks": [],
            "graded_chunks": [],
            "grader_scores": [],
            "needs_fallback": False,
            "generation": "",
            "citations": [],
            "grounding_verdict": None,
            "retry_count": 0,
            "critique_prompt": None,
            "fallback_used": False,
            "telemetry_signal": None,
            "status": "in_progress",
        }

        final_state = self.app.invoke(initial_state)
        latency = (time.perf_counter() - start_time) * 1000

        # Construct and attach TelemetrySignal
        scores = final_state.get("grader_scores", [])
        avg_score = sum(scores) / max(1, len(scores)) if scores else 0.0
        verdict = final_state.get("grounding_verdict")
        grounded = verdict.is_faithful if verdict else True

        signal = TelemetrySignal(
            signal_id=f"sig_{uuid.uuid4().hex[:8]}",
            query=question,
            rewritten_query=final_state.get("rewritten_query"),
            retrieved_chunk_ids=[c.get("chunk_id", "") for c in final_state.get("graded_chunks", [])],
            grader_scores=scores,
            avg_relevance=avg_score,
            grounding_passed=grounded,
            hallucination_detected=not grounded,
            retry_count=final_state.get("retry_count", 0),
            fallback_used=final_state.get("fallback_used", False),
            latency_ms=round(latency, 2),
        )
        final_state["telemetry_signal"] = signal
        final_state["status"] = "completed"

        return final_state
