import logging
import re
from typing import Dict, List, Optional
from src.core.llm_client import BaseLLMClient
from src.core.models import Document, FailureType, TelemetrySignal

logger = logging.getLogger(__name__)


class RootCauseDiagnosticAgent:
    """Diagnoses root cause of RAG failures using telemetry signals and KB inspection."""

    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client

    def diagnose_signal(
        self,
        signal: TelemetrySignal,
        known_documents: List[Document],
        retrieved_chunk_contents: Optional[List[str]] = None,
    ) -> FailureType:
        """Classifies the root cause behind a degraded query signal."""
        query = signal.query.lower()
        query_words = set(re.findall(r"\w{4,}", query))

        # 1. Check for Knowledge Gap: Does any document cover the query keywords?
        all_doc_content = " ".join([d.content.lower() for d in known_documents])
        doc_overlap = query_words.intersection(set(re.findall(r"\w{4,}", all_doc_content)))

        if len(doc_overlap) == 0 and len(query_words) > 0:
            logger.info("Signal '%s' diagnosed as KNOWLEDGE_GAP", signal.signal_id)
            return FailureType.KNOWLEDGE_GAP

        # 2. Check for Temporal Contradiction: Are there multiple versions of docs on this subject?
        matching_titles = [d for d in known_documents if any(w in d.title.lower() for w in query_words)]
        distinct_versions = {d.version for d in matching_titles}
        if len(distinct_versions) > 1:
            logger.info("Signal '%s' diagnosed as TEMPORAL_CONTRADICTION", signal.signal_id)
            return FailureType.TEMPORAL_CONTRADICTION

        # 3. Check for Semantic Bleed: Were chunks retrieved, but had ungrounded answers or cut context?
        if signal.hallucination_detected or signal.retry_count > 0:
            logger.info("Signal '%s' diagnosed as SEMANTIC_BLEED", signal.signal_id)
            return FailureType.SEMANTIC_BLEED

        # 4. Check for Vocabulary Blindspot: Query contains informal/acronym terms
        if signal.avg_relevance < 0.45 and len(doc_overlap) > 0:
            logger.info("Signal '%s' diagnosed as VOCABULARY_BLINDSPOT", signal.signal_id)
            return FailureType.VOCABULARY_BLINDSPOT

        return FailureType.LOW_CONFIDENCE
