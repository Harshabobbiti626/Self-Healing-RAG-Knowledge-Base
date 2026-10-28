import logging
from typing import Any, Dict, List, Optional
from src.core.llm_client import BaseLLMClient
from src.core.models import Chunk, Document, FailureType, HealingAction, TelemetrySignal
from src.slow_loop.contradiction_auditor import ContradictionAuditor
from src.slow_loop.diagnostic_agent import RootCauseDiagnosticAgent
from src.slow_loop.patch_generator import PatchGenerator
from src.slow_loop.telemetry import TelemetryLogger
from src.storage.bm25_store import BM25LexicalStore
from src.storage.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


class KnowledgeBaseImmuneLoop:
    """Orchestrates asynchronous knowledge base self-healing and continuous maintenance."""

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        bm25_store: BM25LexicalStore,
        telemetry_logger: TelemetryLogger,
        llm_client: BaseLLMClient,
    ):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.telemetry = telemetry_logger
        self.llm_client = llm_client

        self.diagnostic_agent = RootCauseDiagnosticAgent(llm_client=llm_client)
        self.auditor = ContradictionAuditor(
            vector_store=vector_store, llm_client=llm_client
        )
        self.patch_gen = PatchGenerator(
            vector_store=vector_store,
            bm25_store=bm25_store,
            llm_client=llm_client,
        )

        self.applied_actions: List[HealingAction] = []

    def _find_target_doc_for_signal(
        self,
        signal: TelemetrySignal,
        known_documents: List[Document],
        known_chunks: List[Chunk],
    ) -> Optional[Document]:
        """Resolves the precise document that generated the degraded signal."""
        if signal.retrieved_chunk_ids:
            target_chunk_id = signal.retrieved_chunk_ids[0]
            for c in known_chunks:
                if c.chunk_id == target_chunk_id:
                    for d in known_documents:
                        if d.doc_id == c.doc_id:
                            return d

        # Fallback: Match by highest keyword overlap
        query_words = set(signal.query.lower().split())
        best_doc = None
        max_overlap = -1
        for d in known_documents:
            doc_words = set((d.title + " " + d.content).lower().split())
            overlap = len(query_words.intersection(doc_words))
            if overlap > max_overlap:
                max_overlap = overlap
                best_doc = d
        return best_doc or (known_documents[0] if known_documents else None)

    def run_healing_cycle(
        self,
        known_documents: List[Document],
        known_chunks: List[Chunk],
    ) -> Dict[str, Any]:
        """Executes a full diagnostic and remediation cycle over degraded telemetry signals."""
        degraded_signals = self.telemetry.get_degraded_signals()
        logger.info("Starting immune cycle with %d degraded signals.", len(degraded_signals))

        actions_taken = []
        new_chunks = []

        # 1. Resolve any multi-version contradictions across known docs
        golden_chunks = self.auditor.resolve_version_conflicts(
            documents=known_documents,
            all_chunks=known_chunks,
        )
        new_chunks.extend(golden_chunks)
        if golden_chunks:
            actions_taken.append({
                "type": "version_contradiction_resolved",
                "count": len(golden_chunks),
            })

        # 2. Iterate through degraded signals and apply targeted patches
        for signal in degraded_signals:
            diag = self.diagnostic_agent.diagnose_signal(
                signal=signal, known_documents=known_documents
            )

            if diag == FailureType.KNOWLEDGE_GAP:
                action = self.patch_gen.create_knowledge_gap_alert(signal)
                self.applied_actions.append(action)
                actions_taken.append({
                    "type": "knowledge_gap_alert",
                    "query": signal.query,
                })

            elif diag == FailureType.VOCABULARY_BLINDSPOT:
                target_doc = self._find_target_doc_for_signal(
                    signal=signal, known_documents=known_documents, known_chunks=known_chunks
                )
                if target_doc:
                    reflection_chunk = self.patch_gen.patch_vocabulary_blindspot(
                        signal=signal, target_doc=target_doc
                    )
                    new_chunks.append(reflection_chunk)
                    actions_taken.append({
                        "type": "reflection_qa_injected",
                        "query": signal.query,
                        "chunk_id": reflection_chunk.chunk_id,
                    })

            elif diag == FailureType.SEMANTIC_BLEED:
                target_doc = self._find_target_doc_for_signal(
                    signal=signal, known_documents=known_documents, known_chunks=known_chunks
                )
                if target_doc:
                    re_chunks = self.patch_gen.patch_semantic_bleed(
                        target_doc=target_doc, existing_chunks=known_chunks
                    )
                    new_chunks.extend(re_chunks)
                    actions_taken.append({
                        "type": "semantic_bleed_rechunked",
                        "doc_title": target_doc.title,
                        "new_chunk_count": len(re_chunks),
                    })

        # Update BM25 index with any newly synthesized golden/reflection chunks
        if new_chunks:
            self.bm25_store.index_chunks(known_chunks + new_chunks)

        report = {
            "degraded_signals_processed": len(degraded_signals),
            "actions_taken": actions_taken,
            "new_chunks_indexed": len(new_chunks),
        }
        logger.info("Immune cycle finished: %s", report)
        return report
