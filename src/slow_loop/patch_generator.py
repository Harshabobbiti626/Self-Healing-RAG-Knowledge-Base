import logging
from typing import Dict, List, Optional
import uuid

from src.core.llm_client import BaseLLMClient
from src.core.models import Chunk, Document, FailureType, HealingAction, TelemetrySignal
from src.ingestion.contextual_chunker import ContextualChunker
from src.storage.bm25_store import BM25LexicalStore
from src.storage.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


class PatchGenerator:
    """Generates and applies autonomous knowledge base patches based on diagnosed failures."""

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        bm25_store: BM25LexicalStore,
        llm_client: BaseLLMClient,
    ):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.llm_client = llm_client

    def patch_vocabulary_blindspot(
        self,
        signal: TelemetrySignal,
        target_doc: Document,
    ) -> Chunk:
        """Injects a synthetic Q&A 'Reflection Artifact' chunk bridging user vocabulary to doc facts."""
        qa_content = (
            f"Frequently Asked Query: {signal.query}\n"
            f"Canonical Verified Answer from '{target_doc.title}':\n"
            f"{target_doc.content[:500]}"
        )
        reflection_chunk = Chunk(
            chunk_id=f"reflect_{uuid.uuid4().hex[:8]}",
            doc_id=target_doc.doc_id,
            content=qa_content,
            contextual_prefix=f"Reflection Artifact for: {target_doc.title}",
            is_active=True,
            is_golden=True,
            metadata={
                "is_reflection_artifact": True,
                "origin_query": signal.query,
                "doc_title": target_doc.title,
            },
        )
        emb = self.llm_client.embed_texts([reflection_chunk.full_contextual_text])[0]
        reflection_chunk.embedding = emb

        # Upsert into vector and BM25 stores
        self.vector_store.upsert_chunks([reflection_chunk])
        self.bm25_store.index_chunks([reflection_chunk])

        logger.info(
            "Injected reflection artifact chunk '%s' for query '%s'",
            reflection_chunk.chunk_id,
            signal.query,
        )
        return reflection_chunk

    def patch_semantic_bleed(
        self,
        target_doc: Document,
        existing_chunks: List[Chunk],
    ) -> List[Chunk]:
        """Re-chunks a document using larger chunk windows and refreshed contextual prefixes."""
        # Deactivate old chunks
        for c in existing_chunks:
            if c.doc_id == target_doc.doc_id and c.is_active:
                c.is_active = False
                self.vector_store.deactivate_chunk(c.chunk_id)

        # Re-chunk with expanded target window
        chunker = ContextualChunker(
            llm_client=self.llm_client,
            target_chunk_chars=800,  # Expanded window to prevent sentence/table splitting
            chunk_overlap_chars=120,
        )
        new_chunks = chunker.chunk_document(target_doc)
        embeddings = self.llm_client.embed_texts([c.full_contextual_text for c in new_chunks])
        for c, emb in zip(new_chunks, embeddings):
            c.embedding = emb

        self.vector_store.upsert_chunks(new_chunks)
        logger.info(
            "Re-chunked doc '%s' into %d expanded chunks (healed semantic bleed).",
            target_doc.title,
            len(new_chunks),
        )
        return new_chunks

    def create_knowledge_gap_alert(self, signal: TelemetrySignal) -> HealingAction:
        """Constructs an actionable knowledge gap ticket for content owners."""
        action = HealingAction(
            patch_id=f"gap_{uuid.uuid4().hex[:8]}",
            failure_type=FailureType.KNOWLEDGE_GAP,
            action_type="alert_knowledge_gap",
            details={
                "missing_topic_query": signal.query,
                "signal_id": signal.signal_id,
                "avg_retrieval_confidence": signal.avg_relevance,
                "recommended_action": "Author new knowledge documentation covering this topic.",
            },
            applied=True,
        )
        logger.warning(
            "Created Knowledge Gap Alert [%s] for query: '%s'",
            action.patch_id,
            signal.query,
        )
        return action
