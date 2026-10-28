import logging
from typing import List, Tuple
import uuid

from src.core.llm_client import BaseLLMClient
from src.core.models import Chunk, Document
from src.storage.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


class ContradictionAuditor:
    """Audits knowledge base for cross-document conflicts, version skew, and semantic contradictions."""

    def __init__(self, vector_store: QdrantVectorStore, llm_client: BaseLLMClient):
        self.vector_store = vector_store
        self.llm_client = llm_client

    def resolve_version_conflicts(
        self,
        documents: List[Document],
        all_chunks: List[Chunk],
    ) -> List[Chunk]:
        """Scans for multi-version documents and deprecates older chunks, synthesizing golden chunks."""
        docs_by_title = {}
        for d in documents:
            title_key = d.title.strip().lower()
            docs_by_title.setdefault(title_key, []).append(d)

        new_golden_chunks: List[Chunk] = []

        for title_key, doc_versions in docs_by_title.items():
            if len(doc_versions) <= 1:
                continue

            # Sort by version or created_at descending (newest first)
            sorted_versions = sorted(
                doc_versions,
                key=lambda d: d.created_at,
                reverse=True,
            )
            canonical_doc = sorted_versions[0]
            stale_docs = sorted_versions[1:]

            stale_doc_ids = {d.doc_id for d in stale_docs}
            logger.info(
                "Detected version skew for '%s': canonical=%s v%s, stale docs=%s",
                title_key,
                canonical_doc.doc_id,
                canonical_doc.version,
                list(stale_doc_ids),
            )

            # Deactivate chunks belonging to stale document versions
            for chunk in all_chunks:
                if chunk.doc_id in stale_doc_ids and chunk.is_active:
                    chunk.is_active = False
                    chunk.superseded_by = canonical_doc.doc_id
                    self.vector_store.deactivate_chunk(
                        chunk_id=chunk.chunk_id,
                        superseded_by=canonical_doc.doc_id,
                    )

            # Synthesize a Golden Reconciliation Chunk
            golden_text = (
                f"OFFICIAL CANONICAL POLICY for '{canonical_doc.title}' (Version {canonical_doc.version}).\n"
                f"Notice: All prior versions ({[d.version for d in stale_docs]}) are formally superseded.\n"
                f"Summary of current active rules:\n{canonical_doc.content[:600]}"
            )
            golden_chunk = Chunk(
                chunk_id=f"golden_{uuid.uuid4().hex[:8]}",
                doc_id=canonical_doc.doc_id,
                content=golden_text,
                contextual_prefix=f"Reconciliation Golden Rule: {canonical_doc.title}",
                chunk_index=999,
                is_active=True,
                is_golden=True,
                metadata={
                    "is_golden": True,
                    "superseded_versions": [d.version for d in stale_docs],
                    "doc_title": canonical_doc.title,
                },
            )
            # Embed and upsert golden chunk
            emb = self.llm_client.embed_texts([golden_chunk.full_contextual_text])[0]
            golden_chunk.embedding = emb
            self.vector_store.upsert_chunks([golden_chunk])
            new_golden_chunks.append(golden_chunk)

        return new_golden_chunks
