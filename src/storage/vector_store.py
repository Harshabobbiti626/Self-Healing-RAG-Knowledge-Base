import logging
import uuid
from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from src.core.models import Chunk

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """Wrapper around Qdrant for storing and searching chunk vector embeddings."""

    def __init__(
        self,
        collection_name: str = "self_healing_knowledge_base",
        vector_size: int = 1536,
        location: str = ":memory:",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size
        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
            logger.info("Connected to remote Qdrant Cloud cluster: %s", url)
        else:
            self.client = QdrantClient(location=location)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create or validate the vector collection to prevent dimension mismatches."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in collections:
            try:
                info = self.client.get_collection(self.collection_name)
                existing_size = info.config.params.vectors.size
                if existing_size != self.vector_size:
                    logger.warning(
                        "Recreating collection '%s': dimension changed from %d to %d.",
                        self.collection_name, existing_size, self.vector_size
                    )
                    self.client.delete_collection(self.collection_name)
                    collections.remove(self.collection_name)
            except Exception:
                pass

        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.vector_size,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection: %s (dim: %d)", self.collection_name, self.vector_size)

        # Ensure payload index on is_active for Qdrant Cloud filtering
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="is_active",
                field_schema=qmodels.PayloadSchemaType.BOOL,
            )
        except Exception:
            pass

    def upsert_chunks(self, chunks: List[Chunk]) -> None:
        """Insert or update chunks with dense vector embeddings."""
        points = []
        for idx, chunk in enumerate(chunks):
            if not chunk.embedding:
                continue
            payload = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "content": chunk.content,
                "contextual_prefix": chunk.contextual_prefix,
                "full_text": chunk.full_contextual_text,
                "is_active": chunk.is_active,
                "is_golden": chunk.is_golden,
                "superseded_by": chunk.superseded_by,
                "metadata": chunk.metadata,
            }
            # Use deterministic UUID5 for Qdrant point id to guarantee consistency across process restarts
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(chunk.chunk_id)))
            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector=chunk.embedding,
                    payload=payload,
                )
            )

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

    def search_dense(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter_active_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """Search vector database for top_k nearest neighbors by cosine similarity."""
        query_filter = None
        if filter_active_only:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="is_active",
                        match=qmodels.MatchValue(value=True),
                    )
                ]
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        formatted_results = []
        for scored_point in results.points:
            formatted_results.append({
                "chunk_id": scored_point.payload.get("chunk_id"),
                "doc_id": scored_point.payload.get("doc_id"),
                "content": scored_point.payload.get("content"),
                "contextual_prefix": scored_point.payload.get("contextual_prefix"),
                "score": scored_point.score,
                "payload": scored_point.payload,
            })

        return formatted_results

    def deactivate_chunk(self, chunk_id: str, superseded_by: Optional[str] = None) -> None:
        """Deactivate a chunk during slow-loop remediation (mark as superseded or deprecated)."""
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(chunk_id)))
        self.client.set_payload(
            collection_name=self.collection_name,
            payload={
                "is_active": False,
                "superseded_by": superseded_by,
            },
            points=[point_id],
        )
