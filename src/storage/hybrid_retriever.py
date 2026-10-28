from collections import defaultdict
from typing import Any, Dict, List
from src.core.llm_client import BaseLLMClient
from src.core.models import Chunk
from src.storage.bm25_store import BM25LexicalStore
from src.storage.vector_store import QdrantVectorStore


class HybridRetriever:
    """Combines Dense Vector Search and BM25 Lexical Search via Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        bm25_store: BM25LexicalStore,
        llm_client: BaseLLMClient,
        rrf_k: int = 60,
    ):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.llm_client = llm_client
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Executes parallel dense and sparse search, merging with RRF."""
        # 1. Dense Vector Search
        query_vector = self.llm_client.embed_texts([query])[0]
        dense_results = self.vector_store.search_dense(query_vector=query_vector, top_k=top_k * 2)

        # 2. BM25 Lexical Search
        bm25_results = self.bm25_store.search_lexical(query=query, top_k=top_k * 2)

        # 3. Reciprocal Rank Fusion
        rrf_scores = defaultdict(float)
        chunk_map = {}

        # Add Dense ranks
        for rank, res in enumerate(dense_results, start=1):
            c_id = res["chunk_id"]
            rrf_scores[c_id] += 1.0 / (self.rrf_k + rank)
            chunk_map[c_id] = res

        # Add BM25 ranks
        for rank, res in enumerate(bm25_results, start=1):
            c_id = res["chunk_id"]
            rrf_scores[c_id] += 1.0 / (self.rrf_k + rank)
            if c_id not in chunk_map:
                chunk_map[c_id] = res

        # Sort by unified RRF score descending
        sorted_chunk_ids = sorted(
            rrf_scores.keys(),
            key=lambda cid: rrf_scores[cid],
            reverse=True,
        )

        final_results = []
        for cid in sorted_chunk_ids[:top_k]:
            item = chunk_map[cid]
            item["rrf_score"] = round(rrf_scores[cid], 5)
            final_results.append(item)

        return final_results
