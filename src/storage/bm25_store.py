import re
from typing import Any, Dict, List
from rank_bm25 import BM25Okapi
from src.core.models import Chunk


class BM25LexicalStore:
    """Lexical token matching index using BM25Okapi for hybrid retrieval."""

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: BM25Okapi | None = None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace and punctuation word tokenizer."""
        return re.findall(r"\w+", text.lower())

    def index_chunks(self, chunks: List[Chunk]) -> None:
        """Build or update BM25 index over contextualized chunks."""
        active_chunks = [c for c in chunks if c.is_active]
        self.chunks = active_chunks
        self.tokenized_corpus = [
            self._tokenize(c.full_contextual_text) for c in self.chunks
        ]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search_lexical(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Perform BM25 lexical keyword search."""
        if not self.bm25 or not self.chunks:
            return []

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        scored_pairs = sorted(
            zip(self.chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for chunk, score in scored_pairs:
            chunk_tokens = set(self._tokenize(chunk.full_contextual_text))
            overlap = set(tokenized_query).intersection(chunk_tokens)
            if not overlap:
                continue
            
            # Normalize score: BM25Okapi produces negative scores when N=1 or term appears in >50% of docs
            normalized_score = max(score, 0.0) if score > 0 else (0.5 * len(overlap))
            results.append({
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "content": chunk.content,
                "contextual_prefix": chunk.contextual_prefix,
                "score": float(normalized_score),
                "payload": chunk.metadata,
            })
            if len(results) >= top_k:
                break

        return results
