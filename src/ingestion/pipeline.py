import logging
from pathlib import Path
from typing import List, Union
from src.core.llm_client import BaseLLMClient
from src.core.models import Chunk, Document
from src.ingestion.contextual_chunker import ContextualChunker
from src.ingestion.parser import DocumentParser
from src.storage.bm25_store import BM25LexicalStore
from src.storage.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """End-to-end ingestion pipeline executing parsing, contextual chunking, and dual indexing."""

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        bm25_store: BM25LexicalStore,
        llm_client: BaseLLMClient,
    ):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.llm_client = llm_client
        self.chunker = ContextualChunker(llm_client=llm_client)
        self.all_indexed_chunks: List[Chunk] = []

    def ingest_document(self, doc: Document) -> List[Chunk]:
        """Ingest a single Document through contextual chunking, embedding, and dual indexing."""
        # 1. Contextual Chunking
        chunks = self.chunker.chunk_document(doc)
        if not chunks:
            return []

        # 2. Dense Vector Embeddings (embed full contextual text: [Context: ...] \n content)
        texts_to_embed = [c.full_contextual_text for c in chunks]
        embeddings = self.llm_client.embed_texts(texts_to_embed)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        # 3. Store in Qdrant
        self.vector_store.upsert_chunks(chunks)

        # 4. Add to BM25 Lexical Store
        self.all_indexed_chunks.extend(chunks)
        self.bm25_store.index_chunks(self.all_indexed_chunks)

        logger.info("Ingested doc '%s' into %d contextual chunks.", doc.title, len(chunks))
        return chunks

    def ingest_files(self, file_paths: List[Union[str, Path]]) -> int:
        """Parse and ingest multiple files from disk."""
        total_chunks = 0
        for fp in file_paths:
            doc = DocumentParser.parse_file(fp)
            chunks = self.ingest_document(doc)
            total_chunks += len(chunks)
        return total_chunks
