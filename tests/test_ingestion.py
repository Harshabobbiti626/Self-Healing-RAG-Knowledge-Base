import pytest
from src.core.llm_client import MockLLMClient
from src.core.models import ContentType, Document
from src.ingestion.contextual_chunker import ContextualChunker
from src.ingestion.parser import DocumentParser
from src.ingestion.pipeline import IngestionPipeline
from src.storage.bm25_store import BM25LexicalStore
from src.storage.vector_store import QdrantVectorStore


def test_document_parser():
    raw_text = "# Corporate Travel Policy\n\nHotel reimbursement is capped at $250."
    doc = DocumentParser.parse_text(
        content=raw_text,
        title="Corporate Travel Policy",
        content_type=ContentType.MARKDOWN,
    )
    assert doc.title == "Corporate Travel Policy"
    assert "reimbursement" in doc.content
    assert doc.content_type == ContentType.MARKDOWN


def test_contextual_chunker():
    llm = MockLLMClient()
    chunker = ContextualChunker(llm_client=llm, target_chunk_chars=100)

    doc = Document(
        doc_id="doc_test_1",
        title="Engineering Onboarding Guide",
        version="2.1",
        content=(
            "Paragraph 1: Welcome to the engineering team.\n\n"
            "Paragraph 2: Setting up your development environment with Git and Docker.\n\n"
            "Paragraph 3: Submitting pull requests and following CI/CD workflows."
        ),
        metadata={"category": "Engineering"},
    )

    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2
    for c in chunks:
        assert "Engineering Onboarding Guide" in c.contextual_prefix
        assert c.full_contextual_text.startswith("[Context:")


def test_ingestion_pipeline_end_to_end():
    llm = MockLLMClient()
    v_store = QdrantVectorStore(collection_name="test_ingest", location=":memory:")
    b_store = BM25LexicalStore()

    pipeline = IngestionPipeline(
        vector_store=v_store,
        bm25_store=b_store,
        llm_client=llm,
    )

    doc = Document(
        doc_id="doc_hr_1",
        title="Parental Leave Policy",
        version="1.0",
        content="Employees receive 16 weeks of fully paid parental leave starting on day one.",
    )

    chunks = pipeline.ingest_document(doc)
    assert len(chunks) >= 1
    assert chunks[0].embedding is not None

    # Test BM25 match
    lexical_hits = b_store.search_lexical("parental leave", top_k=5)
    assert len(lexical_hits) >= 1
    assert lexical_hits[0]["doc_id"] == "doc_hr_1"
