import pytest
from src.core.llm_client import MockLLMClient
from src.core.models import Document
from src.fast_loop.workflow import FastLoopRAGWorkflow
from src.ingestion.pipeline import IngestionPipeline
from src.storage.bm25_store import BM25LexicalStore
from src.storage.hybrid_retriever import HybridRetriever
from src.storage.reranker import Reranker
from src.storage.vector_store import QdrantVectorStore


@pytest.fixture
def rag_system():
    llm = MockLLMClient()
    v_store = QdrantVectorStore(collection_name="test_fast_loop", location=":memory:")
    b_store = BM25LexicalStore()

    ingestion = IngestionPipeline(
        vector_store=v_store,
        bm25_store=b_store,
        llm_client=llm,
    )

    doc = Document(
        doc_id="doc_security_1",
        title="Password Security Policy",
        version="1.0",
        content="Passwords must contain at least 14 characters including numbers and special symbols.",
    )
    ingestion.ingest_document(doc)

    retriever = HybridRetriever(
        vector_store=v_store,
        bm25_store=b_store,
        llm_client=llm,
    )
    reranker = Reranker(llm_client=llm)

    workflow = FastLoopRAGWorkflow(
        hybrid_retriever=retriever,
        reranker=reranker,
        llm_client=llm,
        max_retries=2,
    )

    return workflow, doc


def test_fast_loop_query_success(rag_system):
    workflow, doc = rag_system
    result = workflow.query("What are the password security requirements?")

    assert result["status"] == "completed"
    assert len(result["graded_chunks"]) >= 1
    assert result["generation"] != ""
    assert result["grounding_verdict"] is not None
    assert result["grounding_verdict"].is_faithful is True
    assert result["telemetry_signal"] is not None


def test_fast_loop_query_rewrite_on_low_confidence(rag_system):
    workflow, doc = rag_system
    # Out of domain query that fails initial retrieval
    result = workflow.query("Explain advanced orbital mechanics and propulsion")

    assert result["status"] == "completed"
    # Should attempt rewrite and retry
    assert result["retry_count"] >= 1
    assert result["fallback_used"] is True
