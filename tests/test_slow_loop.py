import pytest
from src.core.llm_client import MockLLMClient
from src.core.models import Document, FailureType, TelemetrySignal
from src.ingestion.pipeline import IngestionPipeline
from src.slow_loop.immune_loop import KnowledgeBaseImmuneLoop
from src.slow_loop.telemetry import TelemetryLogger
from src.storage.bm25_store import BM25LexicalStore
from src.storage.vector_store import QdrantVectorStore


def test_slow_loop_contradiction_resolution():
    llm = MockLLMClient()
    v_store = QdrantVectorStore(collection_name="test_slow_loop", location=":memory:")
    b_store = BM25LexicalStore()
    telemetry = TelemetryLogger()

    ingestion = IngestionPipeline(vector_store=v_store, bm25_store=b_store, llm_client=llm)

    # Ingest v1 and v2 docs with conflicting versions
    doc_v1 = Document(
        doc_id="doc_vpn_v1",
        title="VPN Access Policy",
        version="1.0",
        content="Use OpenVPN portal at vpn.corp with password.",
    )
    doc_v2 = Document(
        doc_id="doc_vpn_v2",
        title="VPN Access Policy",
        version="2.0",
        content="OpenVPN is decommissioned. Use Zero-Trust ZTNA gateway with FIDO2.",
    )

    chunks_v1 = ingestion.ingest_document(doc_v1)
    chunks_v2 = ingestion.ingest_document(doc_v2)
    all_chunks = chunks_v1 + chunks_v2

    # Log a degraded signal to trigger healing
    sig = TelemetrySignal(
        signal_id="sig_test_1",
        query="VPN portal connection",
        avg_relevance=0.40,
        hallucination_detected=False,
    )
    telemetry.log_signal(sig)

    immune = KnowledgeBaseImmuneLoop(
        vector_store=v_store,
        bm25_store=b_store,
        telemetry_logger=telemetry,
        llm_client=llm,
    )

    report = immune.run_healing_cycle(
        known_documents=[doc_v1, doc_v2],
        known_chunks=all_chunks,
    )

    assert report["degraded_signals_processed"] >= 1
    # Check that older v1 chunk was marked inactive
    for c in chunks_v1:
        assert c.is_active is False
        assert c.superseded_by == "doc_vpn_v2"


def test_slow_loop_knowledge_gap_alert():
    llm = MockLLMClient()
    v_store = QdrantVectorStore(collection_name="test_gap", location=":memory:")
    b_store = BM25LexicalStore()
    telemetry = TelemetryLogger()

    # Log a signal for a completely absent topic
    sig = TelemetrySignal(
        signal_id="sig_gap_1",
        query="Kubernetes cluster disaster recovery failover",
        avg_relevance=0.20,
    )
    telemetry.log_signal(sig)

    immune = KnowledgeBaseImmuneLoop(
        vector_store=v_store,
        bm25_store=b_store,
        telemetry_logger=telemetry,
        llm_client=llm,
    )

    report = immune.run_healing_cycle(known_documents=[], known_chunks=[])
    actions = [a["type"] for a in report["actions_taken"]]
    assert "knowledge_gap_alert" in actions
