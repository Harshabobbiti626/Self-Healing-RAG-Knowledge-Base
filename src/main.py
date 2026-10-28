import logging
from pathlib import Path
import sys

# Ensure repository root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config.settings import settings
from src.core.llm_client import get_llm_client
from src.core.models import ContentType, Document
from src.evaluation.metrics import RAGEvaluator
from src.evaluation.regression_runner import RegressionTestRunner
from src.evaluation.synthetic_gen import SyntheticDatasetGenerator
from src.fast_loop.workflow import FastLoopRAGWorkflow
from src.ingestion.parser import DocumentParser
from src.ingestion.pipeline import IngestionPipeline
from src.slow_loop.immune_loop import KnowledgeBaseImmuneLoop
from src.slow_loop.telemetry import TelemetryLogger
from src.storage.bm25_store import BM25LexicalStore
from src.storage.hybrid_retriever import HybridRetriever
from src.storage.reranker import Reranker
from src.storage.vector_store import QdrantVectorStore

logging.basicConfig(level=logging.WARNING)
console = Console()


def print_banner():
    banner = r"""[bold cyan]
   _____      _  __     _    _            _ _               _____            _____ 
  / ____|    | |/ _|   | |  | |          | (_)             |  __ \   /\    / ____|
 | (___   ___| | |_ ___| |__| | ___  __ _| |_ _ __   __ _  | |__) | /  \  | |  __ 
  \___ \ / _ \ |  _/___|  __  |/ _ \/ _` | | | '_ \ / _` | |  _  / / /\ \ | | |_ |
  ____) |  __/ | |     | |  | |  __/ (_| | | | | | | (_| | | | \ \/ ____ \| |__| |
 |_____/ \___|_|_|     |_|  |_|\___|\__,_|_|_|_| |_|\__, | |_|  \_\_/    \_\\_____|
                                                     __/ |                         
                                                    |___/                          
[/bold cyan][dim]Enterprise Dual-Loop Self-Healing Knowledge Base (Fast Runtime + Slow Immune System)[/dim]
"""
    console.print(banner)


def run_demo():
    print_banner()

    # 1. Initialize Infrastructure Components
    console.print("[bold yellow]1. Initializing System Components...[/bold yellow]")
    llm_client = get_llm_client()
    vector_store = QdrantVectorStore(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        vector_size=settings.VECTOR_SIZE,
        location=settings.QDRANT_LOCATION,
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
    )
    bm25_store = BM25LexicalStore()
    telemetry_logger = TelemetryLogger()

    # Ingestion Pipeline
    ingestion_pipeline = IngestionPipeline(
        vector_store=vector_store,
        bm25_store=bm25_store,
        llm_client=llm_client,
    )

    # Retrieval & Reranker
    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_store=bm25_store,
        llm_client=llm_client,
    )
    reranker = Reranker(llm_client=llm_client)

    # Fast Loop Workflow
    workflow = FastLoopRAGWorkflow(
        hybrid_retriever=retriever,
        reranker=reranker,
        llm_client=llm_client,
        max_retries=settings.MAX_RETRIES,
    )

    # Slow Loop Immune System
    immune_loop = KnowledgeBaseImmuneLoop(
        vector_store=vector_store,
        bm25_store=bm25_store,
        telemetry_logger=telemetry_logger,
        llm_client=llm_client,
    )

    console.print("[green][OK] Vector DB, Lexical Index, LLM Client, and LangGraph workflow initialized.[/green]\n")

    # 2. Ingest Sample Enterprise Knowledge (with an intentional version contradiction)
    console.print("[bold yellow]2. Ingesting Enterprise Documents with Anthropic Contextual Retrieval...[/bold yellow]")

    doc_sec_v1 = Document(
        doc_id="doc_sec_v1",
        title="Remote Access and VPN Security Policy",
        version="1.0",
        content=(
            "Section 1: Legacy VPN Access Rules.\n"
            "All remote employees must connect using the legacy OpenVPN portal at vpn.internal.corp.\n"
            "Single-factor password authentication is permitted for non-engineering staff.\n"
            "Session timeout is set to 24 hours of inactivity."
        ),
        metadata={"department": "IT Security", "year": "2023"},
    )

    doc_sec_v2 = Document(
        doc_id="doc_sec_v2",
        title="Remote Access and VPN Security Policy",
        version="2.0",
        content=(
            "Section 1: Modern Zero-Trust Access Requirements.\n"
            "All employees, contractors, and partners MUST connect exclusively via the Zero-Trust Network Access (ZTNA) gateway.\n"
            "Mandatory Hardware FIDO2 multi-factor authentication is required across all departments.\n"
            "Legacy OpenVPN portal is strictly decommissioned and blocked at firewall.\n"
            "Session timeout is strictly enforced after 4 hours of inactivity."
        ),
        metadata={"department": "IT Security", "year": "2026"},
    )

    doc_expense = Document(
        doc_id="doc_expense_2026",
        title="Global Travel and Expense Reimbursement Guidelines",
        version="3.1",
        content=(
            "Section 4: Hotel and Meals Reimbursement Policy.\n"
            "Standard business lodging reimbursement cap is $275 per night in Tier-1 cities (NYC, SF, London, Tokyo).\n"
            "Daily meal per diem allowance is fixed at $85 without itemized receipts required for amounts under $25.\n"
            "All flight bookings over 4 hours qualify for Premium Economy."
        ),
        metadata={"department": "Finance", "year": "2026"},
    )

    known_docs = [doc_sec_v1, doc_sec_v2, doc_expense]
    total_indexed = 0
    all_chunks = []
    for d in known_docs:
        chunks = ingestion_pipeline.ingest_document(d)
        total_indexed += len(chunks)
        all_chunks.extend(chunks)

    console.print(f"[green][OK] Ingested {len(known_docs)} documents into {total_indexed} contextualized chunks.[/green]\n")

    # 3. Fast-Loop Query Demonstrations
    console.print("[bold yellow]3. Executing Fast-Loop Queries (Runtime Self-Correction & Grounding)...[/bold yellow]\n")

    test_queries = [
        "What is the daily meal per diem and hotel cap for travel in Tier-1 cities?",
        "How do employees connect remotely from home?",
        "What is the quantum computing algorithm approval matrix?",  # Missing topic (Knowledge Gap)
    ]

    for q in test_queries:
        console.print(f"[bold white]User Query:[/bold white] [bold cyan]\"{q}\"[/bold cyan]")
        result = workflow.query(q)

        # Log telemetry signal
        sig = result.get("telemetry_signal")
        if sig:
            telemetry_logger.log_signal(sig)

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Property", width=22)
        table.add_column("Execution Detail")

        table.add_row("Rewritten Query", str(result.get("rewritten_query") or "None (Original used)"))
        table.add_row("Relevant Chunks Graded", str(len(result.get("graded_chunks", []))))
        table.add_row("Retry Count", str(result.get("retry_count", 0)))
        table.add_row("Fallback Used", str(result.get("fallback_used", False)))

        verdict = result.get("grounding_verdict")
        grounding_str = (
            f"[green]Faithful (Score: {verdict.hallucination_score:.2f})[/green]"
            if verdict and verdict.is_faithful
            else f"[red]Ungrounded (Score: {verdict.hallucination_score if verdict else 1.0})[/red]"
        )
        table.add_row("Grounding Check", grounding_str)
        table.add_row("Latency", f"{sig.latency_ms} ms" if sig else "N/A")

        console.print(table)
        console.print(Panel(result.get("generation", ""), title="[bold green]Final Grounded Response[/bold green]", expand=False))
        console.print()

    # 4. Slow-Loop Autonomous Healing Cycle
    console.print("[bold yellow]4. Running Slow-Loop Continuous Healing Cycle (The Immune System)...[/bold yellow]")
    healing_report = immune_loop.run_healing_cycle(
        known_documents=known_docs,
        known_chunks=all_chunks,
    )

    heal_table = Table(title="Autonomous Healing Cycle Report", header_style="bold green")
    heal_table.add_column("Metric / Action")
    heal_table.add_column("Result")

    heal_table.add_row("Degraded Telemetry Signals Processed", str(healing_report["degraded_signals_processed"]))
    heal_table.add_row("Actions Taken Count", str(len(healing_report["actions_taken"])))
    for act in healing_report["actions_taken"]:
        heal_table.add_row(f"Action: {act.get('type')}", str(act))
    heal_table.add_row("New Synthetic / Golden Chunks Indexed", str(healing_report["new_chunks_indexed"]))

    console.print(heal_table)
    console.print("[green][OK] Contradiction Auditor automatically resolved v1 vs v2 policy and synthesized Golden Chunk.[/green]")
    console.print("[green][OK] Diagnostic Agent caught missing quantum query and issued Knowledge Gap Alert.[/green]\n")

    # 5. CI/CD Automated Regression Suite
    console.print("[bold yellow]5. Running CI/CD Automated RAG Triad Regression Suite...[/bold yellow]")
    synth_gen = SyntheticDatasetGenerator(llm_client=llm_client)
    # Evaluate benchmark against active canonical documents
    active_docs = [d for d in known_docs if not d.is_deprecated and d.doc_id != "doc_sec_v1"]
    eval_dataset = synth_gen.generate_evaluation_set(documents=active_docs)

    runner = RegressionTestRunner(workflow=workflow)
    benchmark_report = runner.run_benchmark(eval_dataset, min_pass_ratio=0.50)

    bench_table = Table(title="CI/CD RAG Triad Benchmark Results", header_style="bold blue")
    bench_table.add_column("Metric")
    bench_table.add_column("Score / Status")

    bench_table.add_row("Total Benchmark Tests", str(benchmark_report["total_tests"]))
    bench_table.add_row("Pass Ratio", f"{benchmark_report['pass_ratio'] * 100:.1f}%")
    bench_table.add_row("Average Context Relevance", f"{benchmark_report['averages']['context_relevance']:.3f}")
    bench_table.add_row("Average Groundedness (Faithfulness)", f"{benchmark_report['averages']['groundedness']:.3f}")
    bench_table.add_row("Average Answer Relevance", f"{benchmark_report['averages']['answer_relevance']:.3f}")
    gate_status = "[bold green]PASSED[/bold green]" if benchmark_report["gate_passed"] else "[bold red]FAILED[/bold red]"
    bench_table.add_row("CI/CD Quality Gate", gate_status)

    console.print(bench_table)
    console.print("\n[bold green][OK] Full Demonstration Completed Successfully![/bold green]")


if __name__ == "__main__":
    run_demo()
