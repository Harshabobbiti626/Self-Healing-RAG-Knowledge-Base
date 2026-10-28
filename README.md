# Enterprise Self-Healing RAG Knowledge Base

A state-of-the-art, closed-loop **Self-Healing Retrieval-Augmented Generation (RAG)** knowledge base system built from scratch for production scale. 

Unlike conventional open-loop RAG pipelines that degrade silently under data drift and hallucinate on noisy retrieval hits, this system implements a **Dual-Loop Closed Architecture**:

```
+===================================================================================================+
|                                    THE DUAL-LOOP ARCHITECTURE                                     |
+===================================================================================================+
                                                                                                     
    FAST LOOP: Real-Time Query Self-Correction (Latency budget: 800ms - 2.5s)                        
    +-----------------------------------------------------------------------------------------+     
    | [User Query]                                                                            |     
    |      |                                                                                  |     
    |      v                                                                                  |     
    | [Query Transformation] ---> [Hybrid Retrieval] ---> [Document Grader]                   |     
    |    (Decompose/HyDE)            (Dense + BM25)              |                            |     
    |                                                            |--- [Irrelevant] -> [Rewrite]     
    |                                                            v                            |     
    | [Answer Grounding Check] <-- [Context Synthesis] <--- [Relevant Chunks]                 |     
    |         |                                                                               |     
    |         +--- [Hallucination Detected] ---> [Regenerate with Negative Constraints]       |     
    |         v                                                                               |     
    |   [Verified Output]                                                                     |     
    +-----------------------------------------------------------------------------------------+     
           |                                                                                        
           | Traces, Failed Retrievals, Grounding Errors, User Downvotes                            
           v                                                                                        
    SLOW LOOP: Autonomous Knowledge Base Repair (Async Cron / Event Queue)                          
    +-----------------------------------------------------------------------------------------+     
    | [Failure Telemetry Store] (Phoenix / Langfuse / OTel)                                   |     
    |      |                                                                                  |     
    |      v                                                                                  |     
    | [Root-Cause Diagnostic Agent]                                                           |     
    |      |                                                                                  |     
    |      +---> Knowledge Gap? ---------> [Trigger Web/API Ingestion or Human Ticket]        |     
    |      +---> Semantic Bleed? --------> [Contextual Re-Chunking & Parent Expansion]        |     
    |      +---> Fact Contradiction? ----> [Cross-Doc Temporal Conflict Auditor & Diff Tool]  |     
    |      +---> Poor Recall? -----------> [Generate Synthetic Q&A "Reflection Artifacts"]   |     
    |      |                                                                                  |     
    |      v                                                                                  |     
    | [Index Auto-Patch & Re-Embedding] ---> [CI/CD Synthetic Canary Regression (RAGAS)]      |     
    +-----------------------------------------------------------------------------------------+     
```

---

## Key Pillars & Innovations

### 1. Ingestion: Anthropic Contextual Retrieval
* Eliminates orphaned chunk ambiguity by automatically prepending document-level situational context to every chunk prior to dense & BM25 indexing.
* Reduces retrieval failure rates by up to 49%.

### 2. Fast Loop: Runtime Query Self-Correction (LangGraph)
* **Hybrid Search + Reciprocal Rank Fusion (RRF):** Blends Qdrant dense vector cosine similarity with BM25Okapi lexical token search.
* **Precision Reranking:** Re-scores top candidates via cross-encoder ranking.
* **Document Grader (CRAG):** Evaluates factual relevance to query; automatically routes to query reformulation (HyDE / sub-queries) when context is insufficient.
* **Grounding Guard:** Performs claim-level Natural Language Inference (NLI) to catch hallucinations and triggers self-critique regeneration (max 2 retries).

### 3. Slow Loop: Autonomous Knowledge Base Repair (The Immune System)
* **Telemetry Buffer:** Continuously captures low-confidence queries, grounding failures, and user negative feedback.
* **Root-Cause Diagnostic Agent:** Categorizes failures into *Knowledge Gaps*, *Semantic Bleeds*, *Temporal Contradictions*, and *Vocabulary Blindspots*.
* **Contradiction Auditor:** Identifies multi-version document conflicts, deactivates stale chunks, and synthesizes **Golden Reconciliation Chunks**.
* **Reflection Artifacts:** Generates synthetic Q&A chunks bridging colloquial user queries to canonical documentation.

### 4. CI/CD Automated Regression Harness
* Continuous synthetic evaluation measuring the **RAG Triad**:
  1. **Context Relevance**
  2. **Groundedness / Faithfulness**
  3. **Answer Relevance**
* Automatic deployment quality gate blocking regressions.

---

## Quickstart

### 1. Environment Setup
```bash
# Activate virtual environment
.venv\Scripts\activate   # Windows
# or: source .venv/bin/activate  # Linux/macOS

# Copy environment template
cp .env.example .env
```

### 2. Run the Interactive End-to-End Demonstration
```bash
.venv\Scripts\python src\main.py
```

### 3. Run the Automated Test Suite
```bash
.venv\Scripts\pytest -v
```

---

## Repository Structure

```
├── config/
│   ├── settings.py             # Pydantic BaseSettings, model configs & thresholds
├── src/
│   ├── core/
│   │   ├── models.py           # Domain models: Document, Chunk, GroundingVerdict, TelemetrySignal
│   │   ├── state.py            # LangGraph RAGState TypedDict
│   │   └── llm_client.py       # LLM abstraction (Mock, OpenAI, Gemini)
│   ├── ingestion/
│   │   ├── parser.py           # Multi-format document parser (Text, Markdown, JSON)
│   │   ├── contextual_chunker.py # Anthropic Contextual Retrieval chunker
│   │   └── pipeline.py         # End-to-end ingestion pipeline
│   ├── storage/
│   │   ├── vector_store.py     # Qdrant client wrapper
│   │   ├── bm25_store.py       # BM25Okapi lexical search index
│   │   ├── hybrid_retriever.py # Reciprocal Rank Fusion (RRF) searcher
│   │   └── reranker.py         # Precision reranker
│   ├── fast_loop/
│   │   ├── query_rewriter.py   # HyDE & sub-query reformulation
│   │   ├── doc_grader.py       # Document relevance grader
│   │   ├── generator.py        # Context-grounded response generator
│   │   ├── grounding_guard.py  # Hallucination and claim auditor
│   │   └── workflow.py         # Cyclic LangGraph state machine
│   ├── slow_loop/
│   │   ├── telemetry.py        # Telemetry logger & failure buffer
│   │   ├── diagnostic_agent.py # Root-cause diagnostic agent
│   │   ├── contradiction_auditor.py # Cross-doc conflict auditor & golden chunk synthesizer
│   │   ├── patch_generator.py  # Reflection Q&A and re-chunking patcher
│   │   └── immune_loop.py      # Autonomous repair orchestrator
│   ├── evaluation/
│   │   ├── metrics.py          # RAG Triad evaluator
│   │   ├── synthetic_gen.py    # Synthetic benchmark QA generator
│   │   └── regression_runner.py# CI/CD regression harness
│   └── main.py                 # Interactive demonstration dashboard
├── tests/
│   ├── test_ingestion.py
│   ├── test_fast_loop.py
│   ├── test_slow_loop.py
│   └── test_evaluation.py
├── pyproject.toml
└── .env.example
```

---

## Configuration (`.env`)

```ini
LLM_PROVIDER=mock               # "mock" for offline dev, "openai", or "gemini"
OPENAI_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
QDRANT_LOCATION=:memory:        # or "./data/qdrant_storage"
RELEVANCE_THRESHOLD=0.65
MAX_RETRIES=2
```
