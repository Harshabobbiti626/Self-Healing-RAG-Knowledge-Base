import pytest
from src.core.llm_client import MockLLMClient
from src.core.models import Document
from src.evaluation.metrics import RAGEvaluator
from src.evaluation.synthetic_gen import SyntheticDatasetGenerator


def test_rag_triad_evaluator():
    evaluator = RAGEvaluator()
    query = "What is the lodging reimbursement cap in Tier-1 cities?"
    retrieved_contexts = [
        "Lodging reimbursement cap is $275 per night in Tier-1 cities such as NYC and London."
    ]
    generated_answer = (
        "The standard lodging reimbursement cap in Tier-1 cities is $275 per night."
    )

    metrics = evaluator.evaluate_triad(
        query=query,
        retrieved_contexts=retrieved_contexts,
        generated_answer=generated_answer,
    )

    assert metrics.context_relevance > 0.50
    assert metrics.groundedness > 0.50
    assert metrics.answer_relevance > 0.50
    assert metrics.overall_score > 0.50
    assert metrics.passed is True


def test_synthetic_dataset_generator():
    llm = MockLLMClient()
    gen = SyntheticDatasetGenerator(llm_client=llm)

    docs = [
        Document(
            doc_id="doc_test_syn",
            title="Data Retention Policy",
            version="1.0",
            content="Customer audit records must be retained for 7 years according to compliance guidelines.",
        )
    ]

    dataset = gen.generate_evaluation_set(documents=docs, num_questions_per_doc=1)
    assert len(dataset) == 1
    assert "Data Retention Policy" in dataset[0]["question"]
    assert dataset[0]["doc_id"] == "doc_test_syn"
