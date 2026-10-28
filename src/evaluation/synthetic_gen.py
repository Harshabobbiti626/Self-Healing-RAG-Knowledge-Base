import json
from pathlib import Path
import re
from typing import Dict, List
from src.core.llm_client import BaseLLMClient
from src.core.models import Document


class SyntheticDatasetGenerator:
    """Generates synthetic QA evaluation benchmarks directly from ingested documents."""

    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client

    def generate_evaluation_set(
        self,
        documents: List[Document],
        num_questions_per_doc: int = 2,
    ) -> List[Dict[str, str]]:
        """Extracts key factual questions and ground truth answers from docs."""
        dataset = []

        for doc in documents:
            paragraphs = [p.strip() for p in doc.content.split("\n\n") if len(p.strip()) > 60]
            sampled_paras = paragraphs[:num_questions_per_doc]

            for idx, p in enumerate(sampled_paras):
                first_sentence = re.split(r"[.!?]", p)[0]
                question = f"What is specified regarding {doc.title}?"
                dataset.append({
                    "id": f"eval_{doc.doc_id}_{idx}",
                    "doc_id": doc.doc_id,
                    "doc_title": doc.title,
                    "question": question,
                    "reference_context": p,
                    "ground_truth": first_sentence.strip(),
                })

        return dataset

    def save_dataset(self, dataset: List[Dict[str, str]], output_path: Path | str) -> None:
        """Saves evaluation dataset to disk as JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
