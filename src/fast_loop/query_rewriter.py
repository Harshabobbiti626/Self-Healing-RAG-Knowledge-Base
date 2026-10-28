import logging
from typing import List
from src.core.llm_client import BaseLLMClient

logger = logging.getLogger(__name__)


class QueryRewriter:
    """Pre-retrieval and in-flight query transformation for self-healing RAG."""

    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client

    def rewrite_for_retrieval(self, query: str, failure_reason: str = "") -> str:
        """Reformulates a user query that produced poor/irrelevant retrieval hits."""
        prompt = f"""You are an expert query optimizer for enterprise knowledge search.
Original User Query: "{query}"
Retrieval Diagnostic: "{failure_reason or 'No relevant documents found.'}"

Rewrite this query to maximize semantic retrieval hit rate. 
Expand implicit terms, drop conversational noise, and add key enterprise terminology.
Provide ONLY the rewritten query text:"""
        rewritten = self.llm_client.generate(prompt=prompt, temperature=0.2).strip()
        # Clean any surrounding quotes
        rewritten = rewritten.strip('"\'')
        logger.info("Rewrote query from '%s' to '%s'", query, rewritten)
        return rewritten

    def generate_hyde_prompt(self, query: str) -> str:
        """Generates a hypothetical document answering the query for HyDE embedding."""
        prompt = f"""Write a concise, factual excerpt from an enterprise policy or technical documentation
that directly and completely answers this question:
Question: {query}
Excerpt:"""
        return self.llm_client.generate(prompt=prompt, temperature=0.0).strip()

    def decompose_query(self, query: str) -> List[str]:
        """Decomposes a complex multi-part query into atomic sub-questions."""
        prompt = f"""Decompose this complex query into 2-3 atomic sub-questions.
Query: {query}
Return ONLY one sub-question per line:"""
        raw = self.llm_client.generate(prompt=prompt).strip()
        lines = [line.strip("- *1234567890. ") for line in raw.split("\n") if line.strip()]
        return lines if lines else [query]
