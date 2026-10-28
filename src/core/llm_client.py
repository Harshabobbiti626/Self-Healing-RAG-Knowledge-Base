import abc
import hashlib
import json
import logging
import math
import re
from typing import Any, Dict, List, Optional
import httpx

from config.settings import settings
from src.core.models import (
    ClaimVerification,
    DocumentRelevance,
    GroundingVerdict,
    RelevanceStatus,
)

logger = logging.getLogger(__name__)


class BaseLLMClient(abc.ABC):
    """Abstract interface for LLM operations in the Self-Healing RAG system."""

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        """Generate a response text from prompt."""
        pass

    @abc.abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate dense vector embeddings for a list of strings."""
        pass

    @abc.abstractmethod
    def grade_relevance(self, query: str, chunk_content: str, chunk_id: str) -> DocumentRelevance:
        """Fast-Loop Document Grader: Scores factual relevance of chunk to query."""
        pass

    @abc.abstractmethod
    def verify_grounding(
        self,
        query: str,
        answer: str,
        context_chunks: List[Dict[str, str]],
    ) -> GroundingVerdict:
        """Fast-Loop Grounding Guard: Checks answer claims against retrieved contexts."""
        pass


class MockLLMClient(BaseLLMClient):
    """Deterministic, local mock LLM client for testing, offline use, and verification."""

    def __init__(self, vector_dim: int = 1536):
        self.vector_dim = vector_dim

    def _deterministic_embedding(self, text: str) -> List[float]:
        """Creates a normalized deterministic pseudo-embedding vector from text hash."""
        # Use MD5/SHA256 to generate pseudo-random deterministic floats
        hash_bytes = hashlib.sha256(text.lower().encode("utf-8")).digest()
        vector = []
        for i in range(self.vector_dim):
            byte_idx = i % len(hash_bytes)
            val = (hash_bytes[byte_idx] / 255.0) * 2.0 - 1.0  # Range -1.0 to 1.0
            vector.append(val)

        # Normalize to unit length
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]
        return vector

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        # Check for query rewriting prompts
        if "rewrite" in prompt.lower() or "reformulate" in prompt.lower():
            # Extract query
            match = re.search(r'Query:\s*["\']?([^"\']+)["\']?', prompt, re.IGNORECASE)
            q = match.group(1) if match else "general policy"
            return f"detailed information regarding {q} specifications and policy guidelines"

        # Check for synthesis prompts with <context>
        if "<context>" in prompt.lower() or "retrieved context" in prompt.lower() or "sources:" in prompt.lower():
            ctx_match = re.search(r"<context>(.*?)</context>", prompt, re.DOTALL | re.IGNORECASE)
            if ctx_match and ctx_match.group(1).strip():
                lines = [
                    l.strip()
                    for l in ctx_match.group(1).split("\n")
                    if l.strip() and not l.strip().startswith("[") and len(l.strip()) > 10
                ]
                first_fact = lines[0] if lines else "the requested policy requirements"
                return f"According to the official documentation, {first_fact}"
            return "Based on the official documentation provided, the requested policy procedure is verified."

        return "This is a verified response grounded in the enterprise knowledge base."

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self._deterministic_embedding(t) for t in texts]

    def grade_relevance(self, query: str, chunk_content: str, chunk_id: str) -> DocumentRelevance:
        """Calculates token overlap and semantic relevance heuristic for grading."""
        query_words = set(re.findall(r"\w+", query.lower()))
        chunk_words = set(re.findall(r"\w+", chunk_content.lower()))

        # Remove very common stop words
        stopwords = {
            "what", "is", "are", "the", "a", "an", "for", "in", "to", "of",
            "and", "how", "do", "does", "can", "on", "with", "from"
        }
        query_keywords = {w for w in query_words if w not in stopwords and len(w) > 2}

        if not query_keywords:
            return DocumentRelevance(
                chunk_id=chunk_id,
                score=0.75,
                status=RelevanceStatus.RELEVANT,
                reasoning="Generic query terms matched chunk.",
            )

        def stem(w: str) -> str:
            w = w.lower()
            if w.endswith("ies"):
                return w[:-3] + "y"
            if w.endswith("es"):
                return w[:-2]
            if w.endswith("s") and len(w) > 3:
                return w[:-1]
            return w

        stemmed_query_keywords = {stem(w) for w in query_keywords}
        stemmed_chunk_words = {stem(w) for w in chunk_words}

        overlap = stemmed_query_keywords.intersection(stemmed_chunk_words)
        overlap_ratio = len(overlap) / max(1, len(stemmed_query_keywords))

        if overlap_ratio >= 0.25 or len(overlap) >= 1:
            score = min(1.0, 0.70 + (overlap_ratio * 0.30))
            status = RelevanceStatus.RELEVANT
            reasoning = f"Chunk contains {len(overlap)} matching query terms: {list(overlap)}."
        elif overlap_ratio > 0.05:
            score = 0.45
            status = RelevanceStatus.AMBIGUOUS
            reasoning = "Chunk has minor partial overlap but lacks direct key context."
        else:
            score = 0.15
            status = RelevanceStatus.IRRELEVANT
            reasoning = "No significant semantic overlap with query."

        return DocumentRelevance(
            chunk_id=chunk_id,
            score=round(score, 2),
            status=status,
            reasoning=reasoning,
        )

    def verify_grounding(
        self,
        query: str,
        answer: str,
        context_chunks: List[Dict[str, str]],
    ) -> GroundingVerdict:
        """Evaluates whether statements in the answer are supported by retrieved context."""
        if not context_chunks:
            # If the answer acknowledges lack of context/documentation, it is an honest refusal (not a hallucination)
            refusal_markers = ["could not find", "cannot find", "no verified documentation", "not found", "flagged for knowledge review"]
            if any(marker in answer.lower() for marker in refusal_markers):
                return GroundingVerdict(
                    is_faithful=True,
                    hallucination_score=0.0,
                    claims=[],
                    critique_feedback=None,
                )
            return GroundingVerdict(
                is_faithful=False,
                hallucination_score=1.0,
                claims=[
                    ClaimVerification(
                        claim=answer,
                        is_grounded=False,
                        status="unsupported",
                        critique="No context was retrieved to support this answer.",
                    )
                ],
                critique_feedback="Answer was generated without supporting context.",
            )

        # Check if the answer mentions facts that contradict the context or claim untruths
        all_context_text = " ".join([c.get("content", "") for c in context_chunks]).lower()
        answer_words = set(re.findall(r"\w{4,}", answer.lower()))
        context_words = set(re.findall(r"\w{4,}", all_context_text))

        unsupported_words = answer_words - context_words - {"based", "official", "documentation", "provided", "according", "policy", "guidelines"}
        hallucination_ratio = len(unsupported_words) / max(1, len(answer_words))

        is_faithful = hallucination_ratio < 0.60
        claims = [
            ClaimVerification(
                claim=answer,
                is_grounded=is_faithful,
                status="grounded" if is_faithful else "unsupported",
                supporting_chunk_ids=[c.get("chunk_id", "chunk_0") for c in context_chunks],
                critique=None if is_faithful else f"Novel unsupported concepts detected: {list(unsupported_words)[:5]}",
            )
        ]

        return GroundingVerdict(
            is_faithful=is_faithful,
            hallucination_score=round(hallucination_ratio, 2),
            claims=claims,
            citations=[c.get("chunk_id", "0") for c in context_chunks[:2]],
            critique_feedback=None if is_faithful else "Regenerate answer strictly using retrieved chunks.",
        )


class OpenAILLMClient(BaseLLMClient):
    """Production OpenAI client utilizing httpx."""

    def __init__(self, api_key: str, synthesis_model: str, embedding_model: str):
        self.api_key = api_key
        self.synthesis_model = synthesis_model
        self.embedding_model = embedding_model
        self.base_url = "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": self.synthesis_model,
                    "messages": messages,
                    "temperature": temperature,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{self.base_url}/embeddings",
                headers=self.headers,
                json={
                    "model": self.embedding_model,
                    "input": texts,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in data["data"]]

    def grade_relevance(self, query: str, chunk_content: str, chunk_id: str) -> DocumentRelevance:
        prompt = f"""You are a strict retrieval relevance grader.
Query: {query}
Retrieved Chunk:
{chunk_content}

Evaluate if the chunk contains information directly relevant to answering the query.
Return JSON with format:
{{
  "score": <float between 0.0 and 1.0>,
  "status": "<relevant|ambiguous|irrelevant>",
  "reasoning": "<one sentence explanation>"
}}"""
        raw = self.generate(prompt=prompt, system_instruction="Respond ONLY in valid JSON.")
        try:
            data = json.loads(raw.strip().replace("```json", "").replace("```", ""))
            return DocumentRelevance(
                chunk_id=chunk_id,
                score=float(data.get("score", 0.5)),
                status=RelevanceStatus(data.get("status", "ambiguous")),
                reasoning=data.get("reasoning", ""),
            )
        except Exception:
            return DocumentRelevance(
                chunk_id=chunk_id,
                score=0.5,
                status=RelevanceStatus.AMBIGUOUS,
                reasoning="Failed to parse LLM grader output.",
            )

    def verify_grounding(
        self,
        query: str,
        answer: str,
        context_chunks: List[Dict[str, str]],
    ) -> GroundingVerdict:
        if not context_chunks:
            refusal_markers = ["could not find", "cannot find", "no verified documentation", "not found", "flagged for knowledge review"]
            if any(marker in answer.lower() for marker in refusal_markers):
                return GroundingVerdict(is_faithful=True, hallucination_score=0.0)

        contexts_str = "\n\n".join(
            [f"[{c.get('chunk_id')}]: {c.get('content')}" for c in context_chunks]
        )
        prompt = f"""You are a hallucination and faithfulness auditor.
User Query: {query}
Retrieved Context:
{contexts_str}

Generated Answer:
{answer}

Verify if every claim in the generated answer is strictly grounded in the retrieved context.
Return JSON:
{{
  "is_faithful": <true|false>,
  "hallucination_score": <float 0.0 to 1.0, 0.0 = completely faithful>,
  "critique_feedback": "<None or instructions to fix ungrounded claims>"
}}"""
        raw = self.generate(prompt=prompt, system_instruction="Respond ONLY in valid JSON.")
        try:
            data = json.loads(raw.strip().replace("```json", "").replace("```", ""))
            return GroundingVerdict(
                is_faithful=bool(data.get("is_faithful", True)),
                hallucination_score=float(data.get("hallucination_score", 0.0)),
                critique_feedback=data.get("critique_feedback"),
                citations=[c.get("chunk_id", "") for c in context_chunks],
            )
        except Exception:
            return GroundingVerdict(is_faithful=True, hallucination_score=0.0)


class GeminiLLMClient(BaseLLMClient):
    """Production Google Gemini client utilizing httpx REST endpoints."""

    def __init__(
        self,
        api_key: str,
        synthesis_model: str = "gemini-3.8-flash",
        embedding_model: str = "gemini-embedding-001",
    ):
        self.api_key = api_key.strip()
        self.synthesis_model = synthesis_model
        self.embedding_model = embedding_model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        url = f"{self.base_url}/{self.synthesis_model}:generateContent?key={self.api_key}"
        body: Dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature},
        }
        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=body)
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return ""
            parts = candidates[0].get("content", {}).get("parts", [])
            return parts[0].get("text", "") if parts else ""

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        url = f"{self.base_url}/{self.embedding_model}:batchEmbedContents?key={self.api_key}"
        requests = [
            {
                "model": f"models/{self.embedding_model}",
                "content": {"parts": [{"text": t}]},
            }
            for t in texts
        ]
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json={"requests": requests})
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get("embeddings", [])
            return [e.get("values", []) for e in embeddings]

    def grade_relevance(self, query: str, chunk_content: str, chunk_id: str) -> DocumentRelevance:
        prompt = f"""You are a strict retrieval relevance grader.
Query: {query}
Retrieved Chunk:
{chunk_content}

Evaluate if the chunk contains information directly relevant to answering the query.
Return JSON with format:
{{
  "score": <float between 0.0 and 1.0>,
  "status": "<relevant|ambiguous|irrelevant>",
  "reasoning": "<one sentence explanation>"
}}"""
        raw = self.generate(prompt=prompt, system_instruction="Respond ONLY in valid JSON.")
        try:
            data = json.loads(raw.strip().replace("```json", "").replace("```", ""))
            return DocumentRelevance(
                chunk_id=chunk_id,
                score=float(data.get("score", 0.5)),
                status=RelevanceStatus(data.get("status", "ambiguous")),
                reasoning=data.get("reasoning", ""),
            )
        except Exception:
            return DocumentRelevance(
                chunk_id=chunk_id,
                score=0.5,
                status=RelevanceStatus.AMBIGUOUS,
                reasoning="Failed to parse LLM grader output.",
            )

    def verify_grounding(
        self,
        query: str,
        answer: str,
        context_chunks: List[Dict[str, str]],
    ) -> GroundingVerdict:
        if not context_chunks:
            refusal_markers = ["could not find", "cannot find", "no verified documentation", "not found", "flagged for knowledge review"]
            if any(marker in answer.lower() for marker in refusal_markers):
                return GroundingVerdict(is_faithful=True, hallucination_score=0.0)

        contexts_str = "\n\n".join(
            [f"[{c.get('chunk_id')}]: {c.get('content')}" for c in context_chunks]
        )
        prompt = f"""You are a hallucination and faithfulness auditor.
User Query: {query}
Retrieved Context:
{contexts_str}

Generated Answer:
{answer}

Verify if every claim in the generated answer is strictly grounded in the retrieved context.
Return JSON:
{{
  "is_faithful": <true|false>,
  "hallucination_score": <float 0.0 to 1.0, 0.0 = completely faithful>,
  "critique_feedback": "<None or instructions to fix ungrounded claims>"
}}"""
        raw = self.generate(prompt=prompt, system_instruction="Respond ONLY in valid JSON.")
        try:
            data = json.loads(raw.strip().replace("```json", "").replace("```", ""))
            return GroundingVerdict(
                is_faithful=bool(data.get("is_faithful", True)),
                hallucination_score=float(data.get("hallucination_score", 0.0)),
                critique_feedback=data.get("critique_feedback"),
                citations=[c.get("chunk_id", "") for c in context_chunks],
            )
        except Exception:
            return GroundingVerdict(is_faithful=True, hallucination_score=0.0)


def get_llm_client() -> BaseLLMClient:
    """Factory to create LLM client based on configuration."""
    if settings.LLM_PROVIDER == "openai" and settings.OPENAI_API_KEY:
        return OpenAILLMClient(
            api_key=settings.OPENAI_API_KEY,
            synthesis_model=settings.SYNTHESIS_MODEL,
            embedding_model=settings.EMBEDDING_MODEL,
        )
    if settings.LLM_PROVIDER == "gemini" and settings.GEMINI_API_KEY:
        return GeminiLLMClient(
            api_key=settings.GEMINI_API_KEY,
            synthesis_model=settings.SYNTHESIS_MODEL,
            embedding_model=settings.EMBEDDING_MODEL,
        )
    return MockLLMClient(vector_dim=settings.VECTOR_SIZE)
