import logging
from typing import Any, Dict, List, Optional
from src.core.llm_client import BaseLLMClient

logger = logging.getLogger(__name__)


class GroundedGenerator:
    """Synthesizes responses strictly grounded in retrieved and graded context."""

    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client

    def generate_response(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        critique_feedback: Optional[str] = None,
    ) -> str:
        """Generates answer with in-text citations from context chunks."""
        if not context_chunks:
            return (
                "I searched the enterprise knowledge base, but could not find verified "
                "documentation to answer your question. This query has been flagged for knowledge review."
            )

        context_blocks = []
        for idx, chunk in enumerate(context_chunks):
            cid = chunk.get("chunk_id", f"chunk_{idx}")
            doc_title = chunk.get("payload", {}).get("doc_title", "Document")
            context_blocks.append(f"[{cid} | {doc_title}]:\n{chunk.get('content')}")

        joined_context = "\n\n".join(context_blocks)

        system_instruction = (
            "You are an enterprise knowledge assistant. Answer the user question accurately, "
            "strictly using only the verified facts provided in the <context>. "
            "Cite every claim with its corresponding chunk citation format [chunk_id]. "
            "Never fabricate facts not present in the context."
        )

        critique_block = ""
        if critique_feedback:
            critique_block = (
                f"\n\nCRITICAL FIX REQUIRED FROM PREVIOUS DRAFT:\n{critique_feedback}\n"
                f"You must strictly adhere to the context and remove any hallucinated claims."
            )

        prompt = f"""<context>
{joined_context}
</context>

User Question: {query}{critique_block}

Grounded Answer:"""

        response = self.llm_client.generate(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=0.1,
        )
        return response.strip()
