import re
from typing import List, Optional
import uuid

from src.core.llm_client import BaseLLMClient
from src.core.models import Chunk, Document


class ContextualChunker:
    """Implements Anthropic's Contextual Retrieval chunking.
    
    Each chunk is enriched with a short document-level contextual header
    before dense and lexical indexing, resolving semantic ambiguity.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        target_chunk_chars: int = 500,
        chunk_overlap_chars: int = 80,
    ):
        self.llm_client = llm_client
        self.target_chunk_chars = target_chunk_chars
        self.chunk_overlap_chars = chunk_overlap_chars

    def _split_into_raw_segments(self, text: str) -> List[str]:
        """Splits document text into paragraphs or sentence-bounded segments."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current_chunk = ""

        for p in paragraphs:
            if len(current_chunk) + len(p) <= self.target_chunk_chars:
                current_chunk = f"{current_chunk}\n\n{p}".strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                # If paragraph itself is too large, split by sentence
                if len(p) > self.target_chunk_chars:
                    sentences = re.split(r"(?<=[.!?])\s+", p)
                    sub_chunk = ""
                    for s in sentences:
                        if len(sub_chunk) + len(s) <= self.target_chunk_chars:
                            sub_chunk = f"{sub_chunk} {s}".strip()
                        else:
                            if sub_chunk:
                                chunks.append(sub_chunk)
                            sub_chunk = s
                    if sub_chunk:
                        chunks.append(sub_chunk)
                    current_chunk = ""
                else:
                    current_chunk = p

        if current_chunk:
            chunks.append(current_chunk)

        return chunks if chunks else [text]

    def _generate_contextual_prefix(self, doc: Document, chunk_text: str) -> str:
        """Generates document-level situational context to prepend to the chunk."""
        doc_header = f"Document: '{doc.title}' (Version {doc.version})"

        # If live LLM client is configured (not mock), call LLM to generate situational summary
        from src.core.llm_client import MockLLMClient
        if not isinstance(self.llm_client, MockLLMClient):
            prompt = f"""<document>
Title: {doc.title}
{doc.content[:1200]}
</document>

Here is a chunk from the document:
<chunk>
{chunk_text}
</chunk>

Provide a concise 1-sentence context situating this chunk within the overall document to improve retrieval."""
            try:
                situational_context = self.llm_client.generate(prompt=prompt, temperature=0.0).strip()
                if situational_context:
                    return f"{doc_header} | {situational_context}"
            except Exception:
                pass

        # Fast heuristic fallback
        topic_hint = doc.metadata.get("category") or doc.metadata.get("topic")
        if topic_hint:
            return f"{doc_header} | Category: {topic_hint}"
        return f"{doc_header}"

    def chunk_document(self, doc: Document) -> List[Chunk]:
        """Transforms a Document into contextualized Chunks ready for indexing."""
        raw_segments = self._split_into_raw_segments(doc.content)
        chunks: List[Chunk] = []

        for idx, segment in enumerate(raw_segments):
            prefix = self._generate_contextual_prefix(doc, segment)
            chunk = Chunk(
                chunk_id=f"{doc.doc_id}_c{idx}_{uuid.uuid4().hex[:4]}",
                doc_id=doc.doc_id,
                content=segment,
                contextual_prefix=prefix,
                chunk_index=idx,
                metadata={
                    "doc_title": doc.title,
                    "doc_version": doc.version,
                    "valid_from": str(doc.valid_from) if doc.valid_from else None,
                    "valid_until": str(doc.valid_until) if doc.valid_until else None,
                    **doc.metadata,
                },
            )
            chunks.append(chunk)

        return chunks
