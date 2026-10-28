import json
from pathlib import Path
from typing import Dict, List, Optional
import uuid

from src.core.models import ContentType, Document


class DocumentParser:
    """Enterprise document parser for ingesting structured and unstructured documents."""

    @staticmethod
    def parse_text(
        content: str,
        title: str,
        doc_id: Optional[str] = None,
        content_type: ContentType = ContentType.TEXT,
        metadata: Optional[Dict] = None,
    ) -> Document:
        """Parse raw string content into Document."""
        return Document(
            doc_id=doc_id or f"doc_{uuid.uuid4().hex[:8]}",
            title=title,
            content=content.strip(),
            content_type=content_type,
            metadata=metadata or {},
        )

    @classmethod
    def parse_file(cls, file_path: Path | str) -> Document:
        """Parse a local file into a Document model."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        raw_text = path.read_text(encoding="utf-8")
        ext = path.suffix.lower()

        if ext in [".md", ".markdown"]:
            c_type = ContentType.MARKDOWN
        elif ext == ".json":
            c_type = ContentType.JSON
            try:
                data = json.loads(raw_text)
                if isinstance(data, dict) and "content" in data:
                    return cls.parse_text(
                        content=data["content"],
                        title=data.get("title", path.stem),
                        doc_id=data.get("doc_id", f"doc_{uuid.uuid4().hex[:8]}"),
                        content_type=c_type,
                        metadata=data.get("metadata", {}),
                    )
            except Exception:
                pass
        else:
            c_type = ContentType.TEXT

        return cls.parse_text(
            content=raw_text,
            title=path.stem.replace("_", " ").title(),
            doc_id=f"doc_{path.stem}_{uuid.uuid4().hex[:6]}",
            content_type=c_type,
            metadata={"source_path": str(path.resolve()), "file_name": path.name},
        )
