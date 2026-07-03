"""Document researcher tool — load a document and return relevant passages.

Loads a document (URL, local file, or directory), chunks it, and uses BM25
keyword ranking to return the most relevant passages.  For small documents
that fit within the top-k, all chunks are returned directly.

No vectorstore or embeddings model required.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from langchain_core.tools import BaseTool, tool

from tools.research._loaders import load_source, rank_chunks, split_documents

logger = logging.getLogger(__name__)


def build_doc_research(
    *,
    files_dir: Optional[Path] = None,
    extract_images: bool = False,
) -> BaseTool:
    """Build a ``doc_research`` tool bound to a session.

    Binding *files_dir* lets the tool resolve session-relative paths the
    agent passes (e.g. ``/uploads/report.pdf``).  When *extract_images* is
    True, embedded images in PDF/DOCX/PPTX documents are extracted to
    ``{files_dir}/doc_images`` and referenced inline in the returned
    passages so the agent can inspect them with ``view_image``.
    """

    @tool
    async def doc_research(source: str, query: str, k: int = 10) -> str:
        """Search a specific document for passages relevant to a query.

        Loads the document, chunks it, and returns the top-k most relevant
        passages ranked by keyword similarity (BM25).  When the document
        contains images and a vision model is available, passages include
        inline image references (view_image to inspect them).

        Args:
            source: URL (http/https), local file path, or directory path.
            query: The research question to answer from the document.
            k: Number of top passages to return (default 10).
        """
        try:
            docs = await load_source(
                source,
                files_dir=files_dir,
                extract_images=extract_images,
            )
        except Exception as exc:
            return f"Could not load source '{source}': {exc}"

        if not docs:
            return f"No content found in: {source}"

        chunks = await asyncio.to_thread(split_documents, docs)
        if not chunks:
            return "Document loaded but contained no extractable text."

        logger.info(
            "doc_research: %d chunks from %s, returning top %d",
            len(chunks), source, k,
        )

        if len(chunks) <= k:
            relevant = chunks
        else:
            relevant = await asyncio.to_thread(rank_chunks, chunks, query, k)

        return _format_passages(relevant, source)

    return doc_research


# Module-level default (no session binding, no image extraction) for the
# DeepAgent / notebook path that imports ``doc_research`` directly.
doc_research = build_doc_research()


def _format_passages(docs: list, source: str) -> str:
    """Format ranked passages into a readable string."""
    if not docs:
        return f"No relevant passages found in: {source}"
    parts: list[str] = []
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", source)
        parts.append(f"[{i}] Source: {src}\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)
