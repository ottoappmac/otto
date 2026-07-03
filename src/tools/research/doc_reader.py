"""Document reader tool — LLM reads a document and answers a question.

Uses a parallel map → merge strategy (the "scan" approach):
1. Load and chunk the document
2. LLM summarises each chunk in parallel
3. LLM merges the summaries into one coherent answer

Requires a ``BaseChatModel`` instance at construction time.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import re
from pathlib import Path
from typing import List, Optional, Type

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools.base import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from tools.research._loaders import load_source, split_documents

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_MAP_PROMPT = ChatPromptTemplate.from_messages([
    ("human",
     "Summarise the key facts and findings from this document section "
     "concisely:\n\n{context}"),
])

_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("human",
     "Read the following document and answer the question below.\n\n"
     "Document:\n{context}\n\n"
     "Question: {question}\n\n"
     "Answer based only on the document content above."),
])

_MERGE_PROMPT = ChatPromptTemplate.from_messages([
    ("human",
     "You have summaries derived from separate sections of a document.\n\n"
     "{summaries}\n\n"
     "Question: {question}\n\n"
     "Synthesise these into one final, coherent answer. "
     "Resolve any contradictions and remove duplication."),
])

# ---------------------------------------------------------------------------
# Image reference handling
# ---------------------------------------------------------------------------

_IMAGE_PLACEHOLDER_RE = re.compile(r"\[IMAGE:[^\]]+\]")
_MAX_IMAGE_REFS = 50


def _collect_image_refs(docs: list) -> list[str]:
    """Extract verbatim ``[IMAGE: ...]`` placeholders from loaded documents."""
    refs: list[str] = []
    for doc in docs:
        refs.extend(_IMAGE_PLACEHOLDER_RE.findall(doc.page_content))
        if len(refs) >= _MAX_IMAGE_REFS:
            break
    return refs[:_MAX_IMAGE_REFS]


def _append_image_refs(answer: str, image_refs: list[str]) -> str:
    """Append a verbatim image-reference list so exact paths reach the agent."""
    if not image_refs:
        return answer
    listing = "\n".join(f"- {ref}" for ref in image_refs)
    return (
        f"{answer}\n\n"
        f"Images found in this document ({len(image_refs)}). "
        f"Call view_image with the exact path to inspect or ask about one:\n"
        f"{listing}"
    )


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------


class DocReaderInput(BaseModel):
    """Input schema for :class:`DocReader`."""

    source: str = Field(
        ...,
        description=(
            "URL (http/https) or local file path of the document to read. "
            "Supports web pages, PDFs, plain text, Markdown, and HTML."
        ),
    )
    question: str = Field(
        ...,
        description="The question to answer about the document.",
    )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class DocReader(BaseTool):
    """Reads a document and answers a question using parallel map → merge.

    The LLM summarises each chunk independently (fast, parallel), then merges
    all summaries into one answer.

    Args:
        llm: Any ``BaseChatModel`` (Anthropic, OpenAI, Bedrock, etc.).
        chunk_size: Target character size per chunk.
        chunk_overlap: Overlap between consecutive chunks.
        max_concurrency: Max parallel LLM calls during the map phase.
    """

    name: str = "doc_reader"
    description: str = (
        "Reads a document (URL or local file) and answers "
        "a question about it. The document is split into "
        "sections, each summarised in parallel, then merged "
        "into a single coherent answer. Supports web pages, "
        "doc, docx, ppt, pptx, xlsx, csv, json, xml, rst, txt, md, html, htm, "
        "PDFs, plain text, Markdown, and HTML."
    )
    args_schema: Type[BaseModel] = DocReaderInput

    llm: BaseChatModel
    files_dir: Optional[Path] = None
    extract_images: bool = False
    chunk_size: int = 1500
    chunk_overlap: int = 150
    max_concurrency: int = 5

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_llm(
        cls,
        llm: BaseChatModel,
        *,
        files_dir: Optional[Path] = None,
        extract_images: bool = False,
        chunk_size: int = 1500,
        chunk_overlap: int = 150,
        max_concurrency: int = 5,
    ) -> DocReader:
        """Create a :class:`DocReader` from a chat model.

        When *extract_images* is True (set by the backend when a vision model
        is available), embedded images in PDF/DOCX/PPTX documents are extracted
        to ``{files_dir}/doc_images`` and referenced inline so the agent can
        inspect them with ``view_image``.
        """
        return cls(
            llm=llm,
            files_dir=files_dir,
            extract_images=extract_images,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_concurrency=max_concurrency,
        )

    # ── Core logic ────────────────────────────────────────────────────────

    async def _read(self, source: str, question: str) -> str:
        try:
            docs = await load_source(
                source,
                files_dir=self.files_dir,
                extract_images=self.extract_images,
            )
        except Exception as exc:
            logger.error("DocReader failed to load %s: %s", source, exc)
            return f"Error loading document: {type(exc).__name__}: {exc}"
        if not docs:
            return f"Could not load any content from: {source}"

        chunks = await asyncio.to_thread(
            split_documents,
            docs,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        if not chunks:
            return "Document loaded but contained no extractable text."

        logger.info("DocReader: %d chunks from %s", len(chunks), source)

        # The map/merge LLM passes paraphrase away the literal image
        # placeholders, so collect them verbatim from the loaded text and
        # re-attach after summarisation.  This keeps the exact
        # ``/doc_images/...`` paths the agent must pass to ``view_image``.
        image_refs = _collect_image_refs(docs)

        parser = StrOutputParser()
        map_chain = _MAP_PROMPT | self.llm | parser
        qa_chain = _QA_PROMPT | self.llm | parser
        merge_chain = _MERGE_PROMPT | self.llm | parser

        if len(chunks) == 1:
            answer = await qa_chain.ainvoke({
                "context": chunks[0].page_content,
                "question": question,
            })
            return _append_image_refs(answer, image_refs)

        summaries: List[str] = await map_chain.abatch(
            [{"context": c.page_content} for c in chunks],
            config={"max_concurrency": self.max_concurrency},
        )

        joined = "\n\n---\n\n".join(
            f"Section {i}: {s}" for i, s in enumerate(summaries, 1)
        )
        answer = await merge_chain.ainvoke({
            "summaries": joined,
            "question": question,
        })
        return _append_image_refs(answer, image_refs)

    # ── BaseTool interface ────────────────────────────────────────────────

    def _run(
        self,
        source: str,
        question: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    asyncio.run, self._read(source, question),
                ).result()
        except RuntimeError:
            return asyncio.run(self._read(source, question))

    async def _arun(
        self,
        source: str,
        question: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return await self._read(source, question)
