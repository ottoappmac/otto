"""Loader that uses unstructured to load HTML files."""
from __future__ import annotations

import asyncio
import logging
import ssl
import time
from io import BytesIO

import aiohttp
import certifi
from langchain_community.document_transformers import Html2TextTransformer
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_unstructured.document_loaders import UnstructuredLoader
from playwright.async_api import async_playwright

logger = logging.getLogger(__file__)


class WebLoader(BaseLoader):
    """Loader that uses unstructured to load HTML files."""

    def __init__(
        self,
        urls: list[str],
        headless: bool = False,
        timeout: int = 10,
        max_concurrency: int = 10,
    ):
        self.urls = urls
        self.headless = headless
        self.timeout = timeout
        self.max_concurrency = max_concurrency
        self.all_docs: list[Document] = []

    async def _process_url(self, url: str) -> list[Document]:
        try:
            logger.debug("Loading URL: %s", url)
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:

                    if response.status != 200:
                        logger.error("HTTP %d: %s", response.status, url)
                        return []

                    content_type = response.headers.get("Content-Type", "")
                    logger.debug("Content-Type: %s", content_type)

                    content_bytes = await response.read()

            is_text = (
                "text/plain" in content_type or "text/html" in content_type
            )

            if is_text:
                try:
                    content = content_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    logger.warning("UTF-8 decode failed, trying latin-1")
                    content = content_bytes.decode("latin-1")
            else:
                content = None

            has_javascript = (
                content and
                "<script" in content and
                "</script>" in content
            )

            if has_javascript:
                logger.debug("JavaScript detected, using Playwright")
                docs = await self._load_with_playwright(url)
            else:
                logger.debug("No JavaScript, using UnstructuredLoader")
                file_content = BytesIO(content_bytes)
                loader = UnstructuredLoader(file=file_content, metadata_filename=url.split("/")[-1])
                docs = loader.load()

            if docs and docs[0].page_content.strip():
                char_count = len(docs[0].page_content)
                logger.debug(
                    "Successfully loaded %d characters from %s",
                    char_count,
                    url
                )
                loaded_timestamp = int(time.time())
                for chunk_index, doc in enumerate(docs):
                    doc.metadata.setdefault("source", url)
                    doc.metadata["loaded_timestamp"] = loaded_timestamp
                    doc.metadata["chunk_index"] = chunk_index
                return docs
            else:
                logger.warning("Empty content from %s", url)
                return []

        except aiohttp.ClientError as e:
            logger.error("Network error loading %s: %s", url, e)
            return []
        except Exception as e:
            logger.error("Failed to load %s: %s", url, e)
            if not getattr(self, 'continue_on_failure', True):
                raise
            return []

    async def _load_with_playwright(self, url: str) -> list[Document]:
        html_content = None
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless)
            try:
                page = await browser.new_page()
                await page.goto(url, timeout=self.timeout * 1000)
                html_content = await page.content()
            finally:
                await browser.close()

        if not html_content:
            return []

        docs = [Document(page_content=html_content, metadata={"source": url})]
        transformer = Html2TextTransformer()
        return transformer.transform_documents(docs)

    async def _process_urls_loop(
        self,
        urls: list[str],
        max_concurrency: int,
    ) -> list[Document]:
        logger.info("Processing %d URLs (max_concurrency=%d)", len(urls), max_concurrency)
        semaphore = asyncio.Semaphore(max_concurrency)

        async def bounded(url: str) -> list[Document]:
            async with semaphore:
                return await self._process_url(url)

        results = await asyncio.gather(*(bounded(u) for u in urls), return_exceptions=True)

        all_docs: list[Document] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Error loading %s: %s", urls[i], result)
                continue
            all_docs.extend(result)

        logger.debug("Total documents loaded: %d", len(all_docs))
        return all_docs

    def load(self) -> list[Document]:
        """Load documents from URLs.

        Safe to call from both sync and async contexts. When an event loop is
        already running (e.g. inside LangGraph's async executor) the coroutine
        is dispatched to a ``ThreadPoolExecutor`` with its own fresh loop so
        that ``asyncio.run()`` never blocks a running loop.
        """
        logger.debug("URL Loader starting with %d URLs", len(self.urls))
        for i, url in enumerate(self.urls):
            logger.debug("  URL %d: %s", i + 1, url)

        try:
            asyncio.get_running_loop()
            import concurrent.futures
            logger.debug("Event loop running — dispatching to thread pool")
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                self.all_docs = pool.submit(
                    asyncio.run,
                    self._process_urls_loop(self.urls, self.max_concurrency),
                ).result()
        except RuntimeError:
            self.all_docs = asyncio.run(
                self._process_urls_loop(self.urls, self.max_concurrency)
            )
        except Exception as e:
            logger.error("Error in URL loader: %s: %s", type(e).__name__, e)
            logger.exception("Full traceback:")
            self.all_docs = []

        logger.info("URL loader completed: %d documents", len(self.all_docs))
        return self.all_docs

    async def aload(self) -> list[Document]:
        """Async version of :meth:`load` — use when already inside an event loop."""
        logger.debug("URL Loader (async) starting with %d URLs", len(self.urls))
        self.all_docs = await self._process_urls_loop(self.urls, self.max_concurrency)
        logger.info("URL loader (async) completed: %d documents", len(self.all_docs))
        return self.all_docs
