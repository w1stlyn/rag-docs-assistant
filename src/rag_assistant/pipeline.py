from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .llm import YandexGPTClient
from .loaders import Document, load_directory, load_file
from .prompts import SYSTEM_PROMPT, USER_TEMPLATE, format_context
from .splitter import chunk_documents
from .vectorstore import VectorStore


@dataclass
class Source:
    source: str
    page: int | str | None
    score: float


@dataclass
class Answer:
    text: str
    sources: list[Source]


class RAGPipeline:
    def __init__(
        self,
        settings: Settings,
        llm: YandexGPTClient | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm or YandexGPTClient(settings)
        self.store = store or VectorStore(settings.vector_db_path, settings.collection_name)

    def ingest_path(self, path: Path) -> int:
        docs = load_directory(path) if path.is_dir() else load_file(path)
        return self.ingest_documents(docs)

    def ingest_documents(self, docs: list[Document]) -> int:
        chunks = chunk_documents(docs, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            return 0
        ids, texts, metas, embs = [], [], [], []
        for c in chunks:
            cid = hashlib.sha1(
                f"{c.source}|{c.metadata.get('chunk_index')}|{c.text[:64]}".encode("utf-8")
            ).hexdigest()
            ids.append(cid)
            texts.append(c.text)
            metas.append({"source": c.source, **c.metadata})
            embs.append(self.llm.embed(c.text, kind="doc"))
        self.store.add(ids=ids, texts=texts, embeddings=embs, metadatas=metas)
        return len(chunks)

    def ask(self, question: str) -> Answer:
        q_emb = self.llm.embed(question, kind="query")
        hits = self.store.query(q_emb, top_k=self.settings.top_k)
        if not hits:
            return Answer(
                text="В базе пока нет проиндексированных документов. Запустите `rag ingest <путь>`.",
                sources=[],
            )
        context = format_context(hits)
        prompt = USER_TEMPLATE.format(context=context, question=question)
        text = self.llm.complete(SYSTEM_PROMPT, prompt)
        sources = [
            Source(
                source=str(h["metadata"].get("source", "?")),
                page=h["metadata"].get("page"),
                score=round(1.0 - h["distance"], 4),
            )
            for h in hits
        ]
        return Answer(text=text, sources=sources)
