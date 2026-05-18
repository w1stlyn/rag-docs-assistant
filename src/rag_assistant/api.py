from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .config import get_settings
from .loaders import SUPPORTED_SUFFIXES
from .pipeline import RAGPipeline

_pipeline: RAGPipeline | None = None


def _get_pipeline() -> RAGPipeline:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline не инициализирован")
    return _pipeline


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)


class SourceOut(BaseModel):
    source: str
    page: int | str | None = None
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]


class StatsResponse(BaseModel):
    collection: str
    chunks: int
    llm_model: str


class IngestResponse(BaseModel):
    chunks_added: int
    total_chunks: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    _pipeline = RAGPipeline(get_settings())
    yield
    _pipeline = None


app = FastAPI(
    title="RAG Docs Assistant",
    description="REST API ассистента для работы с документацией на базе YandexGPT и Chroma.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    p = _get_pipeline()
    return StatsResponse(
        collection=p.settings.collection_name,
        chunks=p.store.count(),
        llm_model=p.settings.yandex_llm_model,
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    p = _get_pipeline()
    a = p.ask(req.question)
    return AskResponse(
        answer=a.text,
        sources=[SourceOut(source=s.source, page=s.page, score=s.score) for s in a.sources],
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    p = _get_pipeline()
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(400, f"Неподдерживаемый формат: {suffix}")
    upload_dir = Path("./data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / (file.filename or f"upload{suffix}")
    dest.write_bytes(await file.read())
    n = p.ingest_path(dest)
    return IngestResponse(chunks_added=n, total_chunks=p.store.count())
