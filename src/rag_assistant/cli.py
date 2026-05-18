from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import get_settings
from .pipeline import RAGPipeline

app = typer.Typer(
    add_completion=False,
    help="RAG-ассистент для работы с документацией (YandexGPT + Chroma).",
)
console = Console()


def _build_pipeline() -> RAGPipeline:
    return RAGPipeline(get_settings())


@app.command()
def ingest(
    path: Path = typer.Argument(..., exists=True, help="Файл или директория с документами"),
) -> None:
    """Загрузить документы в векторное хранилище."""
    pipeline = _build_pipeline()
    with console.status(f"[bold green]Индексирую {path}…"):
        n = pipeline.ingest_path(path)
    console.print(f"[green]Готово.[/green] Добавлено фрагментов: [bold]{n}[/bold]")
    console.print(f"Всего в коллекции: [bold]{pipeline.store.count()}[/bold]")


@app.command()
def ask(question: str = typer.Argument(..., help="Вопрос к коллекции")) -> None:
    """Задать вопрос по проиндексированным документам."""
    pipeline = _build_pipeline()
    with console.status("[bold green]Ищу контекст и формирую ответ…"):
        answer = pipeline.ask(question)
    console.print(Panel(answer.text, title="Ответ", border_style="green"))
    if not answer.sources:
        return
    table = Table(title="Источники", show_lines=False)
    table.add_column("Файл")
    table.add_column("Стр.", justify="right")
    table.add_column("Сходство", justify="right")
    for s in answer.sources:
        table.add_row(s.source, str(s.page), f"{s.score:.3f}")
    console.print(table)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Хост"),
    port: int = typer.Option(8000, help="Порт"),
) -> None:
    """Запустить FastAPI сервер."""
    import uvicorn

    uvicorn.run("rag_assistant.api:app", host=host, port=port, reload=False)


@app.command()
def stats() -> None:
    """Статистика по коллекции."""
    pipeline = _build_pipeline()
    s = pipeline.settings
    console.print(f"Путь к БД:   [bold]{s.vector_db_path}[/bold]")
    console.print(f"Коллекция:   [bold]{s.collection_name}[/bold]")
    console.print(f"Модель LLM:  [bold]{s.yandex_llm_model}[/bold]")
    console.print(f"Фрагментов:  [bold]{pipeline.store.count()}[/bold]")


@app.command()
def reset() -> None:
    """Очистить коллекцию (с подтверждением)."""
    if not typer.confirm("Очистить коллекцию полностью?", default=False):
        raise typer.Abort()
    pipeline = _build_pipeline()
    pipeline.store.reset()
    console.print("[yellow]Коллекция очищена.[/yellow]")


if __name__ == "__main__":
    app()
