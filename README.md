# rag-docs-assistant

RAG-ассистент для работы с произвольной коллекцией документов на базе **YandexGPT** и **Chroma**.
Загружает PDF / DOCX / Markdown / TXT, индексирует семантически и отвечает на вопросы со ссылками
на конкретные страницы исходных файлов.

Доступен как **CLI** (Typer + Rich) и как **REST API** (FastAPI).

## Стек

- **LLM и эмбеддинги:** YandexGPT (`yandexgpt-lite`, `text-search-doc`, `text-search-query`)
- **Векторная БД:** Chroma (persistent, HNSW, cosine)
- **API:** FastAPI + Uvicorn
- **CLI:** Typer + Rich
- **Конфигурация:** pydantic-settings + `.env`
- **HTTP-клиент:** httpx + tenacity (retry с экспоненциальным backoff)
- **Тесты:** pytest

## Архитектура

```
              ┌──────────┐  ingest   ┌─────────┐  embed   ┌────────────┐
PDF/DOCX/MD ─▶│ Loaders  │──────────▶│ Splitter │─────────▶│ YandexGPT  │
              └──────────┘           └─────────┘          │ Embeddings │
                                                          └─────┬──────┘
                                                                ▼
                                                          ┌────────────┐
                                                          │   Chroma   │
                                                          └─────┬──────┘
                                                                │ top-k
              question ──┐                                      ▼
                         │                              ┌───────────────┐
                         └─────────────────────────────▶│ Prompt + LLM  │──▶ answer + sources
                                                        └───────────────┘
```

## Быстрый старт

### 1. Получить ключи Yandex Cloud

1. Зайдите в [Yandex Cloud Console](https://console.cloud.yandex.ru/).
2. Создайте сервисный аккаунт с ролью `ai.languageModels.user`.
3. Получите API-ключ и `folder_id`.

### 2. Установка

```bash
git clone <repo-url>
cd rag-docs-assistant

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .

cp .env.example .env
# открыть .env и подставить YANDEX_API_KEY и YANDEX_FOLDER_ID
```

### 3. CLI

```bash
# Загрузить документы в индекс
rag ingest ./examples/sample.md
rag ingest ./data/my_textbook.pdf
rag ingest ./data/lectures/        # рекурсивно

# Задать вопрос
rag ask "Что такое RAG и зачем нужен retriever?"

# Посмотреть состояние индекса
rag stats

# Поднять REST API
rag serve --host 0.0.0.0 --port 8000

# Очистить коллекцию
rag reset
```

### 4. REST API

После `rag serve` интерактивная документация Swagger UI доступна по адресу
`http://localhost:8000/docs`.

```bash
# Загрузить файл
curl -X POST http://localhost:8000/ingest \
  -F "file=@./examples/sample.md"

# Спросить
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Какие метрики используются для оценки RAG?"}'

# Статистика
curl http://localhost:8000/stats
```

Пример ответа `/ask`:

```json
{
  "answer": "Для оценки RAG-систем применяются метрики Context Precision, Context Recall, Faithfulness и Answer Relevancy …\n\n[Источник: sample.md, стр. 1]",
  "sources": [
    { "source": "sample.md", "page": 1, "score": 0.842 }
  ]
}
```

## Конфигурация

Все параметры читаются из `.env` через `pydantic-settings`. Полный список — в [`.env.example`](.env.example).

| Переменная | Назначение | По умолчанию |
| --- | --- | --- |
| `YANDEX_API_KEY` | API-ключ сервисного аккаунта | — *(обязательно)* |
| `YANDEX_FOLDER_ID` | ID каталога Yandex Cloud | — *(обязательно)* |
| `YANDEX_LLM_MODEL` | Модель генерации | `yandexgpt-lite/latest` |
| `CHUNK_SIZE` | Размер фрагмента (символы) | `800` |
| `CHUNK_OVERLAP` | Перекрытие фрагментов | `100` |
| `TOP_K` | Количество извлекаемых фрагментов | `4` |
| `TEMPERATURE` | Температура LLM | `0.2` |
| `VECTOR_DB_PATH` | Путь к Chroma | `./chroma_db` |

## Тесты

```bash
pip install pytest
pytest
```

## Структура проекта

```
rag-docs-assistant/
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── README.md
├── examples/
│   └── sample.md
├── src/rag_assistant/
│   ├── __init__.py
│   ├── config.py        # pydantic-settings + .env
│   ├── llm.py           # YandexGPT клиент (completion, embedding)
│   ├── loaders.py       # PDF / DOCX / MD / TXT
│   ├── splitter.py      # рекурсивный чанкинг
│   ├── vectorstore.py   # Chroma persistent
│   ├── prompts.py       # шаблоны промптов
│   ├── pipeline.py      # RAG-конвейер
│   ├── cli.py           # Typer CLI
│   └── api.py           # FastAPI приложение
└── tests/
    └── test_splitter.py
```
