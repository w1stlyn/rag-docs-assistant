from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    yandex_api_key: str = Field(..., description="API-ключ сервисного аккаунта Yandex Cloud")
    yandex_folder_id: str = Field(..., description="Идентификатор каталога в Yandex Cloud")

    yandex_llm_model: str = "yandexgpt-lite/latest"
    yandex_embed_doc_model: str = "text-search-doc/latest"
    yandex_embed_query_model: str = "text-search-query/latest"

    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 4
    temperature: float = 0.2
    max_tokens: int = 2000

    vector_db_path: str = "./chroma_db"
    collection_name: str = "documents"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
