from __future__ import annotations

from typing import Literal

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import Settings

YANDEX_API = "https://llm.api.cloud.yandex.net/foundationModels/v1"
_RETRYABLE = (httpx.HTTPStatusError, httpx.TransportError, httpx.TimeoutException)


class YandexGPTError(RuntimeError):
    pass


class YandexGPTClient:
    """Тонкий клиент Yandex Foundation Models API.

    Использует Api-Key для авторизации (см. документацию Yandex Cloud).
    Реализует два метода: complete() для генерации и embed() для эмбеддингов.
    """

    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self.settings = settings
        self._client = client or httpx.Client(timeout=60.0)
        self._headers = {
            "Authorization": f"Api-Key {settings.yandex_api_key}",
            "x-folder-id": settings.yandex_folder_id,
            "Content-Type": "application/json",
        }

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(_RETRYABLE),
    )
    def complete(self, system: str, user: str) -> str:
        body = {
            "modelUri": f"gpt://{self.settings.yandex_folder_id}/{self.settings.yandex_llm_model}",
            "completionOptions": {
                "stream": False,
                "temperature": self.settings.temperature,
                "maxTokens": str(self.settings.max_tokens),
            },
            "messages": [
                {"role": "system", "text": system},
                {"role": "user", "text": user},
            ],
        }
        resp = self._client.post(f"{YANDEX_API}/completion", headers=self._headers, json=body)
        if resp.status_code >= 400:
            raise YandexGPTError(f"completion failed {resp.status_code}: {resp.text}")
        data = resp.json()
        try:
            return data["result"]["alternatives"][0]["message"]["text"]
        except (KeyError, IndexError) as e:
            raise YandexGPTError(f"unexpected response shape: {data!r}") from e

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(_RETRYABLE),
    )
    def embed(self, text: str, *, kind: Literal["doc", "query"] = "doc") -> list[float]:
        model = (
            self.settings.yandex_embed_doc_model
            if kind == "doc"
            else self.settings.yandex_embed_query_model
        )
        body = {
            "modelUri": f"emb://{self.settings.yandex_folder_id}/{model}",
            "text": text,
        }
        resp = self._client.post(f"{YANDEX_API}/textEmbedding", headers=self._headers, json=body)
        if resp.status_code >= 400:
            raise YandexGPTError(f"embedding failed {resp.status_code}: {resp.text}")
        data = resp.json()
        try:
            return data["embedding"]
        except KeyError as e:
            raise YandexGPTError(f"unexpected response shape: {data!r}") from e

    def close(self) -> None:
        self._client.close()
