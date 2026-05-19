from __future__ import annotations

import time
from typing import Literal

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import Settings

YANDEX_API = "https://llm.api.cloud.yandex.net/foundationModels/v1"

# Лимит Yandex Foundation Models — 10 запросов эмбеддингов в секунду.
# Держим ~8 RPS с запасом, чтобы не упереться при параллельной нагрузке.
_MIN_INTERVAL_SEC = 0.13


class YandexGPTError(RuntimeError):
    pass


class RateLimitError(YandexGPTError):
    """HTTP 429 — превышение rate-quota Yandex Cloud."""


_RETRYABLE = (
    httpx.HTTPStatusError,
    httpx.TransportError,
    httpx.TimeoutException,
    RateLimitError,
)


class YandexGPTClient:
    """Тонкий клиент Yandex Foundation Models API.

    Использует Api-Key для авторизации. Реализует два метода:
    complete() для генерации и embed() для эмбеддингов.
    Содержит простой троттлинг (один запрос в _MIN_INTERVAL_SEC) и
    retry с экспоненциальным backoff на сетевые ошибки и HTTP 429.
    """

    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self.settings = settings
        self._client = client or httpx.Client(timeout=60.0)
        self._headers = {
            "Authorization": f"Api-Key {settings.yandex_api_key}",
            "x-folder-id": settings.yandex_folder_id,
            "Content-Type": "application/json",
        }
        self._last_call_at: float = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call_at
        if elapsed < _MIN_INTERVAL_SEC:
            time.sleep(_MIN_INTERVAL_SEC - elapsed)
        self._last_call_at = time.monotonic()

    def _post(self, endpoint: str, body: dict, *, action: str) -> dict:
        self._throttle()
        resp = self._client.post(f"{YANDEX_API}/{endpoint}", headers=self._headers, json=body)
        if resp.status_code == 429:
            raise RateLimitError(f"{action} rate-limited (429): {resp.text}")
        if resp.status_code >= 400:
            raise YandexGPTError(f"{action} failed {resp.status_code}: {resp.text}")
        return resp.json()

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=16),
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
        data = self._post("completion", body, action="completion")
        try:
            return data["result"]["alternatives"][0]["message"]["text"]
        except (KeyError, IndexError) as e:
            raise YandexGPTError(f"unexpected response shape: {data!r}") from e

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=16),
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
        data = self._post("textEmbedding", body, action="embedding")
        try:
            return data["embedding"]
        except KeyError as e:
            raise YandexGPTError(f"unexpected response shape: {data!r}") from e

    def close(self) -> None:
        self._client.close()
