"""Minimal client for OpenAI-compatible chat completion APIs (UF NaviGator by default).

Configured through the environment so keys never touch the code or the repo:
    NAVIGATOR_API_KEY   required
    NAVIGATOR_BASE_URL  default https://api.navigator.ai.ufl.edu
    WARPSMITH_MODEL     default gpt-oss-120b
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import requests

DEFAULT_BASE_URL = "https://api.navigator.ai.ufl.edu"
DEFAULT_MODEL = "gpt-oss-120b"
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    reasoning: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    finish_reason: str | None = None
    raw: dict = field(default_factory=dict, repr=False)


class LLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        *,
        temperature: float = 1.0,
        max_tokens: int = 16000,
        reasoning_effort: str | None = None,
        timeout: float = 600.0,
        max_retries: int = 4,
    ):
        self.api_key = api_key or os.environ.get("NAVIGATOR_API_KEY")
        if not self.api_key:
            raise LLMError("NAVIGATOR_API_KEY is not set")
        base = (base_url or os.environ.get("NAVIGATOR_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        # Some OpenAI-compatible gateways serve /chat/completions, others only /v1/chat/completions.
        self.endpoints = [f"{base}/chat/completions"]
        if not base.endswith("/v1"):
            self.endpoints.append(f"{base}/v1/chat/completions")
        self.model = model or os.environ.get("WARPSMITH_MODEL") or DEFAULT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        self.max_retries = max_retries

    def complete(self, messages: list[dict]) -> LLMResponse:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        last_error = ""
        for attempt in range(self.max_retries + 1):
            start = time.perf_counter()
            try:
                resp = requests.post(self.endpoints[0], json=body, headers=headers, timeout=self.timeout)
            except requests.RequestException as e:
                last_error = f"{type(e).__name__}: {e}"
            else:
                if resp.status_code == 404 and len(self.endpoints) > 1:
                    self.endpoints.pop(0)
                    continue
                if resp.status_code == 200:
                    return self._parse(resp.json(), time.perf_counter() - start)
                last_error = f"HTTP {resp.status_code}: {resp.text[:500]}"
                if resp.status_code not in RETRY_STATUS:
                    break
            time.sleep(min(2 ** attempt * 5, 60))
        raise LLMError(last_error)

    @staticmethod
    def _parse(data: dict, seconds: float) -> LLMResponse:
        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"unexpected response shape: {str(data)[:500]}") from e
        usage = data.get("usage") or {}
        return LLMResponse(
            text=message.get("content") or "",
            reasoning=message.get("reasoning_content") or message.get("reasoning") or "",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            seconds=round(seconds, 1),
            finish_reason=choice.get("finish_reason"),
            raw=data,
        )
