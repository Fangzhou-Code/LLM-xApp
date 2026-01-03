from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Optional

from .base import LLMClient


class DeepSeekClient(LLMClient):
    provider = "deepseek"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        raw_base_url = base_url or os.getenv("DEEPSEEK_BASE_URL")
        self.base_url = raw_base_url.rstrip("/") if raw_base_url else None

    def complete(
        self,
        *,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout_s: int,
        seed: Optional[int] = None,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set.")
        if not self.base_url:
            raise RuntimeError("DEEPSEEK_BASE_URL is not set.")
        url = f"{self.base_url}/chat/completions"
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a precise optimizer. Output JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            # DeepSeek is largely OpenAI-compatible; if unsupported, it may ignore this.
            "response_format": {"type": "json_object"},
        }
        if seed is not None:
            body["seed"] = int(seed)
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"DeepSeek API error {e.code}: {detail}") from e
        except Exception as e:
            raise RuntimeError(f"DeepSeek request failed: {e}") from e

        try:
            return str(payload["choices"][0]["message"]["content"])
        except Exception as e:
            raise RuntimeError(f"Unexpected DeepSeek response format: {payload}") from e
