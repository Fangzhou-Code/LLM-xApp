from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.request
from typing import Optional

from .base import LLMClient


class GoogleClient(LLMClient):
    provider = "google"

    def __init__(self, *, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        raw_base_url = base_url or os.getenv("GOOGLE_BASE_URL")
        self.base_url = raw_base_url.rstrip("/") if raw_base_url else None

    @staticmethod
    def _ssl_context() -> ssl.SSLContext:
        ctx = ssl.create_default_context()

        def _truthy(v: Optional[str]) -> bool:
            return str(v or "").strip().lower() in {"1", "true", "yes", "on"}

        force_tls12 = _truthy(os.getenv("GOOGLE_TLS_FORCE_TLS12")) or _truthy(os.getenv("LLM_TLS_FORCE_TLS12"))
        if force_tls12 and hasattr(ssl, "TLSVersion"):
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            ctx.maximum_version = ssl.TLSVersion.TLSv1_2

        return ctx

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
            raise RuntimeError("GOOGLE_API_KEY is not set.")
        if not self.base_url:
            raise RuntimeError("GOOGLE_BASE_URL is not set.")
        # Use a chat completions-like endpoint; many Google endpoints use /v1/chat/completions
        url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a precise optimizer. Output JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
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
            with urllib.request.urlopen(req, timeout=timeout_s, context=self._ssl_context()) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except ssl.SSLError as e:
            backend = ssl.OPENSSL_VERSION
            hint = ""
            if "libressl" in backend.lower():
                hint = " On macOS, prefer a Python build with OpenSSL (e.g., Homebrew/python.org)."
            raise RuntimeError(
                "Google API request failed due to an SSL/TLS error. "
                f"Python SSL backend={backend}.{hint} Original error: {e}"
            ) from e
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Google API error {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            reason = getattr(e, "reason", None)
            if isinstance(reason, ssl.SSLError):
                backend = ssl.OPENSSL_VERSION
                hint = ""
                if "libressl" in backend.lower():
                    hint = " On macOS, prefer a Python build with OpenSSL (e.g., Homebrew/python.org)."
                raise RuntimeError(
                    "Google API request failed due to an SSL/TLS error. "
                    f"Python SSL backend={backend}.{hint} Original error: {reason}"
                ) from e
            raise RuntimeError(f"Google request failed: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Google request failed: {e}") from e

        try:
            return str(payload["choices"][0]["message"]["content"])
        except Exception:
            raise RuntimeError(f"Unexpected Google response format: {payload}")
