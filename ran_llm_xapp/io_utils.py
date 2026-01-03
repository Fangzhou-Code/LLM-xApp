from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_csv(path: str | Path, *, fieldnames: list[str], rows: list[Dict[str, Any]]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return p


def write_json(path: str | Path, data: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=False), encoding="utf-8")
    return p


@dataclass
class CacheHit:
    response_text: str
    cache_path: Path


class PromptResponseCache:
    """Disk cache for LLM prompt/response to avoid repeated cost."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = ensure_dir(cache_dir)

    def _key(
        self,
        *,
        provider: str,
        model: str,
        temperature: float,
        prompt: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        payload = {
            "provider": provider,
            "model": model,
            "temperature": float(temperature),
            "prompt": prompt,
            "extra": extra or {},
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return sha256_text(canonical)

    def get(
        self,
        *,
        provider: str,
        model: str,
        temperature: float,
        prompt: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[CacheHit]:
        key = self._key(provider=provider, model=model, temperature=temperature, prompt=prompt, extra=extra)
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            return CacheHit(response_text=str(obj.get("response_text", "")), cache_path=path)
        except Exception:
            return None

    def put(
        self,
        *,
        provider: str,
        model: str,
        temperature: float,
        prompt: str,
        response_text: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        key = self._key(provider=provider, model=model, temperature=temperature, prompt=prompt, extra=extra)
        path = self.cache_dir / f"{key}.json"
        record = {
            "provider": provider,
            "model": model,
            "temperature": float(temperature),
            "prompt": prompt,
            "response_text": response_text,
            "extra": extra or {},
            "created_at_unix_s": time.time(),
        }
        write_json(path, record)
        return path


def load_dotenv(dotenv_path: str | Path | None = None, *, override: bool = False) -> Optional[Path]:
    """Load environment variables from a `.env` file into `os.environ`.

    - Format: `KEY=VALUE` (optionally prefixed by `export `).
    - Ignores blank lines and comments starting with `#`.
    - Supports single/double quoted values.
    - By default, does NOT override existing environment variables.
    - If `dotenv_path` is None, it searches (in order):
        1) `<cwd>/.env`
        2) `<repo_root>/.env` (repo_root inferred from this file location)
    """

    candidates: list[Path] = []
    if dotenv_path is not None:
        candidates.append(Path(dotenv_path))
    else:
        candidates.append(Path.cwd() / ".env")
        candidates.append(Path(__file__).resolve().parents[1] / ".env")

    path = next((p for p in candidates if p.exists() and p.is_file()), None)
    if path is None:
        return None

    def _parse_value(raw: str) -> str:
        """Parse a dotenv value, supporting quotes and inline comments.

        Examples:
        - KEY=value
        - KEY=value # comment
        - KEY="value" # comment
        - KEY='value with # inside'
        """

        s = raw.strip()
        if not s:
            return ""

        if s[0] in {"'", '"'}:
            quote = s[0]
            out_chars: list[str] = []
            escaped = False
            for i in range(1, len(s)):
                ch = s[i]
                if escaped:
                    out_chars.append(ch)
                    escaped = False
                    continue
                if ch == "\\":
                    escaped = True
                    continue
                if ch == quote:
                    # Ignore anything after the closing quote (e.g., inline comments).
                    return "".join(out_chars)
                out_chars.append(ch)
            # Unclosed quote: fall back to best-effort unquoted parsing below.
            s = s[1:]

        # Unquoted value: strip inline comment if it starts after whitespace.
        for i, ch in enumerate(s):
            if ch == "#" and (i == 0 or s[i - 1].isspace()):
                s = s[:i].rstrip()
                break
        return s.strip()

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _parse_value(value)
        if not key:
            continue
        if (not override) and (key in os.environ):
            continue
        os.environ[key] = value

    return path
