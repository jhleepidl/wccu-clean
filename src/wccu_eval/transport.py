"""Exact-payload replay with explicitly enabled, non-retrying HTTP collection.

Nothing contacts a provider unless execute=True and an endpoint are supplied.
Credentials are read from WCCU_API_KEY and never written to the response cache.
"""
from __future__ import annotations
from pathlib import Path
import os
import json
import urllib.error
import urllib.parse
import urllib.request
from .common import canonical, digest, load_json, save_json


class ResponseCache:
    def __init__(self, root: str | Path, model: str, seed: int, *, execute: bool = False,
                 endpoint: str | None = None, max_new_calls: int = 0,
                 request_options: dict | None = None):
        self.root, self.model, self.seed = Path(root), model, seed
        self.execute, self.endpoint = execute, endpoint
        self.max_new_calls, self.new_calls = max_new_calls, 0
        self.options = dict(request_options or {})
        if {"model", "messages", "seed", "max_completion_tokens"} & self.options.keys():
            raise ValueError("request_options must not override model/messages/seed/output limit")
        if execute:
            parsed = urllib.parse.urlparse(endpoint or "")
            if parsed.username or parsed.password:
                raise ValueError("Do not embed credentials in the endpoint URL")
            if not (parsed.scheme == "https" or (parsed.scheme == "http" and parsed.hostname in ("localhost", "127.0.0.1", "::1"))):
                raise ValueError("Use HTTPS, or a loopback HTTP endpoint")
            if max_new_calls <= 0:
                raise ValueError("Set an explicit positive new-call ceiling")
        self.records = {}

    def payload(self, messages: list, max_output: int) -> dict:
        return {"model": self.model, "messages": messages, "temperature": 0,
                "seed": self.seed, "max_completion_tokens": max_output,
                "store": False, "service_tier": "default", **self.options}

    def __call__(self, messages: list, max_output: int) -> tuple[str, str]:
        request = self.payload(messages, max_output)
        h = digest(canonical(request))
        if h not in self.records:
            path = self.root / (h + ".json")
            if path.exists():
                record = load_json(path)
                if record["request"] != request:
                    raise ValueError("Cache payload mismatch")
            elif not self.execute:
                raise FileNotFoundError(f"No cached response for {h}; network disabled")
            else:
                if (self.root / (h + ".failed.json")).exists():
                    raise RuntimeError("Prior failed request; no automatic retry")
                if self.new_calls >= self.max_new_calls:
                    raise RuntimeError("New-call ceiling reached")
                self.new_calls += 1
                save_json(self.root / (h + ".request.json"), request)
                headers = {"Content-Type": "application/json"}
                key = os.environ.get("WCCU_API_KEY", "")
                if key:
                    headers["Authorization"] = "Bearer " + key
                req = urllib.request.Request(self.endpoint, data=canonical(request).encode(), headers=headers, method="POST")
                try:
                    with urllib.request.urlopen(req, timeout=120) as response:
                        raw = json.load(response)
                except Exception as exc:
                    # Do not save exception text, request headers or provider error bodies.
                    save_json(self.root / (h + ".failed.json"), {"request_id": h, "type": type(exc).__name__, "automatic_retry": False})
                    raise RuntimeError(f"Request failed ({type(exc).__name__}); details redacted, no retry") from None
                record = {"request": request, "response": raw}
                save_json(path, record)  # Preserve evidence before parsing.
            response = record["response"]
            if response.get("model") != self.model:
                raise ValueError("Response model differs from pinned request model")
            if not response.get("choices"):
                raise ValueError("Response lacks a completion")
            usage = response.get("usage", {})
            if any(type(usage.get(k)) is not int or usage[k] < 0 for k in ("prompt_tokens", "completion_tokens")):
                raise ValueError("Response must include nonnegative integer token usage")
            self.records[h] = record
        text = self.records[h]["response"]["choices"][0]["message"].get("content") or ""
        if not isinstance(text, str):
            raise ValueError("Expected a text completion")
        return h, text

    def usage(self, ids: list[str]) -> dict:
        # Preserve repeated logical calls even when collection reused a payload.
        return {"calls": len(ids),
                "prompt_tokens": sum(self.records[h]["response"]["usage"]["prompt_tokens"] for h in ids),
                "output_tokens": sum(self.records[h]["response"]["usage"]["completion_tokens"] for h in ids)}
