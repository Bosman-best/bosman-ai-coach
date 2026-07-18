"""
Minimal Ollama client. Talks to the local Ollama server over its REST API.
No other module should import `requests` directly for this purpose —
if we ever swap Ollama for something else, this is the only file that changes.
"""

from __future__ import annotations
import json
import requests

from core.schemas import AppConfig


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, config: AppConfig):
        self.config = config

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        """Send a prompt and force a JSON-shaped response back.

        Uses Ollama's `format: "json"` mode so the model is constrained
        to emit valid JSON, then parses it. Raises OllamaError on any
        network failure or invalid JSON so callers can handle it explicitly.
        """
        url = f"{self.config.ollama_host}/api/chat"
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "format": "json",
            "stream": False,
            "options": {
                # Low temperature: tactical advice should be consistent for the
                # same situation, not wander between different philosophies
                # (e.g. high-press vs. counter-attack) on repeat runs.
                "temperature": 0.15,
            },
        }

        try:
            resp = requests.post(url, json=payload, timeout=self.config.request_timeout_s)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise OllamaError(
                f"Could not reach Ollama at {self.config.ollama_host}. "
                f"Is `ollama serve` running? ({e})"
            ) from e

        data = resp.json()
        content = data.get("message", {}).get("content", "")

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise OllamaError(
                f"Model did not return valid JSON. Raw output:\n{content}"
            ) from e

    def health_check(self) -> bool:
        """Quick check that the Ollama server is reachable at all."""
        try:
            resp = requests.get(f"{self.config.ollama_host}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False
