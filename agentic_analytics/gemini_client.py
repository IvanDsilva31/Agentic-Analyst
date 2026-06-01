"""Thin wrapper around the free Google Gemini API (google-genai SDK).

Get a free key at https://aistudio.google.com/app/apikey and set GEMINI_API_KEY.
"""

from __future__ import annotations

import json
import os
import re

from google import genai
from google.genai import types

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


class GeminiClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No Gemini API key found. Set GEMINI_API_KEY in your environment "
                "or .env file. Get a free key at https://aistudio.google.com/app/apikey"
            )
        self.client = genai.Client(api_key=api_key)
        self.model = model or DEFAULT_MODEL

    def generate(self, prompt: str, *, temperature: float = 0.1) -> str:
        """Return the model's plain-text response."""
        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=temperature),
        )
        return (resp.text or "").strip()

    def generate_json(self, prompt: str, *, temperature: float = 0.1) -> dict:
        """Return the model's response parsed as a JSON object."""
        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json",
            ),
        )
        return _loads_lenient(resp.text or "")


def _loads_lenient(text: str) -> dict:
    """Parse JSON, tolerating ```json fences or surrounding prose."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip markdown code fences if present.
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1).strip())
    # Fall back to the first {...} block.
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        return json.loads(brace.group(0))
    raise ValueError(f"Could not parse JSON from model response: {text[:200]!r}")
