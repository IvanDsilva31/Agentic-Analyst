"""The model behind the harness: free Google Gemini, via LangChain.

``langchain-google-genai`` wraps the same free Gemini API you get a key for at
https://aistudio.google.com/app/apikey, but exposes it as a LangChain chat model
that supports tool calling — which is what the LangGraph harness needs.
"""

from __future__ import annotations

import os

from langchain_google_genai import ChatGoogleGenerativeAI

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


def build_llm(
    api_key: str | None = None,
    model: str | None = None,
    *,
    temperature: float = 0.1,
) -> ChatGoogleGenerativeAI:
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No Gemini API key found. Set GEMINI_API_KEY in your environment or "
            ".env file. Get a free key at https://aistudio.google.com/app/apikey"
        )
    return ChatGoogleGenerativeAI(
        model=model or DEFAULT_MODEL,
        google_api_key=api_key,
        temperature=temperature,
    )
