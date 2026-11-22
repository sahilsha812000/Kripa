"""Load local environment settings for Kripa."""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Environment-backed configuration; secrets are optional until needed."""

    openai_api_key: Optional[str]
    anthropic_api_key: Optional[str]
    gemini_api_key: Optional[str]
    groq_api_key: Optional[str]
    openrouter_api_key: Optional[str]
    provider: str
    openai_model: str
    anthropic_model: str
    gemini_model: str
    groq_model: str
    openrouter_model: str
    ollama_base_url: str
    ollama_model: str


def get_settings() -> Settings:
    """Read API keys from the environment, loading a local .env file if present."""
    load_dotenv()
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        provider=os.getenv("KRIPA_PROVIDER", "gemini").strip().lower(),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-6.2-astra").strip(),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5").strip(),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip(),
        groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip(),
        openrouter_model=os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b").strip(),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip().rstrip("/"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2:3b").strip(),
    )
