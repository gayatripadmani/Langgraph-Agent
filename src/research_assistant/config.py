from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values, load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
CORPUS_DIR = DATA_DIR


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    chat_model: str = "models/gemini-2.5-flash"
    embedding_model: str = "models/gemini-embedding-001"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "research_assistant"
    max_critic_loops: int = 2
    retrieve_k: int = 4


def _clean_value(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip().strip('"').strip("'").strip()


def _load_env_map() -> dict[str, str]:
    file_values = dotenv_values(ROOT_DIR / ".env")
    normalized: dict[str, str] = {}
    for key, value in file_values.items():
        if key is None:
            continue
        normalized[key.strip().upper()] = _clean_value(value)
    for key, value in os.environ.items():
        normalized.setdefault(key.strip().upper(), _clean_value(value))
    return normalized


def load_settings() -> Settings:
    load_dotenv(ROOT_DIR / ".env")
    env_map = _load_env_map()
    api_key = env_map.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY in .env")

    qdrant_url = env_map.get("QDRANT_URL", "") or None
    qdrant_api_key = (
        env_map.get("QDRANT_API_KEY", "")
        or env_map.get("QDRANT_API_KEY ", "")
        or env_map.get("QDRANT_APIKEY", "")
        or None
    )
    if not qdrant_url:
        raise ValueError("Missing QDRANT_URL in .env")
    if not qdrant_api_key:
        raise ValueError("Missing QDRANT_API_KEY in .env")
    return Settings(
        gemini_api_key=api_key,
        chat_model=env_map.get("GEMINI_CHAT_MODEL", "models/gemini-2.5-flash-lite"),
        embedding_model=env_map.get(
            "GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"
        ),
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        qdrant_collection=env_map.get("QDRANT_COLLECTION", "research_assistant"),
    )
