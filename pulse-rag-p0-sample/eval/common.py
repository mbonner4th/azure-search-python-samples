"""Shared bootstrap for eval scripts: env loading + project client."""

from __future__ import annotations

import os
from pathlib import Path

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv


_EVAL_ROOT = Path(__file__).resolve().parent


def load_env() -> None:
    """Load eval/.env if present; otherwise fall back to eval/sample.env so
    docs work out of the box."""
    dotenv_path = _EVAL_ROOT / ".env"
    if not dotenv_path.exists():
        dotenv_path = _EVAL_ROOT / "sample.env"
    load_dotenv(dotenv_path=dotenv_path, override=False)


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(
            f"Missing required environment variable: {name}. "
            f"Copy eval/sample.env to eval/.env and fill it in."
        )
    return value


def project_client(*, allow_preview: bool = False) -> AIProjectClient:
    """Build an ``AIProjectClient`` for the configured project endpoint.

    ``allow_preview=True`` is required to call
    ``get_openai_client(agent_name=...)`` for hosted Foundry agents.
    """
    load_env()
    endpoint = require_env("PROJECT_ENDPOINT")
    kwargs: dict = {}
    if allow_preview:
        kwargs["allow_preview"] = True
    return AIProjectClient(
        endpoint=endpoint, credential=DefaultAzureCredential(), **kwargs
    )


def eval_root() -> Path:
    return _EVAL_ROOT
