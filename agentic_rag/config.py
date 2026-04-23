"""
agentic_rag/config.py — Configuration for the Agentic RAG subproject.

Loads secrets from a .env file (or from the environment) and exposes typed
constants for the OpenAI-compatible API, FAISS index paths, and retrieval
hyper-parameters.  All file paths use pathlib.Path.

Usage:
    from agentic_rag.config import RAG_CONFIG
    client = openai.OpenAI(api_key=RAG_CONFIG.openai_api_key,
                           base_url=RAG_CONFIG.base_url)
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import get_logger  # noqa: E402  (parent project config)

logger = get_logger(__name__)

# Load .env from the repo root if present (silently skip if absent)
try:
    from dotenv import load_dotenv
    _env_path = _REPO_ROOT / ".env"
    load_dotenv(dotenv_path=_env_path, override=False)
    if _env_path.exists():
        logger.info(".env loaded from %s", _env_path)
    else:
        logger.info(".env not found at %s — relying on shell environment", _env_path)
except ImportError:
    logger.warning("python-dotenv not installed; reading env vars from shell only.")


# ---------------------------------------------------------------------------
# Data class — single source of truth for all RAG settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _RagConfig:
    """Immutable configuration container for the Agentic RAG system.

    All values are read at import time.  File paths are resolved relative to
    the repo root so the module works regardless of the working directory.

    Attributes:
        openai_api_key: API key for the OpenAI-compatible endpoint.
                        Read from the ``OPENAI_API_KEY`` environment variable.
        base_url: Base URL of the LLM API endpoint.
        model: LLM model identifier sent in each chat-completion request.
        index_dir: Directory where FAISS index files are persisted.
        index_path: Path to the FAISS flat index file.
        id_map_path: Path to the pickled document-ID-to-text mapping.
        top_k: Number of nearest neighbours returned per query.
        rewrite_temperature: Sampling temperature for query rewriting.
        reflect_temperature: Sampling temperature for self-reflection scoring.
        max_rewrite_attempts: Maximum rewrite-and-reflect iterations.
    """

    openai_api_key: str
    base_url: str
    model: str

    index_dir: Path
    index_path: Path
    id_map_path: Path

    top_k: int
    rewrite_temperature: float
    reflect_temperature: float
    max_rewrite_attempts: int


def _build_config() -> _RagConfig:
    """Read environment variables and return a validated _RagConfig instance.

    Returns:
        Populated :class:`_RagConfig` instance.

    Raises:
        EnvironmentError: When ``OPENAI_API_KEY`` is not set or is empty.

    Example:
        >>> cfg = _build_config()
        >>> cfg.model
        'gpt-5.4'
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set.  "
            "Create a .env file at the repo root (copy .env.example) "
            "and set OPENAI_API_KEY to your real key."
        )

    index_dir = _REPO_ROOT / "agentic_rag" / "vector_store"

    return _RagConfig(
        openai_api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://apigate.solotice.codes/v1"),
        model=os.getenv("OPENAI_MODEL", "gpt-5.4"),
        index_dir=index_dir,
        index_path=index_dir / "sentiment.faiss",
        id_map_path=index_dir / "id_map.pkl",
        top_k=int(os.getenv("RAG_TOP_K", "5")),
        rewrite_temperature=float(os.getenv("RAG_REWRITE_TEMP", "0.7")),
        reflect_temperature=float(os.getenv("RAG_REFLECT_TEMP", "0.2")),
        max_rewrite_attempts=int(os.getenv("RAG_MAX_REWRITES", "3")),
    )


# Module-level singleton — evaluated once at import time.
# Tests that need a different key can monkeypatch os.environ before importing.
RAG_CONFIG: _RagConfig = _build_config()
