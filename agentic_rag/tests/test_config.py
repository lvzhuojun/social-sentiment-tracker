"""
agentic_rag/tests/test_config.py — Unit tests for agentic_rag/config.py.

Tests cover:
  - _RagConfig builds successfully when OPENAI_API_KEY is set
  - Correct defaults for all numeric and string fields
  - EnvironmentError raised when OPENAI_API_KEY is missing
  - index_path and id_map_path are inside index_dir
  - Overrides via environment variables work correctly
"""

import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build(monkeypatch, extra: dict | None = None):
    """Return a fresh _RagConfig with OPENAI_API_KEY set to 'test-key'."""
    # Isolate from any real .env on disk by patching load_dotenv to no-op
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    if extra:
        for k, v in extra.items():
            monkeypatch.setenv(k, v)
    # Force re-evaluation by calling _build_config directly
    from agentic_rag.config import _build_config
    return _build_config()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_build_succeeds_with_key(monkeypatch):
    """_build_config() returns a config when OPENAI_API_KEY is present."""
    cfg = _build(monkeypatch)
    assert cfg.openai_api_key == "test-key"


def test_default_model(monkeypatch):
    """Default model is 'gpt-5.4'."""
    cfg = _build(monkeypatch)
    assert cfg.model == "gpt-5.4"


def test_default_base_url(monkeypatch):
    """Default base_url points to the configured gateway."""
    cfg = _build(monkeypatch)
    assert cfg.base_url == "https://apigate.solotice.codes/v1"


def test_default_top_k(monkeypatch):
    """Default top_k is 5."""
    cfg = _build(monkeypatch)
    assert cfg.top_k == 5


def test_default_max_rewrite_attempts(monkeypatch):
    """Default max_rewrite_attempts is 3."""
    cfg = _build(monkeypatch)
    assert cfg.max_rewrite_attempts == 3


def test_index_paths_inside_index_dir(monkeypatch):
    """index_path and id_map_path are children of index_dir."""
    cfg = _build(monkeypatch)
    assert cfg.index_path.parent == cfg.index_dir
    assert cfg.id_map_path.parent == cfg.index_dir


def test_missing_api_key_raises(monkeypatch):
    """_build_config() raises EnvironmentError when OPENAI_API_KEY is absent."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # Also clear any value that might have been loaded from .env
    os.environ.pop("OPENAI_API_KEY", None)
    from agentic_rag.config import _build_config
    with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
        _build_config()


def test_env_override_top_k(monkeypatch):
    """RAG_TOP_K env var overrides the default top_k."""
    cfg = _build(monkeypatch, {"RAG_TOP_K": "10"})
    assert cfg.top_k == 10


def test_env_override_model(monkeypatch):
    """OPENAI_MODEL env var overrides the default model name."""
    cfg = _build(monkeypatch, {"OPENAI_MODEL": "gpt-4o"})
    assert cfg.model == "gpt-4o"


def test_config_is_frozen(monkeypatch):
    """_RagConfig instances are immutable (frozen dataclass)."""
    cfg = _build(monkeypatch)
    with pytest.raises(Exception):
        cfg.top_k = 99  # type: ignore[misc]
