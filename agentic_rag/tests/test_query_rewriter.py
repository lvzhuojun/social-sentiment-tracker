"""
agentic_rag/tests/test_query_rewriter.py — Unit tests for query_rewriter.py.

All LLM calls are mocked — these tests validate prompt construction,
return-value handling, and input guards without hitting the real API.

Tests cover:
  - rewrite() returns a non-empty string from the mocked API
  - Empty query raises ValueError
  - context snippets are included in the user message
  - temperature is forwarded to the API call
  - rewrite_query() convenience wrapper delegates correctly
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_response(content: str) -> MagicMock:
    """Build a minimal mock that mirrors openai ChatCompletion response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_rewriter(mock_create):
    """Return a QueryRewriter whose OpenAI client is fully mocked."""
    from agentic_rag.query_rewriter import QueryRewriter
    rewriter = QueryRewriter.__new__(QueryRewriter)
    rewriter._cfg = MagicMock()
    rewriter._temperature = 0.7
    rewriter._model = "gpt-5.4"
    rewriter._client = MagicMock()
    rewriter._client.chat.completions.create = mock_create
    return rewriter


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_rewrite_returns_string():
    """rewrite() returns the stripped string from the API response."""
    mock_create = MagicMock(return_value=_make_mock_response("  rewritten query  "))
    rewriter = _make_rewriter(mock_create)
    result = rewriter.rewrite("original query")
    assert result == "rewritten query"


def test_rewrite_calls_api_once():
    """rewrite() makes exactly one API call per invocation."""
    mock_create = MagicMock(return_value=_make_mock_response("result"))
    rewriter = _make_rewriter(mock_create)
    rewriter.rewrite("some query")
    assert mock_create.call_count == 1


def test_rewrite_empty_query_raises():
    """rewrite() raises ValueError for empty or whitespace-only input."""
    mock_create = MagicMock(return_value=_make_mock_response("anything"))
    rewriter = _make_rewriter(mock_create)
    with pytest.raises(ValueError, match="non-empty"):
        rewriter.rewrite("   ")


def test_rewrite_whitespace_stripped_before_check():
    """A query that is only whitespace is caught, not sent to the API."""
    mock_create = MagicMock(return_value=_make_mock_response("ok"))
    rewriter = _make_rewriter(mock_create)
    with pytest.raises(ValueError):
        rewriter.rewrite("\t\n")
    assert mock_create.call_count == 0


def test_rewrite_context_included_in_message():
    """When context is provided, snippets appear in the user message."""
    mock_create = MagicMock(return_value=_make_mock_response("rewritten"))
    rewriter = _make_rewriter(mock_create)
    rewriter.rewrite("my query", context=["doc A", "doc B"])

    call_messages = mock_create.call_args.kwargs.get("messages", [])
    user_msg = next(m["content"] for m in call_messages if m["role"] == "user")
    assert "doc A" in user_msg
    assert "doc B" in user_msg


def test_rewrite_no_context_no_snippet():
    """When context is None, the user message contains only the query."""
    mock_create = MagicMock(return_value=_make_mock_response("rewritten"))
    rewriter = _make_rewriter(mock_create)
    rewriter.rewrite("my query", context=None)

    call_messages = mock_create.call_args.kwargs.get("messages", [])
    user_msg = next(m["content"] for m in call_messages if m["role"] == "user")
    assert "Already retrieved" not in user_msg


def test_rewrite_temperature_forwarded():
    """The temperature passed at construction is forwarded to the API call."""
    mock_create = MagicMock(return_value=_make_mock_response("ok"))
    rewriter = _make_rewriter(mock_create)
    rewriter._temperature = 0.42
    rewriter.rewrite("test query")

    assert mock_create.call_args.kwargs.get("temperature") == 0.42


def test_rewrite_query_convenience_function():
    """rewrite_query() delegates to QueryRewriter.rewrite()."""
    with patch("agentic_rag.query_rewriter.QueryRewriter") as mock_cls:
        instance = MagicMock()
        instance.rewrite.return_value = "delegated result"
        mock_cls.return_value = instance

        from agentic_rag.query_rewriter import rewrite_query
        result = rewrite_query("hello", context=["ctx"], temperature=0.5)

        mock_cls.assert_called_once_with(temperature=0.5)
        instance.rewrite.assert_called_once_with("hello", context=["ctx"])
        assert result == "delegated result"
