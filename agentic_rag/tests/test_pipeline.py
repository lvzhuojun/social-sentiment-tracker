"""
agentic_rag/tests/test_pipeline.py — Unit tests for pipeline.py.

All external dependencies (FAISS index, BertEmbedder, QueryRewriter,
SelfReflector) are mocked so tests run without GPU, disk index, or API keys.

Tests cover:
  - Successful single-attempt pipeline run
  - Pipeline iterates when reflection score is below threshold
  - Pipeline stops early when accepted=True
  - Best result is returned when max_attempts is exhausted without acceptance
  - Empty raw_query raises ValueError
  - PipelineResult contains correct fields
  - Iteration trace length matches actual attempt count
  - run_query() convenience wrapper delegates correctly
"""

import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

torch = pytest.importorskip("torch", reason="PyTorch not installed.")


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_DOCS = [{"clean_text": "Great product!", "label": 1, "score": 0.9}]
_REWRITTEN = "users express positive sentiment about the product"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_reflection(score: float, accepted: bool):
    from agentic_rag.self_reflection import ReflectionResult
    return ReflectionResult(score=score, rationale="test", accepted=accepted)


@contextmanager
def _patched_pipeline(
    rewrite_return: str = _REWRITTEN,
    docs_return: list | None = None,
    reflection_score: float = 0.9,
    accepted: bool = True,
    max_attempts: int = 3,
    top_k: int = 3,
):
    """Return a RAGPipeline with all external I/O mocked."""
    from agentic_rag.pipeline import RAGPipeline

    docs = docs_return or _DOCS
    reflection = _make_reflection(reflection_score, accepted)

    mock_embedder = MagicMock()
    mock_embedder.get_embedding.return_value = np.zeros(768, dtype=np.float32)

    mock_rewriter = MagicMock()
    mock_rewriter.rewrite.return_value = rewrite_return

    mock_reflector = MagicMock()
    mock_reflector.reflect.return_value = reflection

    mock_index = MagicMock()
    mock_id_map = [{"clean_text": "doc", "label": 1}]

    pipeline = RAGPipeline(max_attempts=max_attempts, top_k=top_k)
    pipeline._index = mock_index
    pipeline._id_map = mock_id_map
    pipeline._embedder = mock_embedder
    pipeline._rewriter = mock_rewriter
    pipeline._reflector = mock_reflector

    # search is lazily imported inside query() via
    # `from agentic_rag.build_index import search`, so patch it on the source module.
    with patch("agentic_rag.build_index.search", return_value=docs):
        yield pipeline


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline_ctx():
    with _patched_pipeline() as p:
        yield p


def test_query_returns_pipeline_result(pipeline_ctx):
    """query() returns a PipelineResult instance."""
    from agentic_rag.pipeline import PipelineResult
    result = pipeline_ctx.query("test query")
    assert isinstance(result, PipelineResult)


def test_query_original_query_preserved(pipeline_ctx):
    """original_query field matches the input."""
    result = pipeline_ctx.query("my original query")
    assert result.original_query == "my original query"


def test_query_final_query_is_rewritten(pipeline_ctx):
    """final_query is the rewritten form."""
    result = pipeline_ctx.query("raw query")
    assert result.final_query == _REWRITTEN


def test_query_documents_returned(pipeline_ctx):
    """documents list is non-empty."""
    result = pipeline_ctx.query("test query")
    assert len(result.documents) > 0


def test_query_accepted_true_on_high_score(pipeline_ctx):
    """accepted=True when reflection score >= threshold."""
    result = pipeline_ctx.query("test query")
    assert result.accepted is True


def test_query_single_iteration_on_accept():
    """Only one iteration occurs when first attempt is accepted."""
    with _patched_pipeline(accepted=True) as p:
        result = p.query("test query")
    assert len(result.iterations) == 1


def test_query_iterates_on_reject():
    """Pipeline makes multiple attempts when score is below threshold."""
    call_count = {"n": 0}

    def side_effect(query, context=None):
        call_count["n"] += 1
        return _REWRITTEN

    with _patched_pipeline(accepted=False, max_attempts=2) as p:
        p._rewriter.rewrite.side_effect = side_effect
        result = p.query("test query")

    assert len(result.iterations) == 2
    assert call_count["n"] == 2


def test_query_returns_best_on_exhaustion():
    """When max_attempts is reached without acceptance, best score wins."""
    scores = [0.4, 0.6, 0.5]
    call_index = {"i": 0}

    def reflect_side_effect(query, docs):
        score = scores[call_index["i"]]
        call_index["i"] += 1
        from agentic_rag.self_reflection import ReflectionResult
        return ReflectionResult(score=score, rationale="test", accepted=False)

    with _patched_pipeline(accepted=False, max_attempts=3) as p:
        p._reflector.reflect.side_effect = reflect_side_effect
        result = p.query("test query")

    assert abs(result.score - 0.6) < 1e-6


def test_query_empty_query_raises():
    """query() raises ValueError for empty input."""
    with _patched_pipeline() as p:
        with pytest.raises(ValueError, match="non-empty"):
            p.query("   ")


def test_iteration_trace_length_matches_attempts():
    """iterations list length equals the number of attempts made."""
    with _patched_pipeline(accepted=False, max_attempts=3) as p:
        result = p.query("test query")
    assert len(result.iterations) == 3


def test_run_query_convenience_function():
    """run_query() delegates to RAGPipeline.query()."""
    from agentic_rag.pipeline import PipelineResult

    dummy = PipelineResult(
        original_query="q", final_query="rq", documents=[], score=0.8, accepted=True
    )
    with patch("agentic_rag.pipeline.RAGPipeline") as mock_cls:
        instance = MagicMock()
        instance.query.return_value = dummy
        mock_cls.return_value = instance

        from agentic_rag.pipeline import run_query
        result = run_query("q", max_attempts=2, top_k=4)

        mock_cls.assert_called_once_with(max_attempts=2, top_k=4)
        instance.query.assert_called_once_with("q")
        assert result is dummy
