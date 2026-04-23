"""
agentic_rag/pipeline.py — End-to-end Agentic RAG pipeline.

Orchestrates the full rewrite-retrieve-reflect loop:

  1. Rewrite  — QueryRewriter improves the raw query
  2. Embed    — BertEmbedder encodes the rewritten query
  3. Retrieve — FAISS search returns top-k candidates
  4. Reflect  — SelfReflector scores relevance
  5. Repeat   — if score < threshold and attempts remain, go to step 1
               with retrieved snippets as context for diversification

Returns a PipelineResult dataclass with the final documents, the accepted
query, the reflection score, and a trace of every iteration.

Key exports:
    PipelineResult  — frozen dataclass with query results and trace
    RAGPipeline     — main pipeline class
    run_query       — module-level convenience function
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import get_logger  # noqa: E402

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class IterationTrace:
    """Record of a single rewrite-retrieve-reflect iteration.

    Attributes:
        attempt: 1-based iteration index.
        rewritten_query: Query after rewriting.
        documents: Retrieved document dicts for this attempt.
        score: Reflection score in ``[0.0, 1.0]``.
        rationale: One-sentence explanation from the reflector.
        accepted: Whether the score met the threshold.
    """

    attempt: int
    rewritten_query: str
    documents: List[Dict]
    score: float
    rationale: str
    accepted: bool


@dataclass
class PipelineResult:
    """Final output of a RAGPipeline query.

    Attributes:
        original_query: The raw query as provided by the caller.
        final_query: The rewritten query that produced accepted results
                     (or the last rewritten query if max attempts exhausted).
        documents: Final set of retrieved documents.
        score: Reflection score of the final results.
        accepted: Whether the final results met the acceptance threshold.
        iterations: Full trace of every attempt made.
    """

    original_query: str
    final_query: str
    documents: List[Dict]
    score: float
    accepted: bool
    iterations: List[IterationTrace] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """Agentic RAG pipeline: rewrite → embed → retrieve → reflect → repeat.

    Lazy-loads the FAISS index on first call to ``query()`` so the pipeline
    can be constructed without blocking on disk I/O.

    Args:
        max_attempts: Maximum rewrite-retrieve-reflect iterations before
                      returning the best result found.  Defaults to
                      ``RAG_CONFIG.max_rewrite_attempts``.
        top_k: Number of documents to retrieve per attempt.  Defaults to
               ``RAG_CONFIG.top_k``.

    Example:
        >>> pipeline = RAGPipeline()
        >>> result = pipeline.query("users are angry about slow updates")
        >>> isinstance(result, PipelineResult)
        True
    """

    def __init__(
        self,
        max_attempts: int | None = None,
        top_k: int | None = None,
    ) -> None:
        from agentic_rag.config import RAG_CONFIG
        self._cfg = RAG_CONFIG
        self._max_attempts = (
            max_attempts if max_attempts is not None else RAG_CONFIG.max_rewrite_attempts
        )
        self._top_k = top_k if top_k is not None else RAG_CONFIG.top_k

        # Lazy-loaded on first query()
        self._index = None
        self._id_map: List[Dict] | None = None

        # Component instances (created once per pipeline)
        self._embedder = None
        self._rewriter = None
        self._reflector = None

        logger.info(
            "RAGPipeline ready — max_attempts=%d  top_k=%d",
            self._max_attempts, self._top_k,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, raw_query: str) -> PipelineResult:
        """Run the agentic RAG loop for a single user query.

        Args:
            raw_query: The user's natural-language question or query string.

        Returns:
            :class:`PipelineResult` with documents, final query, score, and
            iteration trace.

        Raises:
            ValueError: When ``raw_query`` is empty after stripping.
            FileNotFoundError: When the FAISS index has not been built yet.
                               Run ``python -m agentic_rag.build_index`` first.

        Example:
            >>> pipeline = RAGPipeline()
            >>> result = pipeline.query("negative reactions to the new policy")
            >>> len(result.documents) > 0
            True
        """
        raw_query = raw_query.strip()
        if not raw_query:
            raise ValueError("raw_query must be a non-empty string.")

        self._ensure_components_loaded()

        context: List[str] = []
        iterations: List[IterationTrace] = []
        best: IterationTrace | None = None

        for attempt in range(1, self._max_attempts + 1):
            logger.info("Pipeline attempt %d / %d", attempt, self._max_attempts)

            # 1. Rewrite
            rewritten = self._rewriter.rewrite(raw_query, context=context or None)

            # 2. Embed
            query_vec = self._embedder.get_embedding(rewritten)

            # 3. Retrieve
            from agentic_rag.build_index import search
            docs = search(query_vec, self._index, self._id_map, top_k=self._top_k)

            # 4. Reflect
            from agentic_rag.self_reflection import SelfReflector
            if self._reflector is None:
                self._reflector = SelfReflector()
            reflection = self._reflector.reflect(rewritten, docs)

            trace = IterationTrace(
                attempt=attempt,
                rewritten_query=rewritten,
                documents=docs,
                score=reflection.score,
                rationale=reflection.rationale,
                accepted=reflection.accepted,
            )
            iterations.append(trace)

            if best is None or reflection.score > best.score:
                best = trace

            if reflection.accepted:
                logger.info("Accepted on attempt %d (score=%.3f)", attempt, reflection.score)
                break

            # Prepare context for next rewrite (diversify away from these docs)
            context = [d.get("clean_text", "") for d in docs if d.get("clean_text")]

        # Use best attempt found (may not be accepted if max_attempts reached)
        assert best is not None
        return PipelineResult(
            original_query=raw_query,
            final_query=best.rewritten_query,
            documents=best.documents,
            score=best.score,
            accepted=best.accepted,
            iterations=iterations,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_components_loaded(self) -> None:
        """Load FAISS index, BertEmbedder, and QueryRewriter on first call."""
        if self._index is None:
            from agentic_rag.build_index import load_index
            self._index, self._id_map = load_index()

        if self._embedder is None:
            from agentic_rag.embedding_utils import BertEmbedder
            self._embedder = BertEmbedder()

        if self._rewriter is None:
            from agentic_rag.query_rewriter import QueryRewriter
            self._rewriter = QueryRewriter()


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def run_query(
    raw_query: str,
    max_attempts: int | None = None,
    top_k: int | None = None,
) -> PipelineResult:
    """Run the Agentic RAG pipeline for a single query.

    Convenience wrapper — creates a RAGPipeline and immediately calls
    ``query()``.  Prefer constructing a ``RAGPipeline`` instance directly
    when processing multiple queries to amortise model-loading cost.

    Args:
        raw_query: Natural-language user query.
        max_attempts: Override ``RAG_CONFIG.max_rewrite_attempts``.
        top_k: Override ``RAG_CONFIG.top_k``.

    Returns:
        :class:`PipelineResult` with the final documents and iteration trace.

    Example:
        >>> result = run_query("people dislike the app redesign")
        >>> isinstance(result, PipelineResult)
        True
    """
    return RAGPipeline(max_attempts=max_attempts, top_k=top_k).query(raw_query)
