"""
agentic_rag/query_rewriter.py — LLM-powered query rewriting for Agentic RAG.

Rewrites a raw user query into a more retrieval-friendly form using an
OpenAI-compatible chat-completion endpoint.  The rewriter is intentionally
stateless: it takes a query string (and optionally prior context) and returns
a single improved query string.

Key exports:
    QueryRewriter  — class wrapping the OpenAI client for query rewriting
    rewrite_query  — module-level convenience function
"""

import sys
from pathlib import Path
from typing import List

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import get_logger  # noqa: E402

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a search-query optimisation assistant.  "
    "Your task is to rewrite the user's query so it retrieves the most relevant "
    "social-media sentiment examples from a vector database.  "
    "Rules:\n"
    "1. Keep the rewritten query to one concise sentence (≤ 25 words).\n"
    "2. Preserve the original intent and sentiment direction.\n"
    "3. Replace vague pronouns with explicit subject nouns.\n"
    "4. Remove filler words and punctuation that add no semantic value.\n"
    "5. Return ONLY the rewritten query — no explanation, no quotes."
)


class QueryRewriter:
    """Rewrite a user query into a retrieval-optimised form via an LLM.

    Uses the OpenAI-compatible endpoint configured in ``agentic_rag.config``.
    Each call is stateless — prior conversation history is not retained.

    Args:
        temperature: Sampling temperature for query rewriting.  Defaults to
                     ``RAG_CONFIG.rewrite_temperature``.

    Example:
        >>> rewriter = QueryRewriter()
        >>> result = rewriter.rewrite("how do ppl feel abt the new iphone lol")
        >>> isinstance(result, str) and len(result) > 0
        True
    """

    def __init__(self, temperature: float | None = None) -> None:
        from openai import OpenAI
        from agentic_rag.config import RAG_CONFIG

        self._cfg = RAG_CONFIG
        self._temperature = (
            temperature if temperature is not None
            else RAG_CONFIG.rewrite_temperature
        )
        self._client = OpenAI(
            api_key=RAG_CONFIG.openai_api_key,
            base_url=RAG_CONFIG.base_url,
        )
        self._model = RAG_CONFIG.model
        logger.info(
            "QueryRewriter ready — model=%s  temperature=%.2f",
            self._model, self._temperature,
        )

    def rewrite(
        self,
        query: str,
        context: List[str] | None = None,
    ) -> str:
        """Rewrite ``query`` into a retrieval-optimised form.

        Args:
            query: The raw user query string.
            context: Optional list of prior retrieved document snippets.
                     When provided, the rewriter is informed of what has
                     already been found so it can diversify the next query.


        Returns:
            Rewritten query string (single sentence, ≤ 25 words).

        Raises:
            ValueError: When ``query`` is empty after stripping whitespace.
            openai.OpenAIError: When the API call fails.

        Example:
            >>> rewriter = QueryRewriter()
            >>> q = rewriter.rewrite("negative feelings about airline delays")
            >>> isinstance(q, str)
            True
        """
        query = query.strip()
        if not query:
            raise ValueError("query must be a non-empty string.")

        user_content = f"Original query: {query}"
        if context:
            snippet = " | ".join(context[:3])
            user_content += (
                f"\n\nAlready retrieved context (diversify away from these):\n{snippet}"
            )

        response = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        rewritten = response.choices[0].message.content.strip()
        logger.info("Rewritten query: %r → %r", query, rewritten)
        return rewritten


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def rewrite_query(
    query: str,
    context: List[str] | None = None,
    temperature: float | None = None,
) -> str:
    """Rewrite a query using a fresh QueryRewriter instance.

    Convenience wrapper — creates a QueryRewriter and immediately calls
    ``rewrite()``.  Prefer constructing a ``QueryRewriter`` instance directly
    when making multiple calls to amortise client initialisation cost.

    Args:
        query: Raw user query string.
        context: Optional list of already-retrieved document snippets.
        temperature: Sampling temperature override.

    Returns:
        Rewritten query string.

    Example:
        >>> q = rewrite_query("people hate the new update")
        >>> isinstance(q, str)
        True
    """
    return QueryRewriter(temperature=temperature).rewrite(query, context=context)
