# Agentic RAG — Sentiment-Aware Retrieval System

Built on top of the Social Sentiment Tracker project, this submodule implements
an **Agentic Retrieval-Augmented Generation** pipeline that:

1. **Rewrites** raw user queries into retrieval-optimised forms via an LLM
2. **Embeds** the rewritten query using the fine-tuned BERT model (reused, unmodified)
3. **Retrieves** the most semantically similar documents from a FAISS index
4. **Reflects** on result quality with an LLM-based scorer
5. **Repeats** if the score is below threshold, providing prior context to diversify

---

## Architecture

```
raw_query
    │
    ▼
QueryRewriter  ──── OpenAI-compatible LLM (gpt-5.4 via gateway)
    │ rewritten_query
    ▼
BertEmbedder   ──── Fine-tuned bert-base-uncased ([CLS] token, 768-dim)
    │ query_vector
    ▼
FAISS Index    ──── IndexFlatIP (cosine similarity via L2-normalised vectors)
    │ top-k documents
    ▼
SelfReflector  ──── LLM scores relevance → JSON {score, rationale}
    │
    ├── score ≥ 0.7 → accept → return PipelineResult
    └── score < 0.7 → rewrite again with context → loop (max 3 attempts)
```

---

## Directory Structure

```
agentic_rag/
├── config.py           # API keys, paths, hyper-parameters (reads .env)
├── embedding_utils.py  # BertEmbedder — [CLS] extraction from bert_sentiment.pt
├── build_index.py      # build_index(), load_index(), search() — FAISS CRUD
├── query_rewriter.py   # QueryRewriter — LLM query rewriting
├── self_reflection.py  # SelfReflector — LLM relevance scoring
├── pipeline.py         # RAGPipeline — orchestrates the full loop
├── tests/
│   ├── conftest.py     # session-level dummy API key for CI
│   ├── test_config.py
│   ├── test_embedding_utils.py
│   ├── test_build_index.py
│   ├── test_query_rewriter.py
│   ├── test_self_reflection.py
│   └── test_pipeline.py
└── vector_store/       # git-ignored — FAISS index + id_map written here
```

---

## Quick Start

### 1. Prerequisites

```bash
# Activate the shared conda environment
# (Python 3.10, PyTorch 2.11+cu128, all deps already installed)
conda activate sentiment-tracker

# Create your .env file (copy template, add real key)
cp .env.example .env
# Edit .env: set OPENAI_API_KEY=<your-key>
```

### 2. Build the FAISS Index

Run once to encode the training data and persist the index to `vector_store/`:

```bash
python -m agentic_rag.build_index
# Optional: specify a different CSV or batch size
python -m agentic_rag.build_index --csv data/processed/train.csv --batch 64
```

### 3. Run a Query

```python
from agentic_rag.pipeline import run_query

result = run_query("users are frustrated with the slow app updates")

print(f"Final query:  {result.final_query}")
print(f"Score:        {result.score:.3f}  (accepted: {result.accepted})")
print(f"Iterations:   {len(result.iterations)}")
print()
for doc in result.documents[:3]:
    print(f"  [{doc['label']}] {doc['clean_text']}")
```

---

## Configuration

All settings are read at import time from environment variables (loaded from `.env`):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | **required** | API key for the LLM endpoint |
| `OPENAI_BASE_URL` | `https://apigate.solotice.codes/v1` | Gateway base URL |
| `OPENAI_MODEL` | `gpt-5.4` | LLM model identifier |
| `RAG_TOP_K` | `5` | Documents retrieved per attempt |
| `RAG_REWRITE_TEMP` | `0.7` | Query rewriting temperature |
| `RAG_REFLECT_TEMP` | `0.2` | Reflection scoring temperature |
| `RAG_MAX_REWRITES` | `3` | Maximum rewrite-retrieve-reflect iterations |

---

## Running Tests

```bash
# From the social-sentiment-tracker root:
pytest agentic_rag/tests/ -v

# Current result: 59 / 59 passed
# LLM and FAISS calls are fully mocked — no API key or GPU required for tests.
```

---

## Module API Reference

### `embedding_utils.BertEmbedder`

```python
embedder = BertEmbedder(batch_size=32)
vec: np.ndarray = embedder.get_embedding("text")      # shape (768,)
mat: np.ndarray = embedder.encode_batch(["a", "b"])   # shape (n, 768)
```

### `build_index`

```python
from agentic_rag.build_index import build_index, load_index, search

index, id_map = build_index()            # encode + persist
index, id_map = load_index()             # load from disk
results = search(query_vec, index, id_map, top_k=5)
# results: List[Dict] — each dict has clean_text, label, score
```

### `query_rewriter.QueryRewriter`

```python
rewriter = QueryRewriter()
rewritten: str = rewriter.rewrite("raw query", context=["prior doc 1"])
```

### `self_reflection.SelfReflector`

```python
from agentic_rag.self_reflection import SelfReflector, ReflectionResult

reflector = SelfReflector(threshold=0.7)
result: ReflectionResult = reflector.reflect("query", documents)
# result.score, result.rationale, result.accepted
```

### `pipeline.RAGPipeline`

```python
from agentic_rag.pipeline import RAGPipeline, PipelineResult

pipeline = RAGPipeline(max_attempts=3, top_k=5)
result: PipelineResult = pipeline.query("what do people think about X?")
# result.original_query, .final_query, .documents, .score, .accepted, .iterations
```

---

## Design Decisions

| Decision | Reason |
|---|---|
| `faiss-cpu` instead of `faiss-gpu` | CUDA 13.1 compatibility — `faiss-gpu` wheels not yet available |
| `IndexFlatIP` with L2-normalised vectors | Exact cosine similarity; no approximate-search overhead for dataset sizes < 100k |
| BERT [CLS] token (not mean pooling) | Consistent with how bert-base-uncased was fine-tuned in this project |
| Lazy index loading in `RAGPipeline` | Avoids blocking disk I/O at construction time; safe for import in tests |
| `frozen=True` on `_RagConfig` | Prevents accidental mutation of shared configuration singleton |
| JSON-structured reflection response | Structured output is easier to parse and validate than free-text scoring |
