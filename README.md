# MovieRAG

Retrieval-Augmented Generation pipeline for movies. Semantic search over plots, metadata, Wikipedia articles, and timestamped subtitle dialogue.

## Pipeline Overview

```
movies.csv → [NB1] Data Collection → movies_corpus.json
                                          ↓
                                    [NB2] Chunk + Embed + Evaluate
                                          ↓
                                    [NB3] Search Pipeline (search.py)
```

### NB1 — Data Collection (`data_collection.py`)

Scrapes Wikipedia (metadata, plot, full article) and subdl.com (timestamped subtitles) for each movie in `movies.csv`. Outputs a structured JSON corpus.

- **Input**: `movies.csv` — swap this file to change the corpus without touching code
- **Output**: `movies_corpus.json`
- **Sources**: MediaWiki API, subdl.com REST API

```bash
pip install wikipedia-api requests beautifulsoup4 tqdm

# Full collection (wiki + subtitles)
python data_collection.py

# Custom movie list
python data_collection.py --movies my_list.csv

# Wiki only, no subtitles
python data_collection.py --no-subs
```

### NB2 — Embed & Evaluate (`eval_embed.py`)

Chunks the corpus using a recursive character splitter (sentence-aware, 750 chars, 50 overlap), embeds with multiple models via OpenRouter, stores in pgvector, and evaluates retrieval quality with synthetic QA.

**Chunking strategy:**
| Source | Method |
|---|---|
| Plot | RecursiveCharacterTextSplitter (750 chars, 50 overlap) |
| Article | RecursiveCharacterTextSplitter (750 chars, 50 overlap) |
| Metadata | Single structured chunk per movie (director, cast, year, etc.) |
| Subtitles | Time-windowed dialogue chunks (2-min windows) |

**QA generation:**
- 2 questions per movie (1 dialogue-based, 1 factual)
- Alternates between two LLMs for diversity (qwen/qwen3.5-9b, mistralai/mistral-small-creative)
- Source-aware prompts: subtitle chunks get dialogue-specific questions

**Evaluation metrics** (following MTEB methodology):
- **Hit Rate @k** — did the correct chunk appear anywhere in top-k?
- **MRR @k** — how high does the correct chunk rank?
- **nDCG @k** — overall ranking quality (MTEB standard)

**Our results (100 queries, @5):**

| Model | Hit Rate | MRR | nDCG | $/M tokens |
|---|---|---|---|---|
| **pplx_0_6b** | **0.8800** | **0.7023** | **0.7475** | $0.004 |
| qwen_4b | 0.8400 | 0.6703 | 0.7130 | $0.020 |
| openai_small | 0.8000 | 0.6250 | 0.6700 | $0.020 |

**Incremental processing** — re-runs only process new data:
- Chunks cached in `chunks_cache.json`
- Embeddings persisted in pgvector (skips existing IDs)
- QA pairs cached in `test_dataset.json`
- Results appended to `eval_results.json` with run history

```bash
# Prerequisites
docker run -d --name pgvector_movierag -e POSTGRES_PASSWORD=postgres \
    -p 5433:5432 pgvector/pgvector:pg16

pip install openai psycopg2-binary tqdm numpy

export OPENROUTER_API_KEY="sk-or-v1-..."

# Full run
python eval_embed.py --pg-dsn "postgresql://postgres:postgres@localhost:5433/postgres"

# Incremental (after adding movies to corpus)
python eval_embed.py --pg-dsn "postgresql://postgres:postgres@localhost:5433/postgres"

# Skip steps
python eval_embed.py --skip-qa          # reuse QA pairs
python eval_embed.py --skip-embed       # reuse embeddings
python eval_embed.py --full-rebuild     # ignore all caches
```

### NB3 — Search Pipeline (`search.py`)

Semantic search engine over the movie corpus. Takes a natural language query, embeds it, retrieves similar chunks from pgvector, and returns results with citations.

Uses pplx_0_6b by default based on our evaluation — swap via `--model` flag.

```bash
# Single query
python search.py -q "which movie has a talking teddy bear?"

# Interactive mode
python search.py

# Different model / more results
python search.py --model qwen_4b --top-k 10 -q "funny wedding scenes"
```

### Utilities

**`show_results.py`** — Inspect retrieval quality with side-by-side expected vs retrieved chunks.

```bash
python show_results.py                      # 5 random examples
python show_results.py --n 10              # more examples
python show_results.py --misses-only       # only failures
python show_results.py --model qwen_4b     # test different model
python show_results.py --all               # all queries
```

**`model_selection.py`** — Filters MTEB leaderboard for candidate embedding models (200-600M params, open-weight).

```bash
pip install mteb
python model_selection.py
```

## Swapping Components

| Component | How to swap |
|---|---|
| **Movies** | Edit `movies.csv`, re-run `data_collection.py` |
| **Embedding model** | Add entry to `EMBEDDING_MODELS` dict in `eval_embed.py`, re-run |
| **QA generation LLMs** | Edit `QA_LLMS` list in `eval_embed.py` |
| **Chunk size** | Change `CHUNK_SIZE` / `CHUNK_OVERLAP` in `eval_embed.py` |
| **Subtitle window** | Change `SUBTITLE_WINDOW_SEC` in `eval_embed.py` |
| **Vector DB** | Replace pgvector calls in `eval_embed.py` and `search.py` |
| **API provider** | Change `base_url` in OpenAI client (any OpenAI-compatible API works) |

## TODO

- [ ] **BM25 + RRF hybrid search** — combine keyword (BM25) and vector retrieval with Reciprocal Rank Fusion for better recall on exact names, quotes, and metadata queries
- [ ] **Embedding fine-tuning** — use train/test split (already in code) to fine-tune embedding models on movie domain data
- [ ] **LLM answer generation** — add an LLM layer on top of retrieval to synthesize natural language answers from retrieved chunks
- [ ] **Re-ranking** — add a cross-encoder re-ranker stage between retrieval and output
- [ ] **Evaluation on more genres** — extend beyond comedy to test generalization
- [ ] **Streaming / API wrapper** — wrap search.py in a FastAPI server for production use

## Project Structure

```
movierag/
├── movies.csv              # Input: movie list (title, wiki_title, year)
├── data_collection.py      # NB1: scrape Wikipedia + subtitles
├── movies_corpus.json      # Output of NB1
├── model_selection.py      # MTEB model filtering utility
├── eval_embed.py           # NB2: chunk, embed, evaluate
├── chunks_cache.json       # Cached chunks (incremental)
├── test_dataset.json       # Cached QA pairs
├── eval_results.json       # Evaluation results with run history
├── search.py               # NB3: semantic search pipeline
├── show_results.py         # Retrieval quality inspector
├── .env                    # API keys (not committed)
└── .gitignore
```
