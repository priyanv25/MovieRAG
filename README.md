# MovieRAG

Semantic search engine for movies. Search using natural language across plots, metadata (cast, director, year), Wikipedia articles, and timestamped subtitle dialogue.

## Quick Start

```bash
# Prerequisites
docker start pgvector_movierag   # or create: see Setup below
pip install openai psycopg2-binary tqdm numpy
export OPENROUTER_API_KEY="sk-or-v1-..."

# Search
python search.py -q "which movie has a talking teddy bear?"
python search.py                    # interactive mode
```

## How It Works

```
data/movies.csv
      |
      v
[1. Collect Data]  ──>  data/movies_corpus.json
      |                  (wiki + subtitles per movie)
      v
[2. Chunk + Embed + Evaluate]  ──>  pgvector DB + data/eval_results.json
      |                              (embeddings + retrieval metrics)
      v
[3. Search]  ──>  query → embed → cosine similarity → ranked results + citations
```

**Chunking strategy:**
| Source | Method |
|---|---|
| Plot | Recursive character split (750 chars, 50 overlap) |
| Article | Recursive character split (750 chars, 50 overlap) |
| Metadata | Single structured chunk per movie (director, cast, year, etc.) |
| Subtitles | Time-windowed dialogue (2-min windows) |

**Embedding model selection** — evaluated 3 models on 100 synthetic QA pairs (2 per movie, alternating LLMs for diversity):

| Model | Hit Rate @5 | MRR | nDCG | $/M tokens |
|---|---|---|---|---|
| **pplx_0_6b** | **0.8800** | **0.7023** | **0.7475** | $0.004 |
| qwen_4b | 0.8400 | 0.6703 | 0.7130 | $0.020 |
| openai_small | 0.8000 | 0.6250 | 0.6700 | $0.020 |

We use pplx_0_6b — best metrics, cheapest. Swap via `--model` flag.

## Setup

```bash
# 1. pgvector (vector database)
docker run -d --name pgvector_movierag -e POSTGRES_PASSWORD=postgres \
    -p 5433:5432 pgvector/pgvector:pg16

# 2. Python deps
pip install openai psycopg2-binary tqdm numpy wikipedia-api requests beautifulsoup4

# 3. API key
export OPENROUTER_API_KEY="sk-or-v1-..."

# 4. Build the pipeline (one-time)
python setup/collect_data.py                    # scrape data
python setup/embed_eval.py                      # chunk + embed + evaluate
```

## Usage

### Search (`search.py`)

The main pipeline. Takes a query, embeds it, retrieves similar chunks from pgvector, returns results with citations.

```bash
python search.py -q "who is Ben's love interest in The Intern?"
python search.py -q "funny wedding scenes" --top-k 10
python search.py --model qwen_4b -q "bachelor party movie"
python search.py                                # interactive mode
```

### Inspect Results (`show_results.py`)

Side-by-side view of expected vs retrieved chunks for QA pairs.

```bash
python show_results.py                          # 5 random examples
python show_results.py --n 10                   # more examples
python show_results.py --misses-only            # only failures
python show_results.py --hits-only              # only successes
python show_results.py --model qwen_4b          # different model
python show_results.py --all                    # all 100 queries
```

### Data Collection (`setup/collect_data.py`)

Scrapes Wikipedia (metadata, plot, full article) and subdl.com (timestamped subtitles).

```bash
python setup/collect_data.py                             # full run
python setup/collect_data.py --movies data/my_list.csv   # custom movie list
python setup/collect_data.py --no-subs                   # wiki only
```

### Embed & Evaluate (`setup/embed_eval.py`)

Chunks corpus, embeds with multiple models, evaluates retrieval quality. Incremental — only processes new movies on re-run.

```bash
python setup/embed_eval.py                      # full run (or incremental)
python setup/embed_eval.py --skip-qa            # reuse QA pairs
python setup/embed_eval.py --skip-embed         # reuse embeddings
python setup/embed_eval.py --full-rebuild       # ignore all caches
```

### Model Selection (`setup/select_model.py`)

Filters the MTEB leaderboard for candidate embedding models.

```bash
pip install mteb
python setup/select_model.py
```

## Swapping Components

| What | How |
|---|---|
| **Movies** | Edit `data/movies.csv`, re-run `setup/collect_data.py` then `setup/embed_eval.py` |
| **Embedding model** | Add to `EMBEDDING_MODELS` in `setup/embed_eval.py`, re-run, then use `--model` in `search.py` |
| **QA generation LLMs** | Edit `QA_LLMS` list in `setup/embed_eval.py` |
| **Chunk size / overlap** | Change `CHUNK_SIZE` / `CHUNK_OVERLAP` in `setup/embed_eval.py` |
| **Subtitle window** | Change `SUBTITLE_WINDOW_SEC` in `setup/embed_eval.py` |
| **Vector DB** | Replace pgvector calls in `setup/embed_eval.py` and `search.py` |
| **API provider** | Change `base_url` in OpenAI client (any OpenAI-compatible endpoint works) |

## TODO

- [ ] BM25 + Reciprocal Rank Fusion — hybrid keyword + vector retrieval for better recall on exact names and quotes
- [ ] Embedding fine-tuning — train/test split already in code, extend for domain-specific tuning
- [ ] LLM answer generation — synthesize natural language answers from retrieved chunks
- [ ] Cross-encoder re-ranking — second-stage re-ranker between retrieval and output
- [ ] More genres — extend beyond comedy, make the data more comprehensive.
- [ ] FastAPI wrapper — production API server around search.py
- [ ] Extend to TV series.

## Project Structure

```
movierag/
├── search.py                   # Main: semantic search with citations
├── show_results.py             # Retrieval quality inspector
├── setup/                      # One-time build scripts
│   ├── collect_data.py         # NB1: scrape Wikipedia + subtitles
│   ├── embed_eval.py           # NB2: chunk, embed, evaluate models
│   └── select_model.py         # MTEB model filtering
├── data/                       # Corpus + cached artifacts
│   ├── movies.csv              # Input movie list
│   ├── movies_corpus.json      # Scraped corpus
│   ├── chunks_cache.json       # Cached chunks (incremental)
│   ├── test_dataset.json       # Synthetic QA pairs
│   └── eval_results.json       # Evaluation results + run history
├── .env                        # API keys (git-ignored)
└── .gitignore
```
+                                                                                       
  ##Credits                                                                            

  Embedding evaluation methodology adapted from [donbr's embedding fine-tuning walkthrough](https://gist.github.com/donbr/696569a74bf7dbe90813177807ce1064).   
