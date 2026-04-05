"""
Movie RAG — Chunk, Embed, Evaluate (Notebook 2)
=================================================
Follows the methodology from the embedding fine-tuning walkthrough:
  1. Load corpus → chunk documents → assign IDs
  2. Split into train / test sets
  3. Generate synthetic QA pairs (question → document_id mapping)
  4. Embed corpus + queries with 3 models via OpenRouter
  5. Evaluate retrieval (hit rate, MRR, nDCG) per model
  6. Save datasets for NB3 (best model + LLM → RAG pipeline)

Chunking Strategy
-----------------
  Plot text:     RecursiveCharacterTextSplitter (sentence-aware, 750 chars, 50 overlap)
  Article text:  RecursiveCharacterTextSplitter (same)
  Metadata:      Single structured chunk per movie (director, cast, year, runtime, etc.)
  Subtitles:     Time-windowed dialogue chunks (2-min windows)

Models (all via OpenRouter)
---------------------------
  pplx-embed-v1-0.6b       596M  retrieval=65.41  $0.004/M tokens  no instructions
  Qwen3-Embedding-0.6B     596M  retrieval=64.65  $0.005/M tokens  instruction-aware
  text-embedding-3-small      ?  retrieval=~62    $0.02/M tokens   no instructions

QA Generation
-------------
  Uses a cheap LLM on OpenRouter (Qwen/Llama/Gemini) to generate
  2 questions per document chunk on the test split.

Requirements
------------
    pip install openai psycopg2-binary tqdm numpy

    docker run -d --name pgvector -e POSTGRES_PASSWORD=postgres \\
        -p 5432:5432 pgvector/pgvector:pg16

    export OPENROUTER_API_KEY="sk-or-v1-..."

Usage
-----
    python notebook2_embed_evaluate.py
    python notebook2_embed_evaluate.py --skip-embed     # reuse embeddings
    python notebook2_embed_evaluate.py --skip-qa        # reuse qa_pairs.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import time
import uuid
from typing import Optional

import numpy as np
from tqdm import tqdm

log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Config
# ═════════════════════════════════════════════════════════════════════════════

EMBEDDING_MODELS = {
    "pplx_0_6b": {
        "id": "perplexity/pplx-embed-v1-0.6b",
        "dims": 1024,
        "needs_instruction": False,
    },
    "qwen_4b": {
        "id": "qwen/qwen3-embedding-4b",
        "dims": 1024,  # truncated from 2560 (Matryoshka support)
        "needs_instruction": True,
        "query_prefix": "Instruct: Given a movie search query, retrieve relevant passages about movies\nQuery: ",
    },
    "openai_small": {
        "id": "openai/text-embedding-3-small",
        "dims": 1536,
        "needs_instruction": False,
    },
}

# Cheap LLMs for QA generation (via OpenRouter) — one question each per chunk
QA_LLMS = [
    "qwen/qwen3.5-9b",
    "mistralai/mistral-small-creative",
]

CHUNK_SIZE = 750
CHUNK_OVERLAP = 50
SUBTITLE_WINDOW_SEC = 120
QUESTIONS_PER_CHUNK = 2
TEST_SPLIT_RATIO = 0.2
TOP_K = 5


# ═════════════════════════════════════════════════════════════════════════════
# Step 1: Chunking — RecursiveCharacterTextSplitter style
# ═════════════════════════════════════════════════════════════════════════════

# Separators used by LangChain's RecursiveCharacterTextSplitter
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]


def recursive_split(text: str, size: int = CHUNK_SIZE,
                    overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text recursively, trying larger separators first."""
    if len(text) <= size:
        return [text.strip()] if text.strip() else []

    # find the best separator that produces chunks
    for sep in _SEPARATORS:
        if sep and sep in text:
            parts = text.split(sep)
            chunks = []
            current = ""
            for part in parts:
                candidate = (current + sep + part) if current else part
                if len(candidate) > size and current:
                    chunks.append(current.strip())
                    # overlap: keep tail words from previous chunk
                    if overlap > 0:
                        tail = current[-overlap:]
                        current = tail + sep + part
                    else:
                        current = part
                else:
                    current = candidate
            if current.strip():
                chunks.append(current.strip())
            if len(chunks) > 1:
                return chunks

    # fallback: hard split
    return [text[i:i + size].strip() for i in range(0, len(text), size - overlap)
            if text[i:i + size].strip()]


def _ts_to_seconds(ts: str) -> float:
    ts = ts.replace(",", ".")
    h, m, s = ts.split(":")
    return float(h) * 3600 + float(m) * 60 + float(s)


def chunk_movie(movie: dict) -> list[dict]:
    """Chunk one movie into documents with metadata.

    Returns list of {id, content, movie_title, source, metadata}.
    Each chunk gets a unique UUID.
    """
    docs = []
    title = movie["title"]
    year = movie.get("year", "")

    # ── Metadata chunk (one per movie) ───────────────────────────────
    meta_parts = [f"Title: {title}", f"Year: {year}"]
    for field, label in [("director", "Director"), ("cast", "Cast"),
                         ("writer", "Writer"), ("producer", "Producer"),
                         ("music", "Music"), ("runtime", "Runtime"),
                         ("country", "Country"), ("language", "Language"),
                         ("budget", "Budget"), ("box_office", "Box Office"),
                         ("release_date", "Release Date")]:
        val = movie.get(field, "")
        if val:
            if isinstance(val, list):
                val = ", ".join(val[:8])
            if val.strip():
                meta_parts.append(f"{label}: {val}")
    meta_text = "\n".join(meta_parts)
    docs.append({
        "id": str(uuid.uuid4()),
        "content": meta_text,
        "movie_title": title,
        "source": "metadata",
    })

    # ── Plot chunks (recursive character split) ──────────────────────
    plot = movie.get("plot_text", "") or ""
    if len(plot) > 50:
        for i, chunk in enumerate(recursive_split(plot)):
            docs.append({
                "id": str(uuid.uuid4()),
                "content": f"[{title} ({year})] Plot:\n{chunk}",
                "movie_title": title,
                "source": "plot",
            })

    # ── Article chunks (recursive character split) ───────────────────
    article = movie.get("full_article_text", "") or ""
    if len(article) > 100:
        for i, chunk in enumerate(recursive_split(article)):
            docs.append({
                "id": str(uuid.uuid4()),
                "content": f"[{title} ({year})] Article:\n{chunk}",
                "movie_title": title,
                "source": "article",
            })

    # ── Subtitle chunks (time-windowed dialogue) ─────────────────────
    cues = movie.get("subtitles", [])
    if cues:
        window_start = 0.0
        window_texts = []
        for cue in cues:
            try:
                cue_start = _ts_to_seconds(cue["start"])
            except (KeyError, ValueError):
                continue
            if cue_start - window_start > SUBTITLE_WINDOW_SEC and window_texts:
                dialogue = " ".join(window_texts)
                docs.append({
                    "id": str(uuid.uuid4()),
                    "content": f"[{title} ({year})] Dialogue ({int(window_start // 60)}m-{int(cue_start // 60)}m):\n{dialogue}",
                    "movie_title": title,
                    "source": "subtitles",
                })
                window_start = cue_start
                window_texts = []
            window_texts.append(cue["text"])
        if window_texts:
            end_sec = _ts_to_seconds(cues[-1]["end"]) if "end" in cues[-1] else window_start + SUBTITLE_WINDOW_SEC
            docs.append({
                "id": str(uuid.uuid4()),
                "content": f"[{title} ({year})] Dialogue ({int(window_start // 60)}m-{int(end_sec // 60)}m):\n{' '.join(window_texts)}",
                "movie_title": title,
                "source": "subtitles",
            })

    return docs


def chunk_corpus(corpus: list[dict]) -> list[dict]:
    """Chunk all movies. Returns list of document dicts with unique IDs."""
    all_docs = []
    for movie in corpus:
        all_docs.extend(chunk_movie(movie))
    return all_docs


def chunk_corpus_incremental(corpus: list[dict], cache_path: str) -> tuple[list[dict], list[dict]]:
    """Chunk only new movies not already in cache.

    Returns (all_docs, new_docs) where new_docs is the subset that was
    freshly chunked this run.
    """
    # Load existing cache: {movie_title: [chunk_dicts]}
    cached = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        log.info("  Loaded chunk cache: %d movies cached", len(cached))

    cached_titles = set(cached.keys())
    corpus_titles = {m["title"] for m in corpus}
    new_movies = [m for m in corpus if m["title"] not in cached_titles]

    # Chunk only new movies
    new_docs = []
    for movie in new_movies:
        chunks = chunk_movie(movie)
        cached[movie["title"]] = chunks
        new_docs.extend(chunks)

    if new_movies:
        log.info("  Chunked %d new movies → %d new chunks", len(new_movies), len(new_docs))
    else:
        log.info("  No new movies to chunk")

    # Remove movies no longer in corpus
    removed = cached_titles - corpus_titles
    for title in removed:
        del cached[title]
    if removed:
        log.info("  Removed %d movies no longer in corpus", len(removed))

    # Save updated cache
    with open(cache_path, "w") as f:
        json.dump(cached, f)

    # Flatten all cached chunks
    all_docs = []
    for title in sorted(cached.keys()):
        all_docs.extend(cached[title])

    return all_docs, new_docs


# ═════════════════════════════════════════════════════════════════════════════
# Step 2: Train / Test split (by movie, not by chunk)
# NOTE: Not used in the current eval pipeline. Kept here to extend
#       functionality for embedding fine-tuning (NB3+), where we need
#       a proper train/test split to avoid data leakage during training.
# ═════════════════════════════════════════════════════════════════════════════

def split_by_movie(documents: list[dict], corpus: list[dict],
                   test_ratio: float = TEST_SPLIT_RATIO):
    """Split documents into train/test by movie title.

    Returns (train_docs, test_docs, test_movie_titles).
    """
    import random
    random.seed(42)

    titles = list({m["title"] for m in corpus})
    random.shuffle(titles)
    n_test = max(1, int(len(titles) * test_ratio))
    test_titles = set(titles[:n_test])
    train_titles = set(titles[n_test:])

    train_docs = [d for d in documents if d["movie_title"] in train_titles]
    test_docs = [d for d in documents if d["movie_title"] in test_titles]

    return train_docs, test_docs, test_titles


# ═════════════════════════════════════════════════════════════════════════════
# Step 3: Synthetic QA generation via cheap OpenRouter LLM
# ═════════════════════════════════════════════════════════════════════════════

QA_PROMPT_DEFAULT = """Given the following context from a movie database, generate exactly {n} questions that could be answered using ONLY the provided context.

The question must be SPECIFIC — reference concrete names, events, or details from the context.
Bad: "Who directed this movie?" Good: "Who directed The Godfather?"
Bad: "What happens in the plot?" Good: "What does Michael Corleone do after his father is shot?"

Format: Return one question per line, numbered.

Context:
{context}"""

QA_PROMPT_DIALOGUE = """Below is a section of movie dialogue/subtitles. Generate exactly {n} questions that someone might search for, where this dialogue is the answer.

Focus on:
- Quoting or paraphrasing a specific memorable line and asking who says it or which movie it's from
- Asking about the topic or situation being discussed in the dialogue
- Asking which movie features a scene where characters talk about [specific topic from dialogue]

The question must reference SPECIFIC details from the dialogue — not generic questions.
Bad: "Which movie has an argument scene?" Good: "In which movie does a character say 'I'm gonna make him an offer he can't refuse'?"

Format: Return one question per line, numbered.

Dialogue:
{context}"""


def _qa_prompt_for(doc: dict) -> str:
    if doc.get("source") == "subtitles":
        return QA_PROMPT_DIALOGUE
    return QA_PROMPT_DEFAULT


def _parse_question(text: str) -> str | None:
    """Extract a question from LLM output, trying multiple formats."""
    # Strip <think>...</think> blocks (some models emit chain-of-thought)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not text:
        return None
    # Try numbered format: "1. question"
    for line in text.split("\n"):
        line = line.strip()
        if line and len(line) > 5 and line[0].isdigit() and ". " in line:
            q_text = line.split(". ", 1)[1].strip()
            if len(q_text) > 10:
                return q_text
    # Try bullet/dash format
    for line in text.split("\n"):
        line = line.strip().lstrip("- *•#").strip()
        if len(line) > 15 and line.endswith("?"):
            return line
    # Any line ending with ?
    for line in text.split("\n"):
        line = line.strip()
        if len(line) > 15 and line.endswith("?"):
            return line
    # Last resort: first substantial line
    for line in text.split("\n"):
        line = line.strip()
        if len(line) > 15:
            return line
    return None


def _generate_one_question(client, llm: str, doc: dict, retries: int = 2) -> str | None:
    """Call LLM to generate one question. Retries with alternate LLM on failure."""
    llms_to_try = [llm] + [m for m in QA_LLMS if m != llm][:retries]
    for model in llms_to_try:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": _qa_prompt_for(doc).format(n=1, context=doc["content"][:1500]),
                }],
                temperature=0.3,
                max_tokens=256,
                timeout=30,
            )
            text = resp.choices[0].message.content or ""
            q = _parse_question(text)
            if q:
                return q
            log.debug("Unparseable response from %s: %s", model, text[:100])
        except Exception as e:
            log.warning("QA gen failed (%s) for doc %s: %s", model, doc["id"][:8], e)
        time.sleep(0.2)
    return None


def generate_qa_pairs(test_docs: list[dict]) -> dict:
    """Generate 2 QA pairs per movie: one from dialogue, one from metadata/plot.

    LLMs alternate across movies so both models do both task types.
    Returns the gist-style dataset: {questions, relevant_contexts, corpus}.
    """
    import random
    random.seed(42)

    from openai import OpenAI
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    corpus = {d["id"]: d["content"] for d in test_docs}
    questions = {}
    relevant_docs = {}

    # Group test docs by movie
    by_movie: dict[str, dict[str, list[dict]]] = {}
    for doc in test_docs:
        title = doc["movie_title"]
        source = doc["source"]
        by_movie.setdefault(title, {}).setdefault(source, []).append(doc)

    for i, title in enumerate(tqdm(sorted(by_movie.keys()), desc="Generating QA (per movie)")):
        sources = by_movie[title]

        # Alternate LLM assignment per movie
        dialogue_llm = QA_LLMS[i % len(QA_LLMS)]
        factual_llm = QA_LLMS[(i + 1) % len(QA_LLMS)]

        # 1. Dialogue question — subtitles preferred, fallback to article
        dialogue_pool = sources.get("subtitles", []) or sources.get("article", [])
        if dialogue_pool:
            doc = random.choice(dialogue_pool)
            q_text = _generate_one_question(client, dialogue_llm, doc)
            if q_text:
                q_id = str(uuid.uuid4())
                questions[q_id] = q_text
                relevant_docs[q_id] = [doc["id"]]

        # 2. Factual question — metadata/plot preferred, fallback to article
        factual_pool = sources.get("metadata", []) + sources.get("plot", [])
        if not factual_pool:
            factual_pool = sources.get("article", [])
        if factual_pool:
            doc = random.choice(factual_pool)
            q_text = _generate_one_question(client, factual_llm, doc)
            if q_text:
                q_id = str(uuid.uuid4())
                questions[q_id] = q_text
                relevant_docs[q_id] = [doc["id"]]

        time.sleep(0.1)

    log.info("Generated %d questions from %d movies (target: %d)",
             len(questions), len(by_movie), len(by_movie) * 2)
    return {
        "questions": questions,
        "relevant_contexts": relevant_docs,
        "corpus": corpus,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Step 4: Embedding via OpenRouter
# ═════════════════════════════════════════════════════════════════════════════

def get_or_client():
    from openai import OpenAI
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


def embed_texts(client, model_id: str, texts: list[str],
                batch_size: int = 64, truncate_dims: int | None = None) -> list[list[float]]:
    """Embed a list of texts in batches via OpenRouter."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            kwargs = {"model": model_id, "input": batch, "timeout": 30}
            if truncate_dims:
                kwargs["dimensions"] = truncate_dims
            resp = client.embeddings.create(**kwargs)
            sorted_data = sorted(resp.data, key=lambda x: x.index)
            embs = [d.embedding for d in sorted_data]
            # Fallback truncation if API doesn't honor dimensions param
            if truncate_dims:
                embs = [e[:truncate_dims] for e in embs]
            all_embs.extend(embs)
        except Exception as e:
            log.error("Embedding batch %d failed: %s", i, e)
            dims = truncate_dims or 10
            all_embs.extend([[0.0] * dims] * len(batch))
        time.sleep(0.1)
    return all_embs


# ═════════════════════════════════════════════════════════════════════════════
# Step 4b: pgvector storage
# ═════════════════════════════════════════════════════════════════════════════

def get_pg(dsn="postgresql://postgres:postgres@localhost:5432/postgres"):
    import psycopg2
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    return conn


def setup_table(conn, table: str, dims: int):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id TEXT PRIMARY KEY,
                movie_title TEXT NOT NULL,
                source TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding vector({dims})
            )
        """)
        # Create index only if it doesn't exist
        cur.execute("""
            SELECT 1 FROM pg_indexes WHERE tablename = %s
            AND (indexdef LIKE '%%hnsw%%' OR indexdef LIKE '%%ivfflat%%')
        """, (table,))
        if not cur.fetchone():
            if dims <= 2000:
                cur.execute(f"""
                    CREATE INDEX ON {table}
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64)
                """)
            else:
                cur.execute(f"""
                    CREATE INDEX ON {table}
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """)


def get_existing_ids(conn, table: str) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(f"SELECT id FROM {table}")
        return {row[0] for row in cur.fetchall()}


def insert_rows(conn, table: str, docs: list[dict], embeddings: list[list[float]]):
    with conn.cursor() as cur:
        for doc, emb in zip(docs, embeddings):
            cur.execute(
                f"INSERT INTO {table} (id, movie_title, source, content, embedding) "
                f"VALUES (%s, %s, %s, %s, %s)",
                (doc["id"], doc["movie_title"], doc["source"],
                 doc["content"], str(emb)),
            )


def search_similar(conn, table: str, query_emb: list[float], top_k: int = 5):
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT id, movie_title, source, content,
                   1 - (embedding <=> %s::vector) AS score
            FROM {table}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (str(query_emb), str(query_emb), top_k))
        cols = ["id", "movie_title", "source", "content", "score"]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


# ═════════════════════════════════════════════════════════════════════════════
# Step 5: Evaluation (following gist methodology)
# TODO: Add BM25 keyword search + Reciprocal Rank Fusion (RRF) as a hybrid
#       retrieval baseline. RRF combines vector similarity and BM25 ranked
#       lists to improve recall on keyword-heavy queries (e.g. actor names,
#       exact dialogue quotes) where pure embedding search underperforms.
#
# Three retrieval metrics, from simplest to most informative:
#
# Hit Rate @k (a.k.a. Recall@k)
#   "Did the correct document appear anywhere in the top-k results?"
#   Yes → 1, No → 0.  Averaged across all queries.
#   Tells you: does the model work at all?
#
# MRR @k (Mean Reciprocal Rank)
#   "How high does the correct document rank?"
#   If correct doc is rank 1 → 1/1 = 1.0, rank 3 → 1/3 = 0.33, not found → 0.
#   Averaged across all queries.
#   Tells you: how well does the model rank the right answer?
#
# nDCG @k (Normalized Discounted Cumulative Gain)
#   "Does the correct document appear early in the ranked list?"
#   Uses logarithmic discount: rank 1 gets full credit, rank 2 gets less,
#   rank 5 gets much less.  Normalized to [0, 1] against the ideal ranking.
#   This is the standard metric on the MTEB leaderboard (nDCG@10).
#   Tells you: how good is the overall ranking quality?
#
# Together: Hit Rate = "does it work", MRR = "how fast do you find it",
#           nDCG = "how good is the full ranking" (MTEB standard).
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_model(conn, table: str, client, model_cfg: dict,
                   dataset: dict, top_k: int = TOP_K) -> list[dict]:
    """Evaluate one model using the gist's is_hit approach.

    For each question, check if the expected document appears in top-k results.
    Returns list of {id, question, expected_id, is_hit, rank, mrr, ndcg}.
    """
    questions = dataset["questions"]
    relevant_docs = dataset["relevant_contexts"]

    eval_results = []
    for q_id, question in tqdm(questions.items(), desc=f"Eval {table}", leave=False):
        # embed query (with instruction prefix if needed)
        q_text = question
        if model_cfg.get("needs_instruction"):
            q_text = model_cfg.get("query_prefix", "") + question

        q_emb = embed_texts(client, model_cfg["id"], [q_text],
                            truncate_dims=model_cfg["dims"])[0]

        # retrieve
        results = search_similar(conn, table, q_emb, top_k=top_k)
        retrieved_ids = [r["id"] for r in results]
        expected_id = relevant_docs[q_id][0]

        # Hit Rate: is the expected doc anywhere in top-k? (binary: 1 or 0)
        is_hit = expected_id in retrieved_ids

        # MRR: reciprocal of the rank where expected doc first appears
        # rank 1 → 1.0, rank 2 → 0.5, rank 5 → 0.2, not found → 0.0
        rank = None
        mrr_score = 0.0
        for i, rid in enumerate(retrieved_ids):
            if rid == expected_id:
                rank = i + 1
                mrr_score = 1.0 / rank
                break

        # nDCG: discounted cumulative gain, normalized against ideal ranking
        # Uses log2 discount: position 1 = full credit, later positions = less
        # Binary relevance: 1.0 if correct doc, 0.0 otherwise
        rels = [1.0 if rid == expected_id else 0.0 for rid in retrieved_ids]
        dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
        ideal = sorted(rels, reverse=True)
        idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
        ndcg_score = dcg / idcg if idcg > 0 else 0.0

        eval_results.append({
            "id": q_id,
            "question": question,
            "expected_id": expected_id,
            "is_hit": is_hit,
            "rank": rank,
            "mrr": mrr_score,
            "ndcg": ndcg_score,
        })

    return eval_results


# ═════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Movie RAG — Embedding Evaluation")
    parser.add_argument("--corpus", default="data/movies_corpus.json")
    parser.add_argument("--output", default="data/eval_results.json")
    parser.add_argument("--qa-cache", default="data/test_dataset.json")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--qa-movies", type=int, default=50)
    parser.add_argument("--skip-embed", action="store_true")
    parser.add_argument("--skip-qa", action="store_true")
    parser.add_argument("--pg-dsn", default="postgresql://postgres:postgres@localhost:5432/postgres")
    parser.add_argument("--chunks-cache", default="data/chunks_cache.json")
    parser.add_argument("--full-rebuild", action="store_true",
                        help="Ignore caches and redo everything from scratch")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── 1. Load corpus ───────────────────────────────────────────────────
    log.info("Loading corpus: %s", args.corpus)
    with open(args.corpus) as f:
        corpus = json.load(f)
    log.info("  %d movies loaded", len(corpus))

    # ── 2. Chunk + assign IDs (incremental) ────────────────────────────
    if args.full_rebuild and os.path.exists(args.chunks_cache):
        os.remove(args.chunks_cache)
    log.info("Chunking corpus (incremental)...")
    all_docs, new_docs = chunk_corpus_incremental(corpus, args.chunks_cache)
    log.info("  %d total chunks (%d new)", len(all_docs), len(new_docs))
    by_source = {}
    for d in all_docs:
        by_source[d["source"]] = by_source.get(d["source"], 0) + 1
    for src, n in sorted(by_source.items()):
        log.info("    %-12s %d", src, n)

    # All docs go into the retrieval corpus
    retrieval_corpus = all_docs

    # ── 3. Select movies for QA (prefer movies with subtitles) ─────────
    import random
    random.seed(42)
    # Check which movies have subtitles
    movies_with_subs = {d["movie_title"] for d in all_docs if d["source"] == "subtitles"}
    movies_without_subs = {m["title"] for m in corpus} - movies_with_subs
    # Pick from movies with subtitles first, then fill remainder
    with_subs = sorted(movies_with_subs)
    without_subs = sorted(movies_without_subs)
    random.shuffle(with_subs)
    random.shuffle(without_subs)
    qa_titles_list = with_subs[:args.qa_movies]
    if len(qa_titles_list) < args.qa_movies:
        qa_titles_list += without_subs[:args.qa_movies - len(qa_titles_list)]
    qa_titles = set(qa_titles_list[:args.qa_movies])
    qa_docs = [d for d in all_docs if d["movie_title"] in qa_titles]
    n_with_subs = len(qa_titles & movies_with_subs)
    log.info("  Selected %d movies for QA (%d chunks, %d/%d have subtitles)",
             len(qa_titles), len(qa_docs), n_with_subs, len(qa_titles))

    # ── 4. Generate synthetic QA (incremental) ───────────────────────────
    if args.full_rebuild and os.path.exists(args.qa_cache):
        os.remove(args.qa_cache)

    existing_qa = {"questions": {}, "relevant_contexts": {}, "corpus": {}}
    if os.path.exists(args.qa_cache):
        with open(args.qa_cache) as f:
            existing_qa = json.load(f)
        log.info("  Loaded %d existing QA pairs", len(existing_qa["questions"]))

    if not args.skip_qa:
        # Find movies that don't already have QA
        existing_movie_titles = set()
        for doc_ids in existing_qa["relevant_contexts"].values():
            for did in doc_ids:
                for d in qa_docs:
                    if d["id"] == did:
                        existing_movie_titles.add(d["movie_title"])
        new_qa_docs = [d for d in qa_docs if d["movie_title"] not in existing_movie_titles]

        if new_qa_docs:
            log.info("Generating QA for %d new movies via %s...",
                     len({d["movie_title"] for d in new_qa_docs}), QA_LLMS)
            new_qa = generate_qa_pairs(new_qa_docs)
            existing_qa["questions"].update(new_qa["questions"])
            existing_qa["relevant_contexts"].update(new_qa["relevant_contexts"])
            log.info("  Total QA pairs: %d", len(existing_qa["questions"]))
        else:
            log.info("  All QA movies already have questions")

        test_dataset = existing_qa
        with open(args.qa_cache, "w") as f:
            json.dump(test_dataset, f, indent=2)
        log.info("  Saved %d QA pairs to %s",
                 len(test_dataset["questions"]), args.qa_cache)
    else:
        test_dataset = existing_qa
        log.info("  %d QA pairs loaded (skip-qa)", len(test_dataset["questions"]))

    corpus_map = {d["id"]: d["content"] for d in retrieval_corpus}
    test_dataset["corpus"] = corpus_map

    # ── 5. Embed + store per model ───────────────────────────────────────
    or_client = get_or_client()
    pg = get_pg(args.pg_dsn)

    corpus_texts = [d["content"] for d in retrieval_corpus]

    if not args.skip_embed:
        for model_name, cfg in EMBEDDING_MODELS.items():
            table = f"emb_{model_name}"
            log.info("── Embedding with %s (%s) ──", model_name, cfg["id"])

            if args.full_rebuild:
                # Drop and recreate for full rebuild
                with pg.cursor() as cur:
                    cur.execute(f"DROP TABLE IF EXISTS {table}")

            setup_table(pg, table, cfg["dims"])

            # Find chunks not yet embedded
            existing_ids = get_existing_ids(pg, table)
            new_to_embed = [(d, t) for d, t in zip(retrieval_corpus, corpus_texts)
                            if d["id"] not in existing_ids]

            if new_to_embed:
                new_embed_docs = [x[0] for x in new_to_embed]
                new_embed_texts = [x[1] for x in new_to_embed]
                log.info("  Embedding %d new chunks (%d already stored)...",
                         len(new_embed_texts), len(existing_ids))
                embeddings = embed_texts(or_client, cfg["id"], new_embed_texts,
                                        truncate_dims=cfg["dims"])
                insert_rows(pg, table, new_embed_docs, embeddings)
                log.info("  Stored %d new embeddings in %s", len(embeddings), table)
            else:
                log.info("  All %d chunks already embedded, skipping", len(existing_ids))
    else:
        log.info("Skipping embedding (--skip-embed)")

    # ── 6. Evaluate each model ───────────────────────────────────────────
    log.info("── Evaluation (top_k=%d) ──", args.top_k)

    all_results = {}
    for model_name, cfg in EMBEDDING_MODELS.items():
        table = f"emb_{model_name}"
        log.info("Evaluating %s...", model_name)

        eval_results = evaluate_model(
            pg, table, or_client, cfg, test_dataset, top_k=args.top_k
        )

        hit_rate = np.mean([r["is_hit"] for r in eval_results])
        avg_mrr = np.mean([r["mrr"] for r in eval_results])
        avg_ndcg = np.mean([r["ndcg"] for r in eval_results])

        all_results[model_name] = {
            "hit_rate": float(hit_rate),
            "mrr": float(avg_mrr),
            "ndcg": float(avg_ndcg),
            "n_queries": len(eval_results),
            "details": eval_results,
        }
        log.info("  hit_rate=%.4f  mrr=%.4f  ndcg=%.4f  (%d queries)",
                 hit_rate, avg_mrr, avg_ndcg, len(eval_results))

    # ── 7. Print comparison table ────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"  MODEL COMPARISON — Retrieval @{args.top_k}")
    print("=" * 78)
    print(f"  {'Model':<22} {'Hit Rate':>10} {'MRR':>10} {'nDCG':>10} {'Queries':>8}")
    print("  " + "-" * 70)

    best_model = None
    best_hr = -1

    for name, res in all_results.items():
        hr = res["hit_rate"]
        m = res["mrr"]
        n = res["ndcg"]
        q = res["n_queries"]
        marker = ""
        if hr > best_hr:
            best_hr = hr
            best_model = name
        print(f"  {name:<22} {hr:>10.4f} {m:>10.4f} {n:>10.4f} {q:>8}")

    print("  " + "-" * 70)
    print(f"  Best model: {best_model} (hit_rate={best_hr:.4f})")

    # cost estimate
    total_chars = sum(len(t) for t in corpus_texts)
    approx_tokens = total_chars / 4
    prices = {"pplx_0_6b": 0.004, "qwen_4b": 0.02, "openai_small": 0.02}
    print(f"\n  Corpus: {len(retrieval_corpus)} chunks, ~{approx_tokens / 1e6:.2f}M tokens")
    for name in EMBEDDING_MODELS:
        cost = approx_tokens / 1e6 * prices.get(name, 0.01)
        print(f"    {name}: ~${cost:.4f}")

    print("=" * 78)
    print(f"\n  → Use {best_model} in NB3 for RAG pipeline with LLM.")
    print(f"  → Datasets saved: {args.qa_cache}")

    # ── Save results (merge with existing) ─────────────────────────────
    existing_output = {}
    if os.path.exists(args.output) and not args.full_rebuild:
        with open(args.output) as f:
            existing_output = json.load(f)

    # Build current run's output
    run_output = {
        "best_model": best_model,
        "top_k": args.top_k,
        "models": {},
        "corpus_stats": {
            "n_movies": len(corpus),
            "n_chunks": len(retrieval_corpus),
            "chunks_by_source": by_source,
            "n_qa_movies": len(qa_titles),
            "n_qa_docs": len(qa_docs),
        },
    }
    for name, res in all_results.items():
        run_output["models"][name] = {
            "config": EMBEDDING_MODELS[name],
            "hit_rate": res["hit_rate"],
            "mrr": res["mrr"],
            "ndcg": res["ndcg"],
            "n_queries": res["n_queries"],
        }

    # Merge: keep history of past runs
    runs = existing_output.get("runs", [])
    runs.append(run_output)
    final_output = run_output.copy()
    final_output["runs"] = runs

    with open(args.output, "w") as f:
        json.dump(final_output, f, indent=2)
    log.info("Results saved to %s (%d total runs)", args.output, len(runs))


if __name__ == "__main__":
    main()
