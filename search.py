"""
Movie RAG — Search Pipeline (Notebook 3)
=========================================
Semantic search engine over the movie corpus. Takes a natural language
query and returns relevant passages with citations.

The default embedding model is pplx_0_6b (perplexity/pplx-embed-v1-0.6b),
selected based on our evaluation in eval_embed.py where it achieved the
best retrieval metrics (hit_rate=0.88, MRR=0.70, nDCG=0.75) at the
lowest cost ($0.004/M tokens). You can swap the model by passing
--model to use any model that was embedded in eval_embed.py.

Prerequisites
-------------
    # pgvector must be running with embeddings already stored via eval_embed.py
    docker start pgvector_movierag

    export OPENROUTER_API_KEY="sk-or-v1-..."

Usage
-----
    # Interactive mode
    python search.py

    # Single query
    python search.py -q "Who directed The Hangover?"

    # More results
    python search.py -q "funny wedding scenes" --top-k 10

    # Use a different model
    python search.py --model openai_small -q "bachelor party movie"
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap

MODELS = {
    "pplx_0_6b": {
        "id": "perplexity/pplx-embed-v1-0.6b",
        "dims": 1024,
    },
    "qwen_4b": {
        "id": "qwen/qwen3-embedding-4b",
        "dims": 1024,
    },
    "openai_small": {
        "id": "openai/text-embedding-3-small",
        "dims": 1536,
    },
}


def connect_pg(dsn):
    import psycopg2
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    return conn


def embed_query(client, model_id, text, dims):
    resp = client.embeddings.create(model=model_id, input=[text], dimensions=dims)
    emb = resp.data[0].embedding
    return emb[:dims]


def search(conn, table, query_emb, top_k=5):
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


def format_citation(result):
    """Build a citation string from a result."""
    title = result["movie_title"]
    source = result["source"]

    if source == "subtitles":
        # Extract time range from content like "[Movie (Year)] Dialogue (10m-12m):"
        content = result["content"]
        if "Dialogue (" in content:
            time_range = content.split("Dialogue (")[1].split("):")[0]
            return f"{title} — dialogue at {time_range}"
        return f"{title} — dialogue"
    elif source == "plot":
        return f"{title} — plot summary"
    elif source == "article":
        return f"{title} — Wikipedia article"
    elif source == "metadata":
        return f"{title} — metadata (cast, crew, etc.)"
    return f"{title} — {source}"


def format_result(result, rank):
    """Pretty-print one search result."""
    score = result["score"]
    citation = format_citation(result)
    content = result["content"]

    # Strip the prefix like "[Movie (Year)] Plot:\n"
    if ":\n" in content[:100]:
        content = content.split(":\n", 1)[1]

    wrapped = textwrap.fill(content.strip(), width=80, initial_indent="    ",
                            subsequent_indent="    ")
    # Truncate long content
    lines = wrapped.split("\n")
    if len(lines) > 6:
        wrapped = "\n".join(lines[:6]) + "\n    ..."

    return (
        f"  #{rank}  [{score:.3f}]  {citation}\n"
        f"\n"
        f"{wrapped}\n"
    )


def deduplicate_by_movie(results, top_k):
    """Keep only the best-scoring chunk per movie."""
    seen = {}
    for r in results:
        title = r["movie_title"]
        if title not in seen:
            seen[title] = r
    return list(seen.values())[:top_k]


def run_query(client, conn, model_cfg, table, query, top_k):
    """Run a single query and print results."""
    q_emb = embed_query(client, model_cfg["id"], query, model_cfg["dims"])
    # Fetch extra results so we have enough after dedup
    raw_results = search(conn, table, q_emb, top_k=top_k * 3)

    if not raw_results:
        print("  No results found.\n")
        return

    results = deduplicate_by_movie(raw_results, top_k)

    for i, r in enumerate(results):
        print(format_result(r, i + 1))

    print(f"  ── {len(results)} movies ──\n")


def main():
    parser = argparse.ArgumentParser(
        description="Movie RAG search — semantic retrieval with citations"
    )
    parser.add_argument("-q", "--query", help="Single query (otherwise interactive mode)")
    parser.add_argument("--model", default="pplx_0_6b", choices=MODELS.keys(),
                        help="Embedding model to use (default: pplx_0_6b)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--pg-dsn",
                        default="postgresql://postgres:postgres@localhost:5433/postgres")
    args = parser.parse_args()

    from openai import OpenAI
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    conn = connect_pg(args.pg_dsn)
    model_cfg = MODELS[args.model]
    table = f"emb_{args.model}"

    # Verify table exists
    with conn.cursor() as cur:
        cur.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_name = %s
            )
        """, (table,))
        if not cur.fetchone()[0]:
            print(f"Error: table '{table}' not found. Run eval_embed.py first to embed the corpus.")
            sys.exit(1)

    if args.query:
        # Single query mode
        print(f"\n  Query: {args.query}\n")
        run_query(client, conn, model_cfg, table, args.query, args.top_k)
    else:
        # Interactive mode
        print(f"\n{'=' * 60}")
        print(f"  Movie RAG Search")
        print(f"  Model: {args.model} ({model_cfg['id']})")
        print(f"  Top-k: {args.top_k}")
        print(f"  Type 'quit' to exit")
        print(f"{'=' * 60}\n")

        while True:
            try:
                query = input("  🔍 ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n")
                break

            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                break

            print()
            run_query(client, conn, model_cfg, table, query, args.top_k)


if __name__ == "__main__":
    main()
