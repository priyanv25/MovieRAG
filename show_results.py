"""
Show retrieval results: sample QA pairs with expected vs retrieved chunks.

Usage:
    python show_results.py                          # 5 random examples
    python show_results.py --n 10                   # 10 random examples
    python show_results.py --n 10 --misses-only     # only show misses
    python show_results.py --model qwen_4b          # use a different model
    python show_results.py --all                    # show all 100 queries
"""

import argparse
import json
import os
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="Number of examples")
    parser.add_argument("--model", default="pplx_0_6b", help="Model table to query")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true", help="Show all queries")
    parser.add_argument("--hits-only", action="store_true")
    parser.add_argument("--misses-only", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--pg-dsn", default="postgresql://postgres:postgres@localhost:5433/postgres")
    parser.add_argument("--qa-cache", default="data/test_dataset.json")
    parser.add_argument("--chunks-cache", default="data/chunks_cache.json")
    args = parser.parse_args()

    import psycopg2
    from openai import OpenAI

    EMBEDDING_MODELS = {
        "pplx_0_6b": {"id": "perplexity/pplx-embed-v1-0.6b", "dims": 1024},
        "qwen_4b": {"id": "qwen/qwen3-embedding-4b", "dims": 1024},
        "openai_small": {"id": "openai/text-embedding-3-small", "dims": 1536},
    }

    cfg = EMBEDDING_MODELS[args.model]
    table = f"emb_{args.model}"

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    with open(args.qa_cache) as f:
        data = json.load(f)

    # Build chunk lookup from cache
    id_to_chunk = {}
    with open(args.chunks_cache) as f:
        cached = json.load(f)
    for title, chunks in cached.items():
        for c in chunks:
            id_to_chunk[c["id"]] = c

    conn = psycopg2.connect(args.pg_dsn)
    conn.autocommit = True

    questions = list(data["questions"].items())
    relevant = data["relevant_contexts"]

    # Run retrieval for all questions
    results_list = []
    for q_id, q_text in questions:
        expected_id = relevant[q_id][0]
        expected_chunk = id_to_chunk.get(expected_id, {})

        kwargs = {"model": cfg["id"], "input": [q_text]}
        if cfg["dims"]:
            kwargs["dimensions"] = cfg["dims"]
        resp = client.embeddings.create(**kwargs)
        q_emb = resp.data[0].embedding[:cfg["dims"]]

        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT id, movie_title, source, content,
                       1 - (embedding <=> %s::vector) AS score
                FROM {table}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (str(q_emb), str(q_emb), args.top_k))
            retrieved = cur.fetchall()

        hit = any(r[0] == expected_id for r in retrieved)
        rank = next((i + 1 for i, r in enumerate(retrieved) if r[0] == expected_id), None)

        results_list.append({
            "q_id": q_id,
            "question": q_text,
            "expected_id": expected_id,
            "expected_chunk": expected_chunk,
            "retrieved": retrieved,
            "hit": hit,
            "rank": rank,
        })

    # Filter
    if args.hits_only:
        results_list = [r for r in results_list if r["hit"]]
    elif args.misses_only:
        results_list = [r for r in results_list if not r["hit"]]

    # Sample
    if not args.all:
        random.seed(args.seed)
        n = min(args.n, len(results_list))
        results_list = random.sample(results_list, n)

    # Aggregate stats
    total = len(data["questions"])
    all_hits = sum(1 for q_id in data["questions"]
                   for _ in [relevant[q_id][0]]
                   if any(True for r in results_list if r["q_id"] == q_id and r["hit"]))

    # Print
    n_hits = sum(1 for r in results_list if r["hit"])
    n_miss = sum(1 for r in results_list if not r["hit"])

    print(f"\n{'=' * 90}")
    print(f"  RETRIEVAL RESULTS — model: {args.model} | top_k: {args.top_k}")
    print(f"  Showing {len(results_list)} examples ({n_hits} hits, {n_miss} misses)")
    print(f"{'=' * 90}\n")

    for i, r in enumerate(results_list):
        ec = r["expected_chunk"]
        expected_content = ec.get("content", "(not in cache)")
        expected_source = ec.get("source", "?")
        expected_movie = ec.get("movie_title", "?")

        status = f"\033[92m✓ HIT at rank {r['rank']}\033[0m" if r["hit"] else "\033[91m✗ MISS\033[0m"

        print(f"─── Example {i + 1} ── {status} {'─' * 50}")
        print()
        print(f"  QUESTION:")
        print(f"    {r['question'][:300]}")
        print()
        print(f"  EXPECTED ({expected_movie} / {expected_source}):")
        for line in expected_content[:400].split("\n"):
            print(f"    {line}")
        if len(expected_content) > 400:
            print(f"    ...")
        print()
        print(f"  RETRIEVED:")
        for j, (rid, title, src, content, score) in enumerate(r["retrieved"]):
            match = " ◄◄ MATCH" if rid == r["expected_id"] else ""
            color = "\033[92m" if rid == r["expected_id"] else "\033[0m"
            print(f"    {color}#{j + 1} [sim={score:.3f}] {title} ({src}){match}\033[0m")
            # Show first 200 chars of content, indented
            preview = content[:200].replace("\n", "\n      ")
            print(f"      {preview}")
            if len(content) > 200:
                print(f"      ...")
            print()
        print()

    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
