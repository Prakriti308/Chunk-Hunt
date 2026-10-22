import json
import pathlib
import time
from collections import defaultdict

import pandas as pd

from retrievers import vector_search, bm25_search, hybrid_search, rerank_search

EVAL_FILE = pathlib.Path("data/eval.json")
RESULTS_DIR = pathlib.Path("results")
DEPTH = 30  # how many results to retrieve per query (reranker pool is 30)

eval_set = json.loads(EVAL_FILE.read_text(encoding="utf-8"))

METHODS = {
    "vector": vector_search,
    "bm25": bm25_search,
    "hybrid": hybrid_search,
    "hybrid+rerank": rerank_search,
}


def first_correct_rank(results, correct):
    """1-based rank of the best-placed correct chunk, or None if missing."""
    ranks = [i + 1 for i, cid in enumerate(results) if cid in correct]
    return min(ranks) if ranks else None


rows = []
for q in eval_set:
    for name, fn in METHODS.items():
        start = time.perf_counter()
        results = fn(q["question"], k=DEPTH)
        elapsed = time.perf_counter() - start
        rank = first_correct_rank(results, q["correct"])
        rows.append({
            "question": q["question"],
            "type": q["type"],
            "method": name,
            "rank": rank,
            "hit@5": int(rank is not None and rank <= 5),
            "rr": (1 / rank) if rank else 0.0,
            "seconds": elapsed,
        })

df = pd.DataFrame(rows)
RESULTS_DIR.mkdir(exist_ok=True)
df.to_csv(RESULTS_DIR / "per_question.csv", index=False)

order = list(METHODS)

overall = (
    df.groupby("method")
    .agg(hit5=("hit@5", "mean"), mrr=("rr", "mean"), avg_seconds=("seconds", "mean"))
    .reindex(order)
    .round(3)
)
print("\n=== Overall (34 questions) ===")
print(overall)
overall.to_csv(RESULTS_DIR / "overall.csv")

by_type = (
    df.pivot_table(index="type", columns="method", values="hit@5", aggfunc="mean")
    [order]
    .round(2)
)
print("\n=== Hit@5 by question type ===")
print(by_type)
by_type.to_csv(RESULTS_DIR / "by_type.csv")

print(f"\nMRR is computed over the top {DEPTH} results (rank beyond {DEPTH} counts as 0).")
