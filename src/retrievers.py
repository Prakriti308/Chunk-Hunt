import json
import pathlib
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

CHUNKS_FILE = pathlib.Path("data/chunks.json")
EMB_FILE = pathlib.Path("data/chunk_embeddings.npy")

# ---------- Load chunks ----------
chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
chunk_ids = [c["id"] for c in chunks]
chunk_texts = [c["text"] for c in chunks]

# ---------- Vector search setup ----------
print("Loading embedding model...")
embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

if EMB_FILE.exists():
    chunk_embeddings = np.load(EMB_FILE)
else:
    print(f"Embedding {len(chunk_texts)} chunks...")
    chunk_embeddings = embed_model.encode(
        chunk_texts, normalize_embeddings=True, show_progress_bar=True
    )
    np.save(EMB_FILE, chunk_embeddings)


def vector_search(query: str, k: int = 10):
    """Return top-k chunk IDs ranked by embedding similarity to the query."""
    query_embedding = embed_model.encode([query], normalize_embeddings=True)[0]
    scores = chunk_embeddings @ query_embedding
    top_indices = np.argsort(-scores)[:k]
    return [chunk_ids[i] for i in top_indices]


# ---------- BM25 setup ----------
def tokenize(text: str):
    return re.findall(r"\w+", text.lower())


bm25 = BM25Okapi([tokenize(t) for t in chunk_texts])


def bm25_search(query: str, k: int = 10):
    """Return top-k chunk IDs ranked by BM25 keyword score."""
    scores = bm25.get_scores(tokenize(query))
    top_indices = np.argsort(-scores)[:k]
    return [chunk_ids[i] for i in top_indices]

def hybrid_search(query: str, k: int = 10, pool: int = 50, rrf_k: int = 60):
    """Merge vector and BM25 rankings with Reciprocal Rank Fusion."""
    rankings = [
        vector_search(query, k=pool),
        bm25_search(query, k=pool),
    ]
    scores = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0) + 1 / (rrf_k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)[:k]

from sentence_transformers import CrossEncoder

print("Loading reranker...")
reranker = CrossEncoder("BAAI/bge-reranker-base")


def rerank_search(query: str, k: int = 10, pool: int = 30):
    """Take top hybrid candidates, then re-score each (query, chunk) pair."""
    candidates = hybrid_search(query, k=pool)
    id_to_text = dict(zip(chunk_ids, chunk_texts))
    pairs = [(query, id_to_text[cid]) for cid in candidates]
    scores = reranker.predict(pairs)
    order = np.argsort(-scores)
    return [candidates[i] for i in order][:k]


# ---------- Quick manual test ----------
if __name__ == "__main__":
    test_query = "What does UploadFile mean in FastAPI?"
    correct = "request-files-3"

    vec = vector_search(test_query, k=len(chunk_ids))
    bm = bm25_search(test_query, k=len(chunk_ids))

    print(f"\nQuery: {test_query}")
    print("Vector top 5:", vec[:5])
    print("BM25 top 5:  ", bm[:5])
    print(f"Vector rank of {correct}: {vec.index(correct) + 1} of {len(vec)}")
    print(f"BM25 rank of {correct}:   {bm.index(correct) + 1} of {len(bm)}")
    hyb = hybrid_search(test_query, k=len(chunk_ids))
    print("Hybrid top 5:", hyb[:5])
    print(f"Hybrid rank of {correct}: {hyb.index(correct) + 1 if correct in hyb else 'not in pool'}")
    rr = rerank_search(test_query, k=30)
    print("Rerank top 5:", rr[:5])
    print(f"Rerank rank of {correct}: {rr.index(correct) + 1 if correct in rr else 'not in pool'}")