<div align="center">

# Chunk Hunt

**Which retrieval method finds the right chunk?**
Vector search, BM25, hybrid (RRF) and cross-encoder reranking, benchmarked on the FastAPI docs.

<div align="center">

# Chunk Hunt

**Which retrieval method finds the right chunk?**
Vector search, BM25, hybrid (RRF) and cross-encoder reranking, benchmarked on the FastAPI docs.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Sentence--Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![BM25](https://img.shields.io/badge/rank__bm25-2E8B57?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-4DABCF?style=for-the-badge&logo=numpy&logoColor=white)

</div>

</div>

## Summary

Most RAG demos use vector search and stop there. This project compares four retrievers on the same 334 documentation chunks, using 34 labeled questions, to show where each one succeeds or fails.

- **Hybrid search had the best Hit@5 (0.853).**
- **BM25 beat vector search** on identifiers and exact terms. **Vector search won** on paraphrase and confusable questions.
- **Reranking hurt on this eval set:** Hit@5 fell to 0.735 and CPU latency rose about 400x.

## Method

- **Corpus:** 51 files from the [FastAPI](https://github.com/fastapi/fastapi) tutorial docs, split at `##` headings into 334 chunks.
- **Retrievers:** vector (`BAAI/bge-small-en-v1.5`), BM25 (`rank_bm25`), hybrid (Reciprocal Rank Fusion, k=60), and hybrid plus rerank (`BAAI/bge-reranker-base` over the top 30 candidates).
- **Eval set:** 34 questions, each labeled with the chunk that answers it and tagged by type. [State honestly how it was built, e.g. "drafted with AI assistance from chunks I selected; I checked every label against the source."]
- **Metrics:** Hit@5 (correct chunk in the top 5) and MRR over the top 30 results.

## Results

| Method | Hit@5 | MRR | Latency / query |
|---|---|---|---|
| Vector | 0.765 | 0.632 | 0.08 s |
| BM25 | 0.824 | 0.657 | 0.003 s |
| **Hybrid (RRF)** | **0.853** | 0.652 | 0.08 s |
| Hybrid + rerank | 0.735 | 0.567 | 31.6 s (CPU) |

**Hit@5 by question type**

| Type (n) | Vector | BM25 | Hybrid | Rerank |
|---|---|---|---|---|
| Acronym / identifier (3) | 0.67 | 1.00 | 1.00 | 0.67 |
| Conceptual (8) | 0.62 | 0.88 | 0.88 | 0.75 |
| Confusable (4) | 1.00 | 0.75 | 1.00 | 0.75 |
| Exact term (9) | 0.67 | 0.78 | 0.78 | 0.67 |
| Paraphrase (10) | 0.90 | 0.80 | 0.80 | 0.80 |

Raw per-question results are in [`results/`](results/).

## Failure cases

- **"How do I make a file upload not required?"** BM25 and hybrid ranked the correct chunk 1st; the reranker moved it to 13th. The chunk is short (225 characters), so truncation does not explain it. [Add what the reranker ranked above it.]
- **"What does UploadFile mean?"** Vector search ranked the correct chunk 8th, BM25 ranked it 1st. The chunk is long (2,882 characters), which may dilute its embedding. This is untested. I corrected this question's label after reading the full source file.

Chunk length explains at most two of the six reranker regressions. I have not confirmed a cause for the rest.

## Limitations

- **Small eval set.** One question moves a score by about 3 points, and per-type cells hold 3 to 10 questions, so read them as patterns, not rankings.
- **Vocabulary overlap.** Questions were written while reading the target chunks, which favors keyword search.
- **One label per question.** An equally good answer in another chunk counts as a miss.
- **Latency is a CPU figure.**

## Next steps

Metadata filtering, query routing and hierarchical retrieval are not built. Experiments I'd try: reranking only the top 10 hybrid results, a smaller reranker, and splitting long chunks at `###` headings.

## Run it

```bash
git clone https://github.com/Prakriti308/Chunk-Hunt.git
cd Chunk-Hunt
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python src/chunk.py      # build data/chunks.json
python src/evaluate.py   # run all methods, write results/
python src/analyze.py    # list reranker regressions
```

The first run downloads the embedding model (~130 MB) and the reranker (~1.1 GB).

Corpus: [FastAPI documentation](https://github.com/fastapi/fastapi) by Sebastián Ramírez, MIT license.
