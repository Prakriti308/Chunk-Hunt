import json
import pandas as pd

df = pd.read_csv("results/per_question.csv")
order = ["vector", "bm25", "hybrid", "hybrid+rerank"]
piv = df.pivot(index="question", columns="method", values="rank")[order]

chunks = {c["id"]: c["text"] for c in json.load(open("data/chunks.json", encoding="utf-8"))}
ev = {q["question"]: q["correct"] for q in json.load(open("data/eval.json", encoding="utf-8"))}
piv["chars"] = [max(len(chunks[c]) for c in ev[q]) for q in piv.index]

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)

reg = piv[piv["hybrid"].le(5) & ~piv["hybrid+rerank"].le(5)].copy()
reg.index = [q[:55] for q in reg.index]
print("Reranker regressions (hybrid hit top 5, reranker missed):")
print(reg)

print("\nChunk length in characters:")
print("  median, all chunks:        ", int(pd.Series([len(t) for t in chunks.values()]).median()))
print("  median, regression chunks: ", int(reg["chars"].median()))
print("  median, all labeled chunks:", int(piv["chars"].median()))