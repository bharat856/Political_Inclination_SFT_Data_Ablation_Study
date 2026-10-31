import json
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INPUT = Path("data/processed/manifesto/chunks/large_300_manifesto_chunks.csv")
OUT_DIR = Path("data/processed/benchmark_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT = OUT_DIR / "benchmark_v2.csv"
MANIFEST = OUT_DIR / "benchmark_v2.metadata.json"

PER_CLASS = 300
MAX_PER_DOC = 10
SIM_THRESHOLD = 0.85
SEED = 42

def diverse_select_doc(g, max_chunks=10, sim_threshold=0.85):
    g = g.sample(frac=1, random_state=SEED).reset_index(drop=True)

    if len(g) <= 1:
        return g

    texts = g["text"].astype(str).tolist()

    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=5000,
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(texts)

    selected_idx = []

    for i in range(len(g)):
        if len(selected_idx) >= max_chunks:
            break

        if not selected_idx:
            selected_idx.append(i)
            continue

        sims = cosine_similarity(X[i], X[selected_idx]).flatten()

        if sims.max() < sim_threshold:
            selected_idx.append(i)

    return g.iloc[selected_idx].copy()

df = pd.read_csv(INPUT)

df = df[df["rile_label_3way"].isin(["left", "center", "right"])].copy()
df = df[df["word_count"] >= 80].copy()

diverse_docs = []

for doc_id, g in df.groupby("doc_id"):
    diverse_docs.append(diverse_select_doc(g, MAX_PER_DOC, SIM_THRESHOLD))

diverse = pd.concat(diverse_docs, ignore_index=True)

print("Available after diversity filtering:")
print(diverse["rile_label_3way"].value_counts())

sampled = []

for label in ["left", "center", "right"]:
    sub = diverse[diverse["rile_label_3way"] == label]
    n = min(PER_CLASS, len(sub))

    if n < PER_CLASS:
        print(f"WARNING: only {n} chunks available for {label}, requested {PER_CLASS}")

    sampled.append(sub.sample(n=n, random_state=SEED))

bench = pd.concat(sampled, ignore_index=True)
bench = bench.sample(frac=1, random_state=SEED).reset_index(drop=True)
bench.insert(0, "benchmark_id", [f"v2_{i:05d}" for i in range(len(bench))])

bench.to_csv(OUT, index=False)

manifest = {
    "input": str(INPUT),
    "output": str(OUT),
    "rows": int(len(bench)),
    "documents": int(bench["doc_id"].nunique()),
    "per_class_requested": PER_CLASS,
    "max_per_doc": MAX_PER_DOC,
    "similarity_threshold": SIM_THRESHOLD,
    "seed": SEED,
    "label_counts": bench["rile_label_3way"].value_counts().to_dict(),
    "documents_by_label": bench.drop_duplicates("doc_id")["rile_label_3way"].value_counts().to_dict(),
    "chunks_per_doc_summary": bench["doc_id"].value_counts().describe().to_dict(),
}

MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

print("\nSaved:", OUT)
print("Rows:", len(bench))
print("Documents:", bench["doc_id"].nunique())

print("\nLabel distribution:")
print(bench["rile_label_3way"].value_counts())

print("\nDocument label counts:")
print(bench.drop_duplicates("doc_id")["rile_label_3way"].value_counts())

print("\nChunks per document summary:")
print(bench["doc_id"].value_counts().describe())
