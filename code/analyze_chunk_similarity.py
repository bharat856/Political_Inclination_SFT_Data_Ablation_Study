import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INPUT = Path("data/processed/manifesto/chunks/large_300_manifesto_chunks.csv")
OUT_DIR = Path("data/processed/manifesto/chunks/similarity_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)
df["text"] = df["text"].astype(str)

rows = []

for doc_id, g in df.groupby("doc_id"):
    if len(g) < 2:
        continue

    g = g.sort_values("chunk_index")
    texts = g["text"].tolist()

    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=5000,
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(texts)
    sim = cosine_similarity(X)

    adjacent_sims = []
    for i in range(len(g) - 1):
        adjacent_sims.append(sim[i, i + 1])

    max_sims = []
    for i in range(len(g)):
        row = sim[i].copy()
        row[i] = -1
        max_sims.append(row.max())

    rows.append({
        "doc_id": doc_id,
        "n_chunks": len(g),
        "label": g["rile_label_3way"].iloc[0],
        "countryname": g["countryname"].iloc[0],
        "partyname": g["partyname"].iloc[0],
        "mean_adjacent_similarity": sum(adjacent_sims) / len(adjacent_sims),
        "max_adjacent_similarity": max(adjacent_sims),
        "mean_max_similarity": sum(max_sims) / len(max_sims),
        "max_similarity": max(max_sims),
    })

summary = pd.DataFrame(rows)

out = OUT_DIR / "chunk_similarity_by_document.csv"
summary.to_csv(out, index=False)

print("Saved:", out)
print("\nOverall similarity summary:")
print(summary[[
    "n_chunks",
    "mean_adjacent_similarity",
    "max_adjacent_similarity",
    "mean_max_similarity",
    "max_similarity"
]].describe())

print("\nMost repetitive documents:")
print(
    summary.sort_values("mean_max_similarity", ascending=False)
    .head(20)[[
        "doc_id", "label", "n_chunks", "mean_adjacent_similarity",
        "mean_max_similarity", "max_similarity", "countryname", "partyname"
    ]]
    .to_string(index=False)
)
