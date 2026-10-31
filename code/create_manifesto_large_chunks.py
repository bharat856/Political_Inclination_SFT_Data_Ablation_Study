import re
import json
import pandas as pd
from pathlib import Path

CORPUS = Path("data/raw/manifesto/corpus/large_300_manifesto_corpus.csv")
META = Path("data/raw/manifesto/manifesto_download_large_300.csv")
OUT_DIR = Path("data/processed/manifesto/chunks")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT = OUT_DIR / "large_300_manifesto_chunks.csv"
MANIFEST = OUT_DIR / "large_300_manifesto_chunks.metadata.json"

CHUNK_WORDS = 220
OVERLAP_WORDS = 40
MIN_WORDS = 80
MAX_CHUNKS_PER_DOC = 30

def normalize_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def make_chunks(text, chunk_words=220, overlap_words=40, min_words=80):
    words = normalize_text(text).split()
    chunks = []
    step = chunk_words - overlap_words

    for start in range(0, len(words), step):
        part = words[start:start + chunk_words]
        if len(part) < min_words:
            continue
        chunks.append(" ".join(part))

    return chunks

corpus = pd.read_csv(CORPUS)
meta = pd.read_csv(META)

# Build doc_id in metadata to match corpus doc_id format.
meta["doc_id"] = meta["party"].astype(str) + "_" + meta["date"].astype(str)

df = corpus.merge(meta, on="doc_id", how="inner", suffixes=("", "_meta"))

print("Corpus rows:", len(corpus))
print("Metadata rows:", len(meta))
print("Merged rows:", len(df))

rows = []

for _, row in df.iterrows():
    chunks = make_chunks(row["text"], CHUNK_WORDS, OVERLAP_WORDS, MIN_WORDS)

    # Limit chunks per document so huge manifestos do not dominate.
    if len(chunks) > MAX_CHUNKS_PER_DOC:
        # Spread chunks across the document instead of only taking the beginning.
        idxs = [
            round(i * (len(chunks) - 1) / (MAX_CHUNKS_PER_DOC - 1))
            for i in range(MAX_CHUNKS_PER_DOC)
        ]
        chunks = [chunks[i] for i in idxs]

    for i, chunk in enumerate(chunks):
        rows.append({
            "chunk_id": f"{row['doc_id']}__chunk_{i:04d}",
            "doc_id": row["doc_id"],
            "chunk_index": i,
            "text": chunk,
            "word_count": len(chunk.split()),
            "country": row.get("country", None),
            "countryname": row.get("countryname", None),
            "party": row.get("party", None),
            "partyname": row.get("partyname", None),
            "date": row.get("date", None),
            "edate": row.get("edate", None),
            "corpusversion": row.get("corpusversion", None),
            "datasetversion": row.get("datasetversion", None),
            "rile": row.get("rile", None),
            "rile_label_3way": row.get("rile_label_3way", None),
        })

chunks_df = pd.DataFrame(rows)
chunks_df.to_csv(OUT, index=False)

manifest = {
    "corpus": str(CORPUS),
    "metadata": str(META),
    "output": str(OUT),
    "documents_merged": int(len(df)),
    "chunks": int(len(chunks_df)),
    "chunk_words": CHUNK_WORDS,
    "overlap_words": OVERLAP_WORDS,
    "min_words": MIN_WORDS,
    "max_chunks_per_doc": MAX_CHUNKS_PER_DOC,
    "label_counts_chunks": chunks_df["rile_label_3way"].value_counts().to_dict(),
    "label_counts_documents": df["rile_label_3way"].value_counts().to_dict(),
}

MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

print("\nSaved:", OUT)
print("Chunks:", len(chunks_df))
print("Documents:", chunks_df["doc_id"].nunique())

print("\nChunk label distribution:")
print(chunks_df["rile_label_3way"].value_counts())

print("\nDocument label distribution:")
print(df["rile_label_3way"].value_counts())

print("\nChunks per document:")
print(chunks_df["doc_id"].value_counts().describe())
