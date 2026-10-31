import re
import pandas as pd
from pathlib import Path

ROOT = Path("/fs/scratch/users/bharatpi/causal_political_content")

IN_PATH = ROOT / "data/processed/manifesto/sample_30_manifesto_with_text.csv"
OUT_PATH = ROOT / "data/processed/manifesto/chunks/sample_30_manifesto_chunks.csv"

MIN_WORDS = 80
MAX_WORDS = 250

def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_into_sentences(text: str):
    # Simple multilingual-ish sentence split.
    # Good enough for first benchmark; can improve later.
    parts = re.split(r"(?<=[.!?。！？])\s+", text)
    return [p.strip() for p in parts if p.strip()]

def chunk_text(text: str, min_words=MIN_WORDS, max_words=MAX_WORDS):
    text = clean_text(text)
    sentences = split_into_sentences(text)

    chunks = []
    current = []
    current_words = 0

    for sent in sentences:
        sent_words = sent.split()
        n = len(sent_words)

        if current and current_words + n > max_words:
            chunk = " ".join(current).strip()
            if len(chunk.split()) >= min_words:
                chunks.append(chunk)
            current = []
            current_words = 0

        current.append(sent)
        current_words += n

    if current:
        chunk = " ".join(current).strip()
        if len(chunk.split()) >= min_words:
            chunks.append(chunk)

    # Fallback for very short manifestos
    if not chunks and len(text.split()) > 0:
        chunks.append(text)

    return chunks

def main():
    df = pd.read_csv(IN_PATH)

    rows = []
    for _, row in df.iterrows():
        chunks = chunk_text(row["text"])

        for i, chunk in enumerate(chunks):
            rows.append({
                "doc_id": row["doc_id"],
                "chunk_id": f"{row['doc_id']}_chunk_{i:04d}",
                "chunk_index": i,
                "countryname": row["countryname"],
                "date": row["date"],
                "edate": row["edate"],
                "party": row["party"],
                "partyname": row["partyname"],
                "rile": row["rile"],
                "label_group": row["label_group"],
                "rile_label_3way": row["rile_label_3way"],
                "text": chunk,
                "word_count": len(chunk.split()),
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)

    print("Input docs:", len(df))
    print("Output chunks:", len(out))
    print("Saved:", OUT_PATH)
    print(out[["doc_id", "chunk_id", "rile_label_3way", "word_count"]].head())

if __name__ == "__main__":
    main()
