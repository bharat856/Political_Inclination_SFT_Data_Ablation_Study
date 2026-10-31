import re
import pandas as pd
from pathlib import Path

ROOT = Path("/fs/scratch/users/bharatpi/causal_political_content")

IN_PATH = ROOT / "data/processed/manifesto/sample_30_manifesto_with_text.csv"
OUT_PATH = ROOT / "data/processed/manifesto/chunks/sample_30_manifesto_chunks_strict.csv"

MIN_WORDS = 80
MAX_WORDS = 250

def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def word_chunks(text: str, max_words=MAX_WORDS):
    words = clean_text(text).split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def main():
    df = pd.read_csv(IN_PATH)

    rows = []
    for _, row in df.iterrows():
        chunks = word_chunks(row["text"])

        for i, chunk in enumerate(chunks):
            wc = len(chunk.split())
            if wc < MIN_WORDS:
                continue

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
                "word_count": wc,
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)

    print("Input docs:", len(df))
    print("Output chunks:", len(out))
    print("Saved:", OUT_PATH)
    print(out["rile_label_3way"].value_counts())
    print(out["word_count"].describe())

if __name__ == "__main__":
    main()
