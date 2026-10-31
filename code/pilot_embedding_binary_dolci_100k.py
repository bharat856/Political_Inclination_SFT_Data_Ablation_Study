from pathlib import Path
import json
import gzip
import random
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

# ----------------------------
# Config
# ----------------------------

RANDOM_SEED = 42
N_SAMPLE = 100_000
EMBEDDING_THRESHOLD = 0.3
MODEL_NAME = "BAAI/bge-base-en-v1.5"

OUT_DIR = Path("outputs/dolci_olmo3_sft_embedding_binary_pilot_100k")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOLCI_FILES = [
    Path("data/processed/dolci_olmo3_sft/dolci_instruct_sft_7b_flattened.jsonl"),
    Path("data/processed/dolci_olmo3_sft/dolci_think_sft_7b_flattened.jsonl"),
]

OLD_MODERNBERT_CANDIDATES = [
    Path("outputs/dolci_olmo3_sft_modernbert_combined/binary_outputs/full_instruct/dolci_instruct_sft_7b_flattened_modernbert_binary_predictions.csv"),
    Path("outputs/dolci_olmo3_sft_modernbert_combined/binary_outputs/full_think/dolci_think_sft_7b_flattened_modernbert_binary_predictions.csv"),
]

BBC_PATH = Path("data/validation_external/bbc_news.csv")

POLITICAL_20NG = {
    "talk.politics.guns",
    "talk.politics.mideast",
    "talk.politics.misc",
}

ALL_20NG = [
    "alt.atheism",
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.windows.x",
    "misc.forsale",
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.crypt",
    "sci.electronics",
    "sci.med",
    "sci.space",
    "soc.religion.christian",
    "talk.politics.guns",
    "talk.politics.mideast",
    "talk.politics.misc",
]


# ----------------------------
# Helpers
# ----------------------------

def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def reservoir_sample_jsonl(files, n, seed=42):
    rng = random.Random(seed)
    sample = []
    seen = 0

    for path in files:
        print("Sampling from:", path)
        for obj in read_jsonl(path):
            seen += 1
            if len(sample) < n:
                sample.append(obj)
            else:
                j = rng.randint(0, seen - 1)
                if j < n:
                    sample[j] = obj

    print("Seen rows:", seen)
    print("Sampled rows:", len(sample))
    return sample


def find_old_modernbert_files():
    existing = [p for p in OLD_MODERNBERT_CANDIDATES if p.exists()]
    if existing:
        return existing

    # Fallback: search likely outputs.
    candidates = list(Path("outputs").glob("**/*modernbert*binary*.csv"))
    print("Could not find expected ModernBERT files.")
    print("Candidate files:")
    for p in candidates[:50]:
        print(" ", p)
    raise FileNotFoundError("Could not find old ModernBERT DOLCI binary prediction CSVs.")


def load_old_modernbert_candidate_ids():
    files = find_old_modernbert_files()
    ids = set()

    for p in files:
        print("Loading old ModernBERT candidates from:", p)
        df = pd.read_csv(p)
        print("Columns:", df.columns.tolist())
        print("Rows:", len(df))

        if "row_id" not in df.columns:
            raise ValueError(f"No row_id column in {p}")

        pred_col = None
        for c in ["modernbert_pred_binary", "pred_binary", "binary_pred", "pred"]:
            if c in df.columns:
                pred_col = c
                break

        if pred_col is None:
            pred_cols = [c for c in df.columns if "pred" in c.lower()]
            if pred_cols:
                pred_col = pred_cols[0]
            else:
                raise ValueError(f"No prediction column found in {p}")

        cand = df[df[pred_col].astype(int) == 1]["row_id"].tolist()
        ids.update(cand)
        print("Candidates added:", len(cand))

    print("Total old ModernBERT candidate IDs:", len(ids))
    return ids


def load_training_data():
    from sklearn.datasets import fetch_20newsgroups

    rows = []

    print("Loading 20NG training data...")
    ng = fetch_20newsgroups(
        subset="test",
        categories=ALL_20NG,
        remove=("headers", "footers", "quotes"),
    )

    for text, target in zip(ng.data, ng.target):
        cat = ng.target_names[target]
        y = 1 if cat in POLITICAL_20NG else 0
        rows.append({"text": text, "gold_binary": y, "source": "20newsgroups"})

    print("Loading BBC training data...")
    bbc = pd.read_csv(BBC_PATH)
    for _, r in bbc.iterrows():
        label = str(r["label_text"]).strip().lower()
        y = 1 if label == "politics" else 0
        rows.append({"text": str(r["text"]), "gold_binary": y, "source": "bbc_news"})

    df = pd.DataFrame(rows)
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.len() > 0].copy()

    print("Training rows:", len(df))
    print(df["gold_binary"].value_counts())
    print(df["source"].value_counts())

    return df


# ----------------------------
# Main
# ----------------------------

def main():
    print("=" * 100)
    print("DOLCI 100k embedding binary pilot")
    print("=" * 100)

    # 1. Sample DOLCI
    sample = reservoir_sample_jsonl(DOLCI_FILES, N_SAMPLE, RANDOM_SEED)

    pilot_rows = []
    for obj in sample:
        pilot_rows.append({
            "row_id": obj.get("row_id"),
            "dataset": obj.get("dataset"),
            "dataset_short_name": obj.get("dataset_short_name"),
            "dataset_source": obj.get("dataset_source"),
            "text": str(obj.get("text", "")),
        })

    pilot = pd.DataFrame(pilot_rows)
    pilot["text"] = pilot["text"].fillna("").astype(str)
    pilot["text_len_chars"] = pilot["text"].str.len()

    print("Pilot rows:", len(pilot))
    print("Dataset distribution:")
    print(pilot["dataset_short_name"].value_counts(dropna=False))
    print("Text length:")
    print(pilot["text_len_chars"].describe())

    # 2. Old ModernBERT candidates
    old_mb_ids = load_old_modernbert_candidate_ids()
    pilot["old_modernbert_candidate"] = pilot["row_id"].isin(old_mb_ids).astype(int)

    # 3. Train embedding classifier on external binary data
    train_df = load_training_data()

    print("Loading embedding model:", MODEL_NAME)
    embedder = SentenceTransformer(MODEL_NAME)

    print("Encoding training data...")
    X_train = embedder.encode(
        train_df["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    y_train = train_df["gold_binary"].astype(int).to_numpy()

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)

    # 4. Predict DOLCI pilot
    print("Encoding DOLCI pilot...")
    X_pilot = embedder.encode(
        pilot["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    probs = clf.predict_proba(X_pilot)[:, 1]
    pilot["embedding_prob_political"] = probs
    pilot["embedding_candidate_t03"] = (probs >= EMBEDDING_THRESHOLD).astype(int)
    pilot["embedding_candidate_t05"] = (probs >= 0.5).astype(int)

    # 5. Compare gates
    pilot["union_t03"] = (
        (pilot["old_modernbert_candidate"] == 1)
        | (pilot["embedding_candidate_t03"] == 1)
    ).astype(int)

    pilot["both_t03"] = (
        (pilot["old_modernbert_candidate"] == 1)
        & (pilot["embedding_candidate_t03"] == 1)
    ).astype(int)

    pilot["embedding_only_t03"] = (
        (pilot["old_modernbert_candidate"] == 0)
        & (pilot["embedding_candidate_t03"] == 1)
    ).astype(int)

    pilot["modernbert_only_t03"] = (
        (pilot["old_modernbert_candidate"] == 1)
        & (pilot["embedding_candidate_t03"] == 0)
    ).astype(int)

    # 6. Summary
    summary = {
        "sample_rows": int(len(pilot)),
        "random_seed": RANDOM_SEED,
        "embedding_model": MODEL_NAME,
        "embedding_threshold": EMBEDDING_THRESHOLD,
        "counts": {
            "old_modernbert_candidate": int(pilot["old_modernbert_candidate"].sum()),
            "embedding_candidate_t03": int(pilot["embedding_candidate_t03"].sum()),
            "embedding_candidate_t05": int(pilot["embedding_candidate_t05"].sum()),
            "union_t03": int(pilot["union_t03"].sum()),
            "both_t03": int(pilot["both_t03"].sum()),
            "embedding_only_t03": int(pilot["embedding_only_t03"].sum()),
            "modernbert_only_t03": int(pilot["modernbert_only_t03"].sum()),
            "neither_t03": int(((pilot["old_modernbert_candidate"] == 0) & (pilot["embedding_candidate_t03"] == 0)).sum()),
        },
        "rates": {},
    }

    for k, v in summary["counts"].items():
        summary["rates"][k] = float(v / len(pilot))

    print("\n" + "=" * 100)
    print("Pilot comparison summary")
    print("=" * 100)
    print(json.dumps(summary, indent=2))

    # 7. Save full pilot
    out_csv = OUT_DIR / "dolci_100k_embedding_modernbert_gate_comparison.csv"
    pilot.to_csv(out_csv, index=False)

    out_json = OUT_DIR / "dolci_100k_embedding_modernbert_gate_comparison_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))

    print("Saved:", out_csv)
    print("Saved:", out_json)

    # 8. Save manual inspection samples
    buckets = {
        "embedding_only_t03": pilot[pilot["embedding_only_t03"] == 1],
        "modernbert_only_t03": pilot[pilot["modernbert_only_t03"] == 1],
        "both_t03": pilot[pilot["both_t03"] == 1],
        "neither_t03": pilot[(pilot["old_modernbert_candidate"] == 0) & (pilot["embedding_candidate_t03"] == 0)],
    }

    rng = np.random.default_rng(RANDOM_SEED)

    for name, bdf in buckets.items():
        n = min(50, len(bdf))
        if n == 0:
            continue

        sampled = bdf.sample(n=n, random_state=RANDOM_SEED).copy()
        sampled["manual_binary_label"] = ""
        sampled["manual_notes"] = ""

        cols = [
            "row_id",
            "dataset_short_name",
            "old_modernbert_candidate",
            "embedding_prob_political",
            "embedding_candidate_t03",
            "embedding_candidate_t05",
            "union_t03",
            "text",
            "manual_binary_label",
            "manual_notes",
        ]

        out = OUT_DIR / f"manual_inspection_{name}_n{n}.csv"
        sampled[cols].to_csv(out, index=False)
        print("Saved manual inspection sample:", out)


if __name__ == "__main__":
    main()
