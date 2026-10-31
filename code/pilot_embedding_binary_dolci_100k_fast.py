from pathlib import Path
import json
import gzip
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_20newsgroups

RANDOM_SEED = 42
N_SAMPLE = 100_000
EMBEDDING_THRESHOLD = 0.3
MODEL_NAME = "BAAI/bge-base-en-v1.5"

OUT_DIR = Path("outputs/dolci_olmo3_sft_embedding_binary_pilot_100k")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOLCI_FILE = Path("data/processed/dolci_olmo3_sft/dolci_instruct_sft_7b_flattened.jsonl")

OLD_MODERNBERT_FILES = [
    Path("outputs/dolci_olmo3_sft_modernbert_combined/dolci_instruct_sft_7b_predictions.jsonl"),
    Path("outputs/dolci_olmo3_sft_modernbert_combined/dolci_think_sft_7b_predictions.jsonl"),
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


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_first_n_dolci(path, n):
    rows = []
    for i, obj in enumerate(read_jsonl(path)):
        if i >= n:
            break
        rows.append({
            "row_id": obj.get("row_id"),
            "dataset": obj.get("dataset"),
            "dataset_short_name": obj.get("dataset_short_name"),
            "dataset_source": obj.get("dataset_source"),
            "text": str(obj.get("text", "")),
        })
    return pd.DataFrame(rows)


def infer_pred_value(obj):
    """
    Robustly detect whether old ModernBERT predicted political.
    """
    possible_pred_keys = [
        "modernbert_pred_binary",
        "pred_binary",
        "binary_pred",
        "pred",
        "prediction",
        "is_political",
        "political_pred",
    ]

    for k in possible_pred_keys:
        if k in obj:
            try:
                return int(obj[k])
            except Exception:
                v = str(obj[k]).strip().lower()
                if v in {"1", "true", "political"}:
                    return 1
                if v in {"0", "false", "non_political", "non-political"}:
                    return 0

    # If this JSONL is candidates-only, every row may be political.
    # But the prediction files should contain a pred key. If not, print once later.
    return None


def load_old_modernbert_candidate_ids():
    ids = set()
    missing_pred_keys = 0
    total = 0

    for path in OLD_MODERNBERT_FILES:
        print("Loading old ModernBERT file:", path, flush=True)
        if not path.exists():
            print("Missing:", path, flush=True)
            continue

        for obj in read_jsonl(path):
            total += 1
            row_id = obj.get("row_id")
            pred = infer_pred_value(obj)

            if pred is None:
                missing_pred_keys += 1
                continue

            if pred == 1:
                ids.add(row_id)

    print("Old ModernBERT total rows seen:", total, flush=True)
    print("Old ModernBERT rows with missing pred key:", missing_pred_keys, flush=True)
    print("Old ModernBERT candidate IDs:", len(ids), flush=True)

    if len(ids) == 0:
        print("WARNING: No ModernBERT candidates found from prediction JSONL.", flush=True)
        print("Inspect keys with:", flush=True)
        print("head -n 1 outputs/dolci_olmo3_sft_modernbert_combined/dolci_instruct_sft_7b_predictions.jsonl", flush=True)

    return ids


def load_training_data():
    rows = []

    print("Loading 20NG external binary data...", flush=True)
    ng = fetch_20newsgroups(
        subset="test",
        categories=ALL_20NG,
        remove=("headers", "footers", "quotes"),
    )

    for text, target in zip(ng.data, ng.target):
        cat = ng.target_names[target]
        y = 1 if cat in POLITICAL_20NG else 0
        rows.append({"text": text, "gold_binary": y, "source": "20newsgroups"})

    print("Loading BBC external binary data...", flush=True)
    bbc = pd.read_csv(BBC_PATH)
    for _, r in bbc.iterrows():
        label = str(r["label_text"]).strip().lower()
        y = 1 if label == "politics" else 0
        rows.append({"text": str(r["text"]), "gold_binary": y, "source": "bbc_news"})

    df = pd.DataFrame(rows)
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.len() > 0].copy()

    print("Training rows:", len(df), flush=True)
    print(df["gold_binary"].value_counts(), flush=True)
    print(df["source"].value_counts(), flush=True)

    return df


def main():
    print("=" * 100, flush=True)
    print("Fast DOLCI first-100k embedding binary pilot", flush=True)
    print("=" * 100, flush=True)

    print("Loading first 100k DOLCI rows from:", DOLCI_FILE, flush=True)
    pilot = load_first_n_dolci(DOLCI_FILE, N_SAMPLE)

    pilot["text"] = pilot["text"].fillna("").astype(str)
    pilot["text_len_chars"] = pilot["text"].str.len()

    print("Pilot rows:", len(pilot), flush=True)
    print("Dataset distribution:", flush=True)
    print(pilot["dataset_short_name"].value_counts(dropna=False), flush=True)
    print("Text length:", flush=True)
    print(pilot["text_len_chars"].describe(), flush=True)

    old_mb_ids = load_old_modernbert_candidate_ids()
    pilot["old_modernbert_candidate"] = pilot["row_id"].isin(old_mb_ids).astype(int)

    train_df = load_training_data()

    print("Loading embedding model:", MODEL_NAME, flush=True)
    embedder = SentenceTransformer(MODEL_NAME)

    print("Encoding training data...", flush=True)
    X_train = embedder.encode(
        train_df["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    y_train = train_df["gold_binary"].astype(int).to_numpy()

    print("Training logistic regression...", flush=True)
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)

    print("Encoding DOLCI pilot...", flush=True)
    X_pilot = embedder.encode(
        pilot["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    probs = clf.predict_proba(X_pilot)[:, 1]
    pilot["embedding_prob_political"] = probs
    pilot["embedding_candidate_t03"] = (probs >= EMBEDDING_THRESHOLD).astype(int)
    pilot["embedding_candidate_t04"] = (probs >= 0.4).astype(int)
    pilot["embedding_candidate_t05"] = (probs >= 0.5).astype(int)

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

    counts = {
        "old_modernbert_candidate": int(pilot["old_modernbert_candidate"].sum()),
        "embedding_candidate_t03": int(pilot["embedding_candidate_t03"].sum()),
        "embedding_candidate_t04": int(pilot["embedding_candidate_t04"].sum()),
        "embedding_candidate_t05": int(pilot["embedding_candidate_t05"].sum()),
        "union_t03": int(pilot["union_t03"].sum()),
        "both_t03": int(pilot["both_t03"].sum()),
        "embedding_only_t03": int(pilot["embedding_only_t03"].sum()),
        "modernbert_only_t03": int(pilot["modernbert_only_t03"].sum()),
        "neither_t03": int(((pilot["old_modernbert_candidate"] == 0) & (pilot["embedding_candidate_t03"] == 0)).sum()),
    }

    summary = {
        "sample_type": "first_100k_instruct",
        "sample_rows": int(len(pilot)),
        "embedding_model": MODEL_NAME,
        "embedding_threshold": EMBEDDING_THRESHOLD,
        "counts": counts,
        "rates": {k: float(v / len(pilot)) for k, v in counts.items()},
    }

    print("\n" + "=" * 100, flush=True)
    print("Pilot comparison summary", flush=True)
    print("=" * 100, flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    out_csv = OUT_DIR / "dolci_first100k_embedding_modernbert_gate_comparison.csv"
    out_json = OUT_DIR / "dolci_first100k_embedding_modernbert_gate_comparison_summary.json"

    pilot.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(summary, indent=2))

    print("Saved:", out_csv, flush=True)
    print("Saved:", out_json, flush=True)

    buckets = {
        "embedding_only_t03": pilot[pilot["embedding_only_t03"] == 1],
        "modernbert_only_t03": pilot[pilot["modernbert_only_t03"] == 1],
        "both_t03": pilot[pilot["both_t03"] == 1],
        "neither_t03": pilot[(pilot["old_modernbert_candidate"] == 0) & (pilot["embedding_candidate_t03"] == 0)],
    }

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
            "embedding_candidate_t04",
            "embedding_candidate_t05",
            "union_t03",
            "text",
            "manual_binary_label",
            "manual_notes",
        ]

        out = OUT_DIR / f"manual_inspection_{name}_n{n}.csv"
        sampled[cols].to_csv(out, index=False)
        print("Saved manual inspection sample:", out, flush=True)


if __name__ == "__main__":
    main()
