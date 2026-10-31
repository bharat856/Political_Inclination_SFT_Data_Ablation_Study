from pathlib import Path
import json
import gzip
import argparse
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_20newsgroups
from tqdm import tqdm


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


def write_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def infer_modernbert_pred(obj):
    possible_keys = [
        "modernbert_pred_binary",
        "pred_binary",
        "binary_pred",
        "pred",
        "prediction",
        "is_political",
        "political_pred",
    ]

    for k in possible_keys:
        if k in obj:
            v = obj[k]
            try:
                return int(v)
            except Exception:
                s = str(v).strip().lower()
                if s in {"1", "true", "political"}:
                    return 1
                if s in {"0", "false", "non_political", "non-political", "apolitical"}:
                    return 0

    return None


def load_modernbert_predictions(path):
    """
    Returns:
      row_id -> old_modernbert_candidate
    """
    row_to_pred = {}
    missing = 0
    total = 0

    print(f"Loading ModernBERT predictions: {path}", flush=True)

    for obj in tqdm(read_jsonl(path), desc="ModernBERT rows"):
        total += 1
        row_id = obj.get("row_id")
        pred = infer_modernbert_pred(obj)

        if row_id is None:
            continue

        if pred is None:
            missing += 1
            continue

        row_to_pred[row_id] = int(pred)

    print("ModernBERT rows loaded:", len(row_to_pred), flush=True)
    print("ModernBERT missing pred:", missing, "/", total, flush=True)

    return row_to_pred


def load_external_train(bbc_path):
    rows = []

    print("Loading 20 Newsgroups external training data...", flush=True)
    ng = fetch_20newsgroups(
        subset="test",
        categories=ALL_20NG,
        remove=("headers", "footers", "quotes"),
    )

    for text, target in zip(ng.data, ng.target):
        category = ng.target_names[target]
        y = 1 if category in POLITICAL_20NG else 0
        rows.append({
            "text": text,
            "label": y,
            "source": "20newsgroups",
        })

    print("Loading BBC external training data...", flush=True)
    bbc = pd.read_csv(bbc_path)

    for _, r in bbc.iterrows():
        label = str(r["label_text"]).strip().lower()
        y = 1 if label == "politics" else 0
        rows.append({
            "text": str(r["text"]),
            "label": y,
            "source": "bbc_news",
        })

    df = pd.DataFrame(rows)
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.len() > 0].copy()

    print("External training rows:", len(df), flush=True)
    print(df["label"].value_counts(), flush=True)
    print(df["source"].value_counts(), flush=True)

    return df


def train_embedding_classifier(model_name, bbc_path, batch_size):
    train_df = load_external_train(bbc_path)

    print("Loading embedding model:", model_name, flush=True)
    embedder = SentenceTransformer(model_name)

    print("Encoding external training data...", flush=True)
    X = embedder.encode(
        train_df["text"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    y = train_df["label"].astype(int).to_numpy()

    print("Training logistic regression...", flush=True)
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(X, y)

    return embedder, clf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--modernbert-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--short-name", required=True)
    parser.add_argument("--bbc-path", default="data/validation_external/bbc_news.csv")
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument("--max-chars", type=int, default=6000)
    args = parser.parse_args()

    input_jsonl = Path(args.input_jsonl)
    modernbert_jsonl = Path(args.modernbert_jsonl)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_all_csv = out_dir / f"{args.short_name}_embedding_union_gate_all_rows.csv"
    out_candidates_csv = out_dir / f"{args.short_name}_union_t04_candidates_for_4way.csv"
    out_candidates_jsonl = out_dir / f"{args.short_name}_union_t04_candidates_for_4way.jsonl"
    out_summary = out_dir / f"{args.short_name}_embedding_union_gate_summary.json"

    print("=" * 100, flush=True)
    print("DOLCI embedding + ModernBERT union gate", flush=True)
    print("=" * 100, flush=True)
    print("Input:", input_jsonl, flush=True)
    print("ModernBERT:", modernbert_jsonl, flush=True)
    print("Output:", out_dir, flush=True)
    print("Threshold:", args.threshold, flush=True)

    row_to_mb = load_modernbert_predictions(modernbert_jsonl)
    embedder, clf = train_embedding_classifier(
        args.embedding_model,
        args.bbc_path,
        args.batch_size,
    )

    all_rows = []
    candidate_rows = []

    counts = {
        "rows": 0,
        "old_modernbert_candidate": 0,
        "embedding_candidate_t04": 0,
        "union_t04": 0,
        "both_t04": 0,
        "embedding_only_t04": 0,
        "modernbert_only_t04": 0,
        "neither_t04": 0,
    }

    buffer = []

    def process_buffer(buffer):
        if not buffer:
            return

        texts = [
            str(r.get("text", ""))[:args.max_chars]
            for r in buffer
        ]

        X = embedder.encode(
            texts,
            batch_size=args.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        probs = clf.predict_proba(X)[:, 1]

        for obj, prob in zip(buffer, probs):
            row_id = obj.get("row_id")
            mb = int(row_to_mb.get(row_id, 0))
            emb = int(prob >= args.threshold)
            union = int((mb == 1) or (emb == 1))

            both = int(mb == 1 and emb == 1)
            emb_only = int(mb == 0 and emb == 1)
            mb_only = int(mb == 1 and emb == 0)
            neither = int(mb == 0 and emb == 0)

            counts["rows"] += 1
            counts["old_modernbert_candidate"] += mb
            counts["embedding_candidate_t04"] += emb
            counts["union_t04"] += union
            counts["both_t04"] += both
            counts["embedding_only_t04"] += emb_only
            counts["modernbert_only_t04"] += mb_only
            counts["neither_t04"] += neither

            row = {
                "row_id": row_id,
                "dataset": obj.get("dataset"),
                "dataset_short_name": obj.get("dataset_short_name"),
                "dataset_source": obj.get("dataset_source"),
                "old_modernbert_candidate": mb,
                "embedding_prob_political": float(prob),
                "embedding_candidate_t04": emb,
                "union_t04": union,
                "both_t04": both,
                "embedding_only_t04": emb_only,
                "modernbert_only_t04": mb_only,
                "text": obj.get("text", ""),
            }

            all_rows.append(row)

            if union == 1:
                candidate_rows.append(row)

    print("Applying embedding gate to DOLCI rows...", flush=True)

    for obj in tqdm(read_jsonl(input_jsonl), desc=args.short_name):
        buffer.append(obj)

        if len(buffer) >= args.chunk_size:
            process_buffer(buffer)
            buffer = []

    process_buffer(buffer)

    print("Writing outputs...", flush=True)

    all_df = pd.DataFrame(all_rows)
    cand_df = pd.DataFrame(candidate_rows)

    all_df.to_csv(out_all_csv, index=False)
    cand_df.to_csv(out_candidates_csv, index=False)
    write_jsonl(candidate_rows, out_candidates_jsonl)

    rates = {
        k: (v / counts["rows"] if k != "rows" and counts["rows"] else v)
        for k, v in counts.items()
    }

    summary = {
        "input_jsonl": str(input_jsonl),
        "modernbert_jsonl": str(modernbert_jsonl),
        "short_name": args.short_name,
        "embedding_model": args.embedding_model,
        "threshold": args.threshold,
        "max_chars": args.max_chars,
        "counts": counts,
        "rates": rates,
        "outputs": {
            "all_rows_csv": str(out_all_csv),
            "union_candidates_csv": str(out_candidates_csv),
            "union_candidates_jsonl": str(out_candidates_jsonl),
        },
    }

    out_summary.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2), flush=True)
    print("Saved:", out_all_csv, flush=True)
    print("Saved:", out_candidates_csv, flush=True)
    print("Saved:", out_candidates_jsonl, flush=True)
    print("Saved:", out_summary, flush=True)


if __name__ == "__main__":
    main()
