from pathlib import Path
import json
import gzip
import argparse
import csv

import pandas as pd
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


def infer_modernbert_pred(obj):
    for k in [
        "modernbert_pred_binary",
        "pred_binary",
        "binary_pred",
        "pred",
        "prediction",
        "is_political",
        "political_pred",
    ]:
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
        rows.append({"text": text, "label": y, "source": "20newsgroups"})

    print("Loading BBC external training data...", flush=True)
    bbc = pd.read_csv(bbc_path)

    for _, r in bbc.iterrows():
        label = str(r["label_text"]).strip().lower()
        y = 1 if label == "politics" else 0
        rows.append({"text": str(r["text"]), "label": y, "source": "bbc_news"})

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
    parser.add_argument("--skip-rows", type=int, default=0)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    input_jsonl = Path(args.input_jsonl)
    modernbert_jsonl = Path(args.modernbert_jsonl)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"t{str(args.threshold).replace('.', '')}"

    out_candidates_csv = out_dir / f"{args.short_name}_union_{suffix}_candidates_for_4way.csv"
    out_candidates_jsonl = out_dir / f"{args.short_name}_union_{suffix}_candidates_for_4way.jsonl"
    out_summary = out_dir / f"{args.short_name}_embedding_union_gate_summary.json"

    print("=" * 100, flush=True)
    print("DOLCI embedding + ModernBERT union gate STREAMING", flush=True)
    print("=" * 100, flush=True)
    print("Input:", input_jsonl, flush=True)
    print("ModernBERT:", modernbert_jsonl, flush=True)
    print("Output:", out_dir, flush=True)
    print("Threshold:", args.threshold, flush=True)

    row_to_mb = load_modernbert_predictions(modernbert_jsonl)
    embedder, clf = train_embedding_classifier(args.embedding_model, args.bbc_path, args.batch_size)

    counts = {
        "rows": 0,
        "old_modernbert_candidate": 0,
        f"embedding_candidate_{suffix}": 0,
        f"union_{suffix}": 0,
        f"both_{suffix}": 0,
        f"embedding_only_{suffix}": 0,
        f"modernbert_only_{suffix}": 0,
        f"neither_{suffix}": 0,
    }

    fieldnames = [
        "row_id",
        "dataset",
        "dataset_short_name",
        "dataset_source",
        "old_modernbert_candidate",
        "embedding_prob_political",
        f"embedding_candidate_{suffix}",
        f"union_{suffix}",
        f"both_{suffix}",
        f"embedding_only_{suffix}",
        f"modernbert_only_{suffix}",
        "text",
    ]

    print("Applying embedding gate to DOLCI rows...", flush=True)

    buffer = []

    csv_mode = "a" if args.append else "w"
    jsonl_mode = "a" if args.append else "w"

    with open(out_candidates_csv, csv_mode, encoding="utf-8", newline="") as csv_f, \
         open(out_candidates_jsonl, jsonl_mode, encoding="utf-8") as jsonl_f:

        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        if not args.append:
            writer.writeheader()

        def process_buffer(buf):
            if not buf:
                return

            texts = [str(r.get("text", ""))[:args.max_chars] for r in buf]

            X = embedder.encode(
                texts,
                batch_size=args.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            probs = clf.predict_proba(X)[:, 1]

            for obj, prob in zip(buf, probs):
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
                counts[f"embedding_candidate_{suffix}"] += emb
                counts[f"union_{suffix}"] += union
                counts[f"both_{suffix}"] += both
                counts[f"embedding_only_{suffix}"] += emb_only
                counts[f"modernbert_only_{suffix}"] += mb_only
                counts[f"neither_{suffix}"] += neither

                if union == 1:
                    row = {
                        "row_id": row_id,
                        "dataset": obj.get("dataset"),
                        "dataset_short_name": obj.get("dataset_short_name"),
                        "dataset_source": obj.get("dataset_source"),
                        "old_modernbert_candidate": mb,
                        "embedding_prob_political": float(prob),
                        f"embedding_candidate_{suffix}": emb,
                        f"union_{suffix}": union,
                        f"both_{suffix}": both,
                        f"embedding_only_{suffix}": emb_only,
                        f"modernbert_only_{suffix}": mb_only,
                        "text": obj.get("text", ""),
                    }

                    writer.writerow(row)
                    jsonl_f.write(json.dumps(row, ensure_ascii=False) + "\n")

        skipped = 0
        if args.skip_rows:
            print(f"Skipping first {args.skip_rows} rows before resuming...", flush=True)

        for obj in tqdm(read_jsonl(input_jsonl), desc=args.short_name):
            if skipped < args.skip_rows:
                skipped += 1
                continue

            buffer.append(obj)
            if len(buffer) >= args.chunk_size:
                process_buffer(buffer)
                buffer = []

        process_buffer(buffer)

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
            "union_candidates_csv": str(out_candidates_csv),
            "union_candidates_jsonl": str(out_candidates_jsonl),
        },
    }

    out_summary.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2), flush=True)
    print("Saved:", out_candidates_csv, flush=True)
    print("Saved:", out_candidates_jsonl, flush=True)
    print("Saved:", out_summary, flush=True)


if __name__ == "__main__":
    main()
