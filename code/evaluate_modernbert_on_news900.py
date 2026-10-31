from pathlib import Path
import argparse
import json

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


DEFAULT_MODEL = Path("outputs/target_domain_binary_modernbert_combined/best_model")


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if path.suffix.lower() in [".tsv"]:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def find_text_column(df: pd.DataFrame, user_col: str | None = None) -> str:
    if user_col:
        if user_col not in df.columns:
            raise ValueError(f"Text column '{user_col}' not found. Available columns: {list(df.columns)}")
        return user_col

    candidates = [
        "text",
        "article",
        "content",
        "body",
        "full_text",
        "news_text",
        "sentence",
        "headline",
        "title",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(
        "Could not infer text column. Please pass --text-column. "
        f"Available columns: {list(df.columns)}"
    )


def normalize_label(x):
    if pd.isna(x):
        return "unknown"
    return str(x).strip().lower()


def predict_batches(df, text_col, model_dir, batch_size, max_length, threshold):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    pred_labels = []
    prob_non_pol = []
    prob_pol = []

    texts = df[text_col].fillna("").astype(str).tolist()

    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
        batch_texts = texts[i : i + batch_size]

        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu()

        p0 = probs[:, 0].tolist()
        p1 = probs[:, 1].tolist()

        prob_non_pol.extend(p0)
        prob_pol.extend(p1)
        pred_labels.extend([1 if p >= threshold else 0 for p in p1])

    return pred_labels, prob_non_pol, prob_pol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="News 900 CSV/TSV/XLSX file")
    ap.add_argument("--output-dir", default="outputs/news900_modernbert_validation")
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL))
    ap.add_argument("--text-column", default=None)
    ap.add_argument("--label-column", default="label")
    ap.add_argument("--label-code-column", default="label_code")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument(
        "--neutral-is-political",
        action="store_true",
        help="Treat neutral news as political/center-neutral rather than non-political.",
    )
    args = ap.parse_args()

    inp = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(inp)
    print("Loaded:", inp)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    text_col = find_text_column(df, args.text_column)
    print("Using text column:", text_col)

    if args.label_column not in df.columns:
        raise ValueError(f"Label column '{args.label_column}' not found. Available columns: {list(df.columns)}")

    df["gold_label_norm"] = df[args.label_column].apply(normalize_label)

    print("\nGold label distribution:")
    print(df["gold_label_norm"].value_counts(dropna=False))

    # For a political binary gate:
    # left/right are political.
    # neutral can mean either center-political or non-political depending on dataset design.
    # For Tanise's news ideology dataset, neutral should usually be treated as center-political.
    if args.neutral_is_political:
        df["gold_binary"] = df["gold_label_norm"].map(
            lambda x: 1 if x in {"left", "right", "neutral", "center", "centre", "center_neutral"} else 0
        )
    else:
        df["gold_binary"] = df["gold_label_norm"].map(
            lambda x: 1 if x in {"left", "right"} else 0
        )

    pred, p0, p1 = predict_batches(
        df=df,
        text_col=text_col,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        threshold=args.threshold,
    )

    df["modernbert_pred_binary"] = pred
    df["modernbert_prob_non_political"] = p0
    df["modernbert_prob_political"] = p1

    acc = accuracy_score(df["gold_binary"], df["modernbert_pred_binary"])
    cm = confusion_matrix(df["gold_binary"], df["modernbert_pred_binary"], labels=[0, 1])
    report = classification_report(
        df["gold_binary"],
        df["modernbert_pred_binary"],
        labels=[0, 1],
        target_names=["non_political", "political"],
        digits=4,
        zero_division=0,
    )

    print("\nBinary ModernBERT evaluation")
    print("neutral_is_political:", args.neutral_is_political)
    print("Accuracy:", acc)
    print("Confusion matrix [[gold_non_pol->pred_non_pol, pred_pol], [gold_pol->pred_non_pol, pred_pol]]:")
    print(cm)
    print(report)

    print("\nPredictions by original label:")
    print(pd.crosstab(df["gold_label_norm"], df["modernbert_pred_binary"], dropna=False))

    print("\nMean political probability by original label:")
    print(df.groupby("gold_label_norm")["modernbert_prob_political"].describe())

    out_csv = out_dir / "news900_modernbert_predictions.csv"
    df.to_csv(out_csv, index=False)

    summary = {
        "input": str(inp),
        "rows": int(len(df)),
        "text_column": text_col,
        "label_column": args.label_column,
        "neutral_is_political": bool(args.neutral_is_political),
        "accuracy": float(acc),
        "confusion_matrix_labels_0_1": cm.tolist(),
        "gold_label_distribution": df["gold_label_norm"].value_counts(dropna=False).to_dict(),
        "pred_binary_distribution": df["modernbert_pred_binary"].value_counts(dropna=False).to_dict(),
        "output_csv": str(out_csv),
    }

    out_json = out_dir / "news900_modernbert_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))

    print("\nSaved predictions:", out_csv)
    print("Saved summary:", out_json)


if __name__ == "__main__":
    main()