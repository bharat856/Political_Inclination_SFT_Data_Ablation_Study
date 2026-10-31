from pathlib import Path
import argparse
import json

import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_DIR = Path("outputs/target_domain_binary_modernbert_combined/best_model")


def read_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--text-column", default="text")
    ap.add_argument("--label-column", default="label_text")
    ap.add_argument("--output-dir", default="outputs/validation_external/bbc_news_modernbert_binary")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_df(Path(args.input))

    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

    if args.text_column not in df.columns:
        raise ValueError(f"Text column {args.text_column} not found. Available: {df.columns.tolist()}")

    if args.label_column not in df.columns:
        raise ValueError(f"Label column {args.label_column} not found. Available: {df.columns.tolist()}")

    df["label_norm"] = df[args.label_column].astype(str).str.strip().str.lower()

    # BBC News binary mapping:
    # politics = political
    # business/sport/tech/entertainment = non-political
    df["gold_binary"] = (df["label_norm"] == "politics").astype(int)

    print("\nGold label distribution:")
    print(df["label_norm"].value_counts(dropna=False))

    print("\nGold binary distribution:")
    print(df["gold_binary"].value_counts(dropna=False))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\nDevice:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    texts = df[args.text_column].fillna("").astype(str).tolist()

    preds = []
    p_non = []
    p_pol = []

    for i in tqdm(range(0, len(texts), args.batch_size), desc="Predicting"):
        batch = texts[i:i + args.batch_size]

        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=args.max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu()

        p0 = probs[:, 0].tolist()
        p1 = probs[:, 1].tolist()

        p_non.extend(p0)
        p_pol.extend(p1)
        preds.extend([1 if p >= args.threshold else 0 for p in p1])

    df["modernbert_pred_binary"] = preds
    df["modernbert_prob_non_political"] = p_non
    df["modernbert_prob_political"] = p_pol

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

    print("\nAccuracy:", acc)
    print("\nConfusion matrix labels [0=non_political, 1=political]:")
    print(cm)

    print("\nClassification report:")
    print(report)

    print("\nPredictions by BBC category:")
    print(pd.crosstab(df["label_norm"], df["modernbert_pred_binary"], dropna=False))

    print("\nMean political probability by BBC category:")
    print(df.groupby("label_norm")["modernbert_prob_political"].describe())

    out_csv = out_dir / "bbc_news_modernbert_predictions.csv"
    out_json = out_dir / "bbc_news_modernbert_summary.json"

    df.to_csv(out_csv, index=False)

    summary = {
        "dataset": "bbc_news",
        "input": args.input,
        "rows": int(len(df)),
        "text_column": args.text_column,
        "label_column": args.label_column,
        "binary_mapping": {
            "politics": 1,
            "business": 0,
            "sport": 0,
            "tech": 0,
            "entertainment": 0,
        },
        "accuracy": float(acc),
        "confusion_matrix_labels_0_1": cm.tolist(),
        "classification_report": report,
        "gold_label_distribution": df["label_norm"].value_counts(dropna=False).to_dict(),
        "gold_binary_distribution": df["gold_binary"].value_counts(dropna=False).to_dict(),
        "pred_binary_distribution": df["modernbert_pred_binary"].value_counts(dropna=False).to_dict(),
        "output_csv": str(out_csv),
    }

    out_json.write_text(json.dumps(summary, indent=2))

    print("\nSaved predictions:", out_csv)
    print("Saved summary:", out_json)


if __name__ == "__main__":
    main()
