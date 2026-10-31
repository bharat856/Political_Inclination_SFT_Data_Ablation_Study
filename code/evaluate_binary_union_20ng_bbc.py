from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

OUT_DIR = Path("outputs/validation_external/binary_union_20ng_bbc")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "20newsgroups": {
        "modernbert": Path("outputs/validation_external/20newsgroups_modernbert_binary/20newsgroups_modernbert_predictions.csv"),
        "embedding": Path("outputs/validation_external/embedding_binary_20ng_bbc/20newsgroups_embedding_binary_cv_predictions.csv"),
    },
    "bbc_news": {
        "modernbert": Path("outputs/validation_external/bbc_news_modernbert_binary/bbc_news_modernbert_predictions.csv"),
        "embedding": Path("outputs/validation_external/embedding_binary_20ng_bbc/bbc_news_embedding_binary_cv_predictions.csv"),
    },
}

def find_text_col(df):
    for c in ["text", "content", "full_text"]:
        if c in df.columns:
            return c
    raise ValueError(f"No text column found. Columns: {df.columns.tolist()}")

def find_modernbert_pred_col(df):
    for c in ["modernbert_pred_binary", "pred_binary", "pred", "prediction"]:
        if c in df.columns:
            return c
    candidates = [c for c in df.columns if "pred" in c.lower() and "binary" in c.lower()]
    if candidates:
        return candidates[0]
    raise ValueError(f"No ModernBERT prediction column found. Columns: {df.columns.tolist()}")

def evaluate(y, pred):
    acc = accuracy_score(y, pred)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    report = classification_report(
        y,
        pred,
        labels=[0, 1],
        target_names=["non_political", "political"],
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        y,
        pred,
        labels=[0, 1],
        target_names=["non_political", "political"],
        digits=4,
        zero_division=0,
    )

    return {
        "accuracy": float(acc),
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": cm.tolist(),
        "non_political_precision": float(report["non_political"]["precision"]),
        "non_political_recall": float(report["non_political"]["recall"]),
        "non_political_f1": float(report["non_political"]["f1-score"]),
        "political_precision": float(report["political"]["precision"]),
        "political_recall": float(report["political"]["recall"]),
        "political_f1": float(report["political"]["f1-score"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "report_text": report_text,
    }

all_summary = {}

for name, paths in DATASETS.items():
    print("\n" + "=" * 100)
    print("Dataset:", name)

    mb = pd.read_csv(paths["modernbert"])
    emb = pd.read_csv(paths["embedding"])

    mb_text_col = find_text_col(mb)
    emb_text_col = find_text_col(emb)

    # Align by row order. These prediction files were generated from the same dataset order.
    if len(mb) != len(emb):
        raise ValueError(f"Row count mismatch for {name}: ModernBERT={len(mb)}, Embedding={len(emb)}")

    if "gold_binary" in mb.columns:
        y = mb["gold_binary"].astype(int)
    elif "gold_binary" in emb.columns:
        y = emb["gold_binary"].astype(int)
    else:
        raise ValueError("No gold_binary column found.")

    mb_pred_col = find_modernbert_pred_col(mb)
    mb_pred = mb[mb_pred_col].astype(int)

    emb_pred = emb["embedding_pred_binary_t03"].astype(int)

    union_pred = ((mb_pred == 1) | (emb_pred == 1)).astype(int)

    df = emb.copy()
    df["modernbert_pred_binary"] = mb_pred.values
    df["embedding_pred_binary_t03"] = emb_pred.values
    df["union_pred_binary"] = union_pred.values

    df["modernbert_only"] = ((df["modernbert_pred_binary"] == 1) & (df["embedding_pred_binary_t03"] == 0)).astype(int)
    df["embedding_only"] = ((df["modernbert_pred_binary"] == 0) & (df["embedding_pred_binary_t03"] == 1)).astype(int)
    df["both_candidate"] = ((df["modernbert_pred_binary"] == 1) & (df["embedding_pred_binary_t03"] == 1)).astype(int)

    methods = {
        "modernbert": mb_pred,
        "embedding_t03": emb_pred,
        "union": union_pred,
    }

    summary = {
        "dataset": name,
        "rows": int(len(df)),
        "gold_distribution": y.value_counts().astype(int).to_dict(),
        "candidate_counts": {
            "modernbert": int(mb_pred.sum()),
            "embedding_t03": int(emb_pred.sum()),
            "union": int(union_pred.sum()),
            "both_candidate": int(df["both_candidate"].sum()),
            "modernbert_only": int(df["modernbert_only"].sum()),
            "embedding_only": int(df["embedding_only"].sum()),
        },
        "methods": {},
    }

    for method, pred in methods.items():
        res = evaluate(y, pred)
        summary["methods"][method] = {k: v for k, v in res.items() if k != "report_text"}

        print("\nMethod:", method)
        print("Accuracy:", res["accuracy"])
        print("Political precision:", res["political_precision"])
        print("Political recall:", res["political_recall"])
        print("Political F1:", res["political_f1"])
        print("Confusion matrix:", res["confusion_matrix"])

    print("\nCandidate counts:")
    print(summary["candidate_counts"])

    out_csv = OUT_DIR / f"{name}_binary_union_predictions.csv"
    df.to_csv(out_csv, index=False)

    out_json = OUT_DIR / f"{name}_binary_union_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))

    print("Saved:", out_csv)
    print("Saved:", out_json)

    all_summary[name] = summary

combined = OUT_DIR / "binary_union_20ng_bbc_combined_summary.json"
combined.write_text(json.dumps(all_summary, indent=2))
print("\nSaved combined:", combined)
