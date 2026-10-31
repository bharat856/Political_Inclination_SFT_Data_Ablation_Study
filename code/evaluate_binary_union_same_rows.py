from pathlib import Path
import json
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

OUT_DIR = Path("outputs/validation_external/binary_union_same_rows")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "BAAI/bge-base-en-v1.5"

DATASETS = {
    "20newsgroups": Path("outputs/validation_external/20newsgroups_modernbert_binary/20newsgroups_modernbert_predictions.csv"),
    "bbc_news": Path("outputs/validation_external/bbc_news_modernbert_binary/bbc_news_modernbert_predictions.csv"),
}

def find_col(df, candidates, contains=None):
    for c in candidates:
        if c in df.columns:
            return c
    if contains:
        hits = [c for c in df.columns if contains.lower() in c.lower()]
        if hits:
            return hits[0]
    raise ValueError(f"Could not find column. Columns: {df.columns.tolist()}")

def get_gold(df):
    if "gold_binary" in df.columns:
        return df["gold_binary"].astype(int)

    if "label_text" in df.columns:
        return (df["label_text"].astype(str).str.lower() == "politics").astype(int)

    if "category" in df.columns:
        return df["category"].astype(str).str.startswith("talk.politics").astype(int)

    raise ValueError(f"No gold column found. Columns: {df.columns.tolist()}")

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
    }

print("Loading embedding model:", MODEL_NAME)
embedder = SentenceTransformer(MODEL_NAME)

combined = {}

for name, path in DATASETS.items():
    print("\n" + "=" * 100)
    print("Dataset:", name)
    print("Input:", path)

    df = pd.read_csv(path)
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

    text_col = find_col(df, ["text", "content", "full_text"], contains="text")
    mb_pred_col = find_col(
        df,
        ["modernbert_pred_binary", "pred_binary", "binary_pred", "pred"],
        contains="pred",
    )

    y = get_gold(df).astype(int)
    modernbert_pred = df[mb_pred_col].astype(int)

    texts = df[text_col].fillna("").astype(str).tolist()

    print("Gold distribution:")
    print(y.value_counts())

    print("ModernBERT pred col:", mb_pred_col)
    print("Embedding exact same rows...")

    X = embedder.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probs = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]

    embedding_t03 = (probs >= 0.3).astype(int)
    embedding_t05 = (probs >= 0.5).astype(int)

    union_t03 = ((modernbert_pred == 1) | (embedding_t03 == 1)).astype(int)
    union_t05 = ((modernbert_pred == 1) | (embedding_t05 == 1)).astype(int)

    df["embedding_prob_political_cv"] = probs
    df["embedding_pred_binary_t03"] = embedding_t03
    df["embedding_pred_binary_t05"] = embedding_t05
    df["union_pred_binary_t03"] = union_t03
    df["union_pred_binary_t05"] = union_t05

    df["embedding_only_t03"] = ((embedding_t03 == 1) & (modernbert_pred == 0)).astype(int)
    df["modernbert_only_t03"] = ((embedding_t03 == 0) & (modernbert_pred == 1)).astype(int)
    df["both_t03"] = ((embedding_t03 == 1) & (modernbert_pred == 1)).astype(int)

    methods = {
        "modernbert": modernbert_pred,
        "embedding_t03": embedding_t03,
        "embedding_t05": embedding_t05,
        "union_t03": union_t03,
        "union_t05": union_t05,
    }

    summary = {
        "dataset": name,
        "rows": int(len(df)),
        "gold_distribution": y.value_counts().astype(int).to_dict(),
        "text_col": text_col,
        "modernbert_pred_col": mb_pred_col,
        "candidate_counts": {
            "modernbert": int(modernbert_pred.sum()),
            "embedding_t03": int(embedding_t03.sum()),
            "embedding_t05": int(embedding_t05.sum()),
            "union_t03": int(union_t03.sum()),
            "union_t05": int(union_t05.sum()),
            "both_t03": int(df["both_t03"].sum()),
            "embedding_only_t03": int(df["embedding_only_t03"].sum()),
            "modernbert_only_t03": int(df["modernbert_only_t03"].sum()),
        },
        "methods": {},
    }

    for method, pred in methods.items():
        res = evaluate(y, pred)
        summary["methods"][method] = res

        print("\nMethod:", method)
        print("Accuracy:", res["accuracy"])
        print("Political precision:", res["political_precision"])
        print("Political recall:", res["political_recall"])
        print("Political F1:", res["political_f1"])
        print("Confusion matrix:", res["confusion_matrix"])

    print("\nCandidate counts:")
    print(summary["candidate_counts"])

    out_csv = OUT_DIR / f"{name}_binary_union_same_rows_predictions.csv"
    out_json = OUT_DIR / f"{name}_binary_union_same_rows_summary.json"

    df.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(summary, indent=2))

    print("Saved:", out_csv)
    print("Saved:", out_json)

    combined[name] = summary

combined_path = OUT_DIR / "binary_union_same_rows_combined_summary.json"
combined_path.write_text(json.dumps(combined, indent=2))
print("\nSaved combined:", combined_path)
