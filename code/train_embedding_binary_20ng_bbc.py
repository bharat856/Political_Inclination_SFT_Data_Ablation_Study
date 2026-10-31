from pathlib import Path
import json
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

OUT_DIR = Path("outputs/validation_external/embedding_binary_20ng_bbc")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "BAAI/bge-base-en-v1.5"

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

def load_20ng():
    data = fetch_20newsgroups(
        subset="test",
        categories=ALL_20NG,
        remove=("headers", "footers", "quotes"),
    )

    rows = []
    for text, target in zip(data.data, data.target):
        category = data.target_names[target]
        gold = 1 if category in POLITICAL_20NG else 0
        rows.append({
            "dataset": "20newsgroups",
            "text": text,
            "category": category,
            "gold_binary": gold,
        })

    return pd.DataFrame(rows)

def load_bbc():
    p = Path("data/validation_external/bbc_news.csv")
    df = pd.read_csv(p)

    rows = []
    for _, r in df.iterrows():
        label = str(r["label_text"]).strip().lower()
        gold = 1 if label == "politics" else 0
        rows.append({
            "dataset": "bbc_news",
            "text": str(r["text"]),
            "category": label,
            "gold_binary": gold,
        })

    return pd.DataFrame(rows)

def evaluate_dataset(name, df, model):
    print("\n" + "=" * 100)
    print("Dataset:", name)
    print("Rows:", len(df))
    print(df["gold_binary"].value_counts())

    texts = df["text"].fillna("").astype(str).tolist()
    y = df["gold_binary"].to_numpy()

    X = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    print("Embeddings:", X.shape)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probs = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]

    thresholds = [0.5, 0.4, 0.3, 0.2]
    threshold_summaries = {}

    for th in thresholds:
        pred = (probs >= th).astype(int)

        acc = accuracy_score(y, pred)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        report_dict = classification_report(
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

        print("\nThreshold:", th)
        print("Accuracy:", acc)
        print("Confusion matrix labels [0, 1]:")
        print(cm)
        print(report_text)

        threshold_summaries[str(th)] = {
            "accuracy": float(acc),
            "confusion_matrix_labels": [0, 1],
            "confusion_matrix": cm.tolist(),
            "political_precision": float(report_dict["political"]["precision"]),
            "political_recall": float(report_dict["political"]["recall"]),
            "political_f1": float(report_dict["political"]["f1-score"]),
            "non_political_precision": float(report_dict["non_political"]["precision"]),
            "non_political_recall": float(report_dict["non_political"]["recall"]),
            "non_political_f1": float(report_dict["non_political"]["f1-score"]),
            "macro_f1": float(report_dict["macro avg"]["f1-score"]),
            "weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
        }

    df_out = df.copy()
    df_out["embedding_prob_political_cv"] = probs
    df_out["embedding_pred_binary_t05"] = (probs >= 0.5).astype(int)
    df_out["embedding_pred_binary_t04"] = (probs >= 0.4).astype(int)
    df_out["embedding_pred_binary_t03"] = (probs >= 0.3).astype(int)
    df_out["embedding_pred_binary_t02"] = (probs >= 0.2).astype(int)

    out_csv = OUT_DIR / f"{name}_embedding_binary_cv_predictions.csv"
    df_out.to_csv(out_csv, index=False)

    summary = {
        "dataset": name,
        "model_name": MODEL_NAME,
        "classifier": "LogisticRegression",
        "evaluation": "5-fold StratifiedKFold cross-validation",
        "rows": int(len(df)),
        "gold_distribution": df["gold_binary"].value_counts().astype(int).to_dict(),
        "thresholds": threshold_summaries,
        "output_csv": str(out_csv),
    }

    out_json = OUT_DIR / f"{name}_embedding_binary_cv_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))

    print("Saved:", out_csv)
    print("Saved:", out_json)

    return summary

print("Loading embedding model:", MODEL_NAME)
model = SentenceTransformer(MODEL_NAME)

df20 = load_20ng()
dfbbc = load_bbc()

summary20 = evaluate_dataset("20newsgroups", df20, model)
summarybbc = evaluate_dataset("bbc_news", dfbbc, model)

combined = {
    "model_name": MODEL_NAME,
    "classifier": "LogisticRegression",
    "datasets": {
        "20newsgroups": summary20,
        "bbc_news": summarybbc,
    },
}

combined_path = OUT_DIR / "embedding_binary_20ng_bbc_combined_summary.json"
combined_path.write_text(json.dumps(combined, indent=2))
print("\nSaved combined summary:", combined_path)
