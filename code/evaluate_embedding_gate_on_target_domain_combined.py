from pathlib import Path
import json
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

INPUT = Path("data/processed/tulu_olmo2_sft/target_domain_binary_combined_train.csv")
OUT_DIR = Path("outputs/validation_target_domain/embedding_gate_combined_train")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "BAAI/bge-base-en-v1.5"
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


def load_external_train():
    rows = []

    print("Loading 20NG...")
    ng = fetch_20newsgroups(
        subset="test",
        categories=ALL_20NG,
        remove=("headers", "footers", "quotes"),
    )

    for text, target in zip(ng.data, ng.target):
        category = ng.target_names[target]
        y = 1 if category in POLITICAL_20NG else 0
        rows.append({"text": text, "label": y, "source": "20newsgroups"})

    print("Loading BBC...")
    bbc = pd.read_csv(BBC_PATH)

    for _, r in bbc.iterrows():
        label = str(r["label_text"]).strip().lower()
        y = 1 if label == "politics" else 0
        rows.append({"text": str(r["text"]), "label": y, "source": "bbc_news"})

    df = pd.DataFrame(rows)
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.len() > 0].copy()

    print("External train rows:", len(df))
    print(df["label"].value_counts())
    print(df["source"].value_counts())

    return df


def evaluate(y, pred):
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
        "accuracy": float(accuracy_score(y, pred)),
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


def main():
    print("Loading target-domain validation:", INPUT)
    target = pd.read_csv(INPUT)
    target["text"] = target["text"].fillna("").astype(str)
    target["label"] = target["label"].astype(int)

    print("Target rows:", len(target))
    print("Target distribution:")
    print(target["label"].value_counts())
    if "source_split" in target.columns:
        print("Source split:")
        print(target["source_split"].value_counts())

    train = load_external_train()

    print("Loading embedding model:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding external train...")
    X_train = model.encode(
        train["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    y_train = train["label"].astype(int).to_numpy()

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)

    print("Encoding target-domain validation...")
    X_target = model.encode(
        target["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    probs = clf.predict_proba(X_target)[:, 1]
    target["embedding_prob_political"] = probs

    y = target["label"].astype(int).to_numpy()

    thresholds = [0.2, 0.3, 0.4, 0.5]
    summary = {
        "input": str(INPUT),
        "rows": int(len(target)),
        "gold_distribution": target["label"].value_counts().astype(int).to_dict(),
        "embedding_model": MODEL_NAME,
        "external_train_sources": ["20newsgroups", "bbc_news"],
        "thresholds": {},
    }

    print("\n" + "=" * 100)
    print("Embedding gate validation on target-domain combined labels")
    print("=" * 100)

    for th in thresholds:
        pred = (probs >= th).astype(int)
        col = f"embedding_pred_t{str(th).replace('.', '')}"
        target[col] = pred

        res = evaluate(y, pred)
        candidate_rate = float(pred.mean())
        res["candidate_rate"] = candidate_rate
        res["false_negatives"] = int(((y == 1) & (pred == 0)).sum())
        res["false_positives"] = int(((y == 0) & (pred == 1)).sum())

        summary["thresholds"][str(th)] = res

        print(f"\nThreshold {th}")
        print("Candidate rate:", candidate_rate)
        print("Accuracy:", res["accuracy"])
        print("Political precision:", res["political_precision"])
        print("Political recall:", res["political_recall"])
        print("Political F1:", res["political_f1"])
        print("False negatives:", res["false_negatives"])
        print("False positives:", res["false_positives"])
        print("Confusion matrix:", res["confusion_matrix"])

    out_csv = OUT_DIR / "target_domain_combined_embedding_gate_predictions.csv"
    out_json = OUT_DIR / "target_domain_combined_embedding_gate_summary.json"

    target.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(summary, indent=2))

    print("\nSaved:", out_csv)
    print("Saved:", out_json)


if __name__ == "__main__":
    main()
