from pathlib import Path
import json
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

INPUT = Path("data/validation_tanise/slant_testset_for_4way.csv")
OUT_DIR = Path("outputs/validation_tanise/slant_testset_embedding_3way")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEXT_COL = "full_text"
LABEL_COL = "label"

MODEL_NAME = "BAAI/bge-base-en-v1.5"

df = pd.read_csv(INPUT)

gold_map = {
    "left": "left",
    "neutral": "center_neutral",
    "right": "right",
}

df["gold_3way"] = df[LABEL_COL].astype(str).str.strip().str.lower().map(gold_map)
df = df[df["gold_3way"].notna()].copy()

texts = df[TEXT_COL].fillna("").astype(str).tolist()
y = df["gold_3way"].to_numpy()

print("Rows:", len(df))
print(df["gold_3way"].value_counts())

print("Loading embedding model:", MODEL_NAME)
model = SentenceTransformer(MODEL_NAME)

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
pred = cross_val_predict(clf, X, y, cv=cv)

labels = ["left", "center_neutral", "right"]

acc = accuracy_score(y, pred)
cm = confusion_matrix(y, pred, labels=labels)
report = classification_report(y, pred, labels=labels, digits=4, zero_division=0)

print("\n5-fold CV accuracy:", acc)
print("\nConfusion matrix labels:", labels)
print(cm)
print("\nReport:")
print(report)

df["embedding_pred_3way_cv"] = pred

out_csv = OUT_DIR / "slant_embedding_3way_cv_predictions.csv"
df.to_csv(out_csv, index=False)

summary = {
    "model_name": MODEL_NAME,
    "rows": int(len(df)),
    "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
    "accuracy": float(acc),
    "labels": labels,
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
    "output_csv": str(out_csv),
}

out_json = OUT_DIR / "slant_embedding_3way_cv_summary.json"
out_json.write_text(json.dumps(summary, indent=2))

print("Saved:", out_csv)
print("Saved:", out_json)
