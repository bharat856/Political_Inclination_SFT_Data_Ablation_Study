from pathlib import Path
import json
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = Path("outputs/target_domain_binary_modernbert_combined/best_model")
OUT_DIR = Path("outputs/validation_external/20newsgroups_modernbert_binary")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Political-ish 20NG categories.
POLITICAL_CATEGORIES = {
    "talk.politics.guns",
    "talk.politics.mideast",
    "talk.politics.misc",
}

# Exclude religion for the cleanest politics/non-politics setup at first.
ALL_CATEGORIES = [
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

print("Loading 20 Newsgroups...")
data = fetch_20newsgroups(
    subset="test",
    categories=ALL_CATEGORIES,
    remove=("headers", "footers", "quotes"),
)

rows = []
for text, target in zip(data.data, data.target):
    category = data.target_names[target]
    rows.append({
        "text": text,
        "category": category,
        "gold_binary": 1 if category in POLITICAL_CATEGORIES else 0,
    })

df = pd.DataFrame(rows)
df = df[df["text"].fillna("").str.strip().ne("")].reset_index(drop=True)

print("Rows:", len(df))
print("Gold distribution:")
print(df["gold_binary"].value_counts())
print("By category:")
print(pd.crosstab(df["category"], df["gold_binary"]))

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

preds = []
p_pol = []
p_non = []

texts = df["text"].fillna("").astype(str).tolist()
batch_size = 32
max_length = 512

for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
    batch = texts[i:i+batch_size]
    enc = tokenizer(
        batch,
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

    p_non.extend(p0)
    p_pol.extend(p1)
    preds.extend([1 if p >= 0.5 else 0 for p in p1])

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
print("Confusion matrix:")
print(cm)
print(report)

print("\nBy category predictions:")
print(pd.crosstab(df["category"], df["modernbert_pred_binary"]))

out_csv = OUT_DIR / "20newsgroups_modernbert_predictions.csv"
out_json = OUT_DIR / "20newsgroups_modernbert_summary.json"

df.to_csv(out_csv, index=False)

summary = {
    "dataset": "20newsgroups",
    "rows": int(len(df)),
    "political_categories": sorted(POLITICAL_CATEGORIES),
    "accuracy": float(acc),
    "confusion_matrix_labels_0_1": cm.tolist(),
    "classification_report": report,
    "gold_distribution": df["gold_binary"].value_counts().to_dict(),
    "pred_distribution": df["modernbert_pred_binary"].value_counts().to_dict(),
    "output_csv": str(out_csv),
}

out_json.write_text(json.dumps(summary, indent=2))
print("Saved:", out_csv)
print("Saved:", out_json)
