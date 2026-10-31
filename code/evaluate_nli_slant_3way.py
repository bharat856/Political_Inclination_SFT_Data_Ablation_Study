from pathlib import Path
import json
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

INPUT = Path("data/validation_tanise/slant_testset_for_4way.csv")
OUT_DIR = Path("outputs/validation_tanise/slant_testset_nli_3way")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
TEXT_COL = "full_text"
LABEL_COL = "label"

HYPOTHESES = {
    "left": "The author advances a left-leaning political agenda or left-leaning political views.",
    "center_neutral": "The author advances political views neutral with respect to the left-right divide.",
    "right": "The author advances a right-leaning political agenda or right-leaning political views.",
}

gold_map = {
    "left": "left",
    "neutral": "center_neutral",
    "right": "right",
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
print("Loading:", MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()

label2id = {k.lower(): v for k, v in model.config.label2id.items()}
print("Model labels:", model.config.label2id)

entailment_id = None
for k, v in label2id.items():
    if "entail" in k:
        entailment_id = v

if entailment_id is None:
    raise ValueError(f"Could not find entailment label in {model.config.label2id}")

print("Entailment id:", entailment_id)

df = pd.read_csv(INPUT)
df["gold_3way"] = df[LABEL_COL].astype(str).str.strip().str.lower().map(gold_map)
df = df[df["gold_3way"].notna()].copy()

texts = df[TEXT_COL].fillna("").astype(str).tolist()

def score_batch(premises, hypotheses, batch_size=16):
    scores = []

    pairs = list(zip(premises, hypotheses))

    for i in tqdm(range(0, len(pairs), batch_size), desc="NLI batches"):
        batch = pairs[i:i + batch_size]
        batch_premises = [x[0] for x in batch]
        batch_hypotheses = [x[1] for x in batch]

        enc = tokenizer(
            batch_premises,
            batch_hypotheses,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            entail_scores = probs[:, entailment_id].detach().cpu().tolist()

        scores.extend(entail_scores)

    return scores

all_rows = []
preds = []

for text in tqdm(texts, desc="Rows"):
    labels = list(HYPOTHESES.keys())
    hypotheses = [HYPOTHESES[l] for l in labels]
    premises = [text] * len(labels)

    scores = score_batch(premises, hypotheses, batch_size=3)

    score_dict = dict(zip(labels, scores))
    pred = max(score_dict, key=score_dict.get)

    preds.append(pred)
    all_rows.append(score_dict)

df["nli_pred_3way"] = preds
for label in HYPOTHESES:
    df[f"nli_entail_{label}"] = [r[label] for r in all_rows]

labels = ["left", "center_neutral", "right"]

acc = accuracy_score(df["gold_3way"], df["nli_pred_3way"])
cm = confusion_matrix(df["gold_3way"], df["nli_pred_3way"], labels=labels)
report = classification_report(
    df["gold_3way"],
    df["nli_pred_3way"],
    labels=labels,
    digits=4,
    zero_division=0,
)

print("\nRows:", len(df))
print("\nPred distribution:")
print(df["nli_pred_3way"].value_counts())

print("\nGold distribution:")
print(df["gold_3way"].value_counts())

print("\nCrosstab:")
print(pd.crosstab(df["gold_3way"], df["nli_pred_3way"]))

print("\nNLI 3-way accuracy:", acc)
print("\nConfusion matrix labels:", labels)
print(cm)
print("\nReport:")
print(report)

out_csv = OUT_DIR / "slant_nli_3way_predictions.csv"
df.to_csv(out_csv, index=False)

summary = {
    "model_name": MODEL_NAME,
    "rows": int(len(df)),
    "accuracy": float(acc),
    "labels": labels,
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
    "pred_distribution": df["nli_pred_3way"].value_counts().astype(int).to_dict(),
    "gold_distribution": df["gold_3way"].value_counts().astype(int).to_dict(),
    "hypotheses": HYPOTHESES,
    "output_csv": str(out_csv),
}

out_json = OUT_DIR / "slant_nli_3way_summary.json"
out_json.write_text(json.dumps(summary, indent=2))

print("Saved:", out_csv)
print("Saved:", out_json)
