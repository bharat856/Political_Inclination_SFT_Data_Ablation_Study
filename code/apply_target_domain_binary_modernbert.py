from pathlib import Path
import json
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = Path("outputs/target_domain_binary_modernbert/best_model")
IN = Path("data/processed/tulu_olmo2_sft/tulu_olmo2_sft_political_candidates.jsonl")
OUT = Path("outputs/target_domain_binary_modernbert/tulu_olmo2_sft_candidates_binary_predictions.csv")

BATCH_SIZE = 32
MAX_LENGTH = 512

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

rows = []
with open(IN, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

print("Loaded rows:", len(rows))

def get_text(r):
    for col in ["text", "full_text", "text_preview"]:
        if col in r and r[col]:
            return str(r[col])
    if "messages" in r:
        return "\n".join([str(m.get("content", "")) for m in r["messages"]])
    return json.dumps(r, ensure_ascii=False)

texts = [get_text(r) for r in rows]

all_preds = []
all_probs = []

for i in tqdm(range(0, len(texts), BATCH_SIZE)):
    batch_texts = texts[i:i+BATCH_SIZE]

    enc = tokenizer(
        batch_texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)

    preds = torch.argmax(probs, dim=-1).cpu().tolist()
    probs = probs.cpu().tolist()

    all_preds.extend(preds)
    all_probs.extend(probs)

out_rows = []
for r, pred, prob in zip(rows, all_preds, all_probs):
    out_rows.append({
        "row_id": r.get("row_id"),
        "id": r.get("id"),
        "source": r.get("source"),
        "strong_hits": r.get("strong_hits"),
        "issue_hits": r.get("issue_hits"),
        "context_hits": r.get("context_hits"),
        "binary_pred_id": pred,
        "binary_pred_label": "political" if pred == 1 else "non_political",
        "prob_non_political": prob[0],
        "prob_political": prob[1],
        "text_preview": get_text(r)[:1000],
    })

OUT.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(out_rows).to_csv(OUT, index=False)

print("Saved:", OUT)
print(pd.Series([x["binary_pred_label"] for x in out_rows]).value_counts())
