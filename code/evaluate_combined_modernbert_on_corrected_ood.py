from pathlib import Path
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = Path("outputs/target_domain_binary_modernbert_combined/best_model")
DATA_IN = Path("outputs/modernbert_ood_test/modernbert_ood_predictions_corrected.csv")
OUT = Path("outputs/target_domain_binary_modernbert_combined/corrected_ood_predictions.csv")

df = pd.read_csv(DATA_IN)

def to_binary(label):
    s = str(label).strip()
    if s == "0":
        return 0
    if s in {"1", "2", "3"}:
        return 1
    return None

df["gold"] = df["final_review_label"].apply(to_binary)
df = df[df["gold"].notna()].copy()
df["gold"] = df["gold"].astype(int)

texts = df["text_preview"].astype(str).tolist()

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

all_probs = []
all_preds = []

batch_size = 8

with torch.no_grad():
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).detach().cpu()
        preds = probs.argmax(dim=-1).numpy()

        all_probs.extend(probs.numpy().tolist())
        all_preds.extend(preds.tolist())

df["combined_modernbert_pred_binary"] = all_preds
df["combined_prob_non_political"] = [p[0] for p in all_probs]
df["combined_prob_political"] = [p[1] for p in all_probs]

y_true = df["gold"].tolist()
y_pred = df["combined_modernbert_pred_binary"].tolist()

print("Rows evaluated:", len(df))
print("\nGold distribution:")
print(df["gold"].value_counts().sort_index())

print("\nPred distribution:")
print(df["combined_modernbert_pred_binary"].value_counts().sort_index())

print("\nAccuracy:", accuracy_score(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
print("\nConfusion matrix:")
print(pd.DataFrame(
    cm,
    index=["gold_non_political", "gold_political"],
    columns=["pred_non_political", "pred_political"],
))

print("\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["non_political", "political"],
    digits=4,
))

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print("\nSaved:", OUT)
