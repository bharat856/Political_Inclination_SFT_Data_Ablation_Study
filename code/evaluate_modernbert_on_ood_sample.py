from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm

input_path = Path("outputs/modernbert_ood_test/manual_review_ood_sample_300_labeled.csv")
model_path = Path("outputs/target_domain_binary_modernbert/best_model")
output_path = Path("outputs/modernbert_ood_test/modernbert_ood_predictions.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_path)

def normalize_manual(x):
    s = str(x).strip().lower()
    if s in ["u", "unclear", "nan", "none", ""]:
        return None
    if s.endswith(".0"):
        s = s[:-2]
    if s in ["0", "non_political", "non-political", "non political"]:
        return 0
    if s in ["1", "left", "2", "neutral", "center", "centre", "center_neutral", "3", "right"]:
        return 1
    return None

df["gold_binary"] = df["manual_label"].apply(normalize_manual)
df = df[df["gold_binary"].notna()].copy()
df["gold_binary"] = df["gold_binary"].astype(int)

text_col = "text_preview"
if text_col not in df.columns:
    raise ValueError("Expected text_preview column.")

device = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()

preds = []
probs_pol = []
probs_non = []

batch_size = 16
texts = df[text_col].astype(str).tolist()

for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i:i+batch_size]
    enc = tok(
        batch,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)

    pred = probs.argmax(dim=-1).cpu().tolist()
    preds.extend(pred)
    probs_non.extend(probs[:, 0].cpu().tolist())
    probs_pol.extend(probs[:, 1].cpu().tolist())

df["modernbert_pred_binary"] = preds
df["prob_non_political"] = probs_non
df["prob_political"] = probs_pol

y_true = df["gold_binary"]
y_pred = df["modernbert_pred_binary"]

print("Rows evaluated:", len(df))
print("Gold distribution:")
print(y_true.value_counts().sort_index())
print("\nPred distribution:")
print(pd.Series(y_pred).value_counts().sort_index())
print("\nAccuracy:", accuracy_score(y_true, y_pred))
print("\nConfusion matrix labels [0 non-political, 1 political]:")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
print("\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["non_political", "political"],
    zero_division=0,
))

df.to_csv(output_path, index=False)
print("\nSaved:", output_path)
