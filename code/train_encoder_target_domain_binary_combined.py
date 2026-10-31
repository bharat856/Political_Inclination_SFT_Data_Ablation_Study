from pathlib import Path
import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

IN = Path("data/processed/tulu_olmo2_sft/target_domain_binary_combined_train.csv")
OUT_DIR = Path("outputs/target_domain_binary_modernbert_combined")
MODEL_NAME = "answerdotai/ModernBERT-base"

df = pd.read_csv(IN)
df = df[["text", "label"]].dropna().copy()
df["label"] = df["label"].astype(int)

print("Rows:", len(df))
print("Label counts:")
print(df["label"].value_counts())

train_df, test_df = train_test_split(
    df,
    test_size=0.25,
    random_state=42,
    stratify=df["label"],
)

print("\nTrain label counts:")
print(train_df["label"].value_counts())
print("\nTest label counts:")
print(test_df["label"].value_counts())

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=512,
    )

train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

id2label = {0: "non_political", 1: "political"}
label2id = {"non_political": 0, "political": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
    }

args = TrainingArguments(
    output_dir=str(OUT_DIR),
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    num_train_epochs=8,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    logging_steps=5,
    save_total_limit=2,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

pred = trainer.predict(test_ds)
y_true = pred.label_ids
y_pred = np.argmax(pred.predictions, axis=-1)

print("\nFinal test accuracy:")
print(accuracy_score(y_true, y_pred))

print("\nConfusion matrix:")
print(pd.DataFrame(
    confusion_matrix(y_true, y_pred, labels=[0, 1]),
    index=["gold_non_political", "gold_political"],
    columns=["pred_non_political", "pred_political"],
))

print("\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["non_political", "political"],
    zero_division=0,
))

trainer.save_model(str(OUT_DIR / "best_model"))
tokenizer.save_pretrained(str(OUT_DIR / "best_model"))

test_out = test_df.reset_index(drop=True).copy()
test_out["pred_label"] = y_pred
test_out["pred_name"] = test_out["pred_label"].map(id2label)
test_out.to_csv(OUT_DIR / "test_predictions.csv", index=False)

print("\nSaved model:", OUT_DIR / "best_model")
print("Saved predictions:", OUT_DIR / "test_predictions.csv")
