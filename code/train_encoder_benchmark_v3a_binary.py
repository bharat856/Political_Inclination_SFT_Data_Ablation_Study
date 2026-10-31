import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

LABEL2ID = {"left": 0, "right": 1}
ID2LABEL = {0: "left", 1: "right"}

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.benchmark)
    df = df[df["label"].isin(LABEL2ID)].copy()
    df["labels"] = df["label"].map(LABEL2ID)

    train_df = df[df["source_split"] == "train"].reset_index(drop=True)
    val_df = df[df["source_split"] == "validation"].reset_index(drop=True)
    test_df = df[df["source_split"] == "test"].reset_index(drop=True)

    print("Model:", args.model)
    print("Train:", len(train_df))
    print("Validation:", len(val_df))
    print("Test:", len(test_df))
    print("\nTrain labels:")
    print(train_df["label"].value_counts())
    print("\nValidation labels:")
    print(val_df["label"].value_counts())
    print("\nTest labels:")
    print(test_df["label"].value_counts())

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
        )

    train_ds = Dataset.from_pandas(train_df[["text", "labels"]])
    val_ds = Dataset.from_pandas(val_df[["text", "labels"]])
    test_ds = Dataset.from_pandas(test_df[["text", "labels"]])

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        id2label={str(k): v for k, v in ID2LABEL.items()},
        label2id=LABEL2ID,
        trust_remote_code=True,
    )

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=args.seed,
        bf16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("\nEvaluating on test split...")
    pred = trainer.predict(test_ds)
    logits = pred.predictions
    y_true = pred.label_ids
    y_pred = np.argmax(logits, axis=-1)

    test_df["pred_id"] = y_pred
    test_df["pred_label"] = [ID2LABEL[int(x)] for x in y_pred]
    test_df["is_correct"] = test_df["pred_id"] == test_df["labels"]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    report = classification_report(
        y_true,
        y_pred,
        target_names=["left", "right"],
        digits=4,
        output_dict=True,
    )

    cm = confusion_matrix(y_true, y_pred)

    test_out = out_dir / "test_predictions.csv"
    metrics_out = out_dir / "test_metrics.json"
    report_out = out_dir / "classification_report.json"

    test_df.to_csv(test_out, index=False)

    metrics = {
        "model": args.model,
        "test_rows": int(len(test_df)),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix_labels": ["left", "right"],
        "confusion_matrix": cm.tolist(),
        "pred_distribution": test_df["pred_label"].value_counts().to_dict(),
        "gold_distribution": test_df["label"].value_counts().to_dict(),
    }

    metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nTest accuracy:", round(acc, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Weighted F1:", round(weighted_f1, 4))

    print("\nConfusion matrix rows=gold, cols=pred [left,right]:")
    print(cm)

    print("\nPred distribution:")
    print(test_df["pred_label"].value_counts())

    print("\nSaved:")
    print(test_out)
    print(metrics_out)
    print(report_out)

if __name__ == "__main__":
    main()
