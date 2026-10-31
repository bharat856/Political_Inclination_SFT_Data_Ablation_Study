import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

LABEL2ID = {"left": 0, "center": 1, "right": 2}
ID2LABEL = {0: "left", 1: "center", 2: "right"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def doc_level_split(df, seed=42, train_frac=0.70, val_frac=0.15):
    """
    Stratified document-level split.
    No doc_id appears in more than one split.
    """
    docs = df[["doc_id", "rile_label_3way"]].drop_duplicates().copy()
    docs = docs.rename(columns={"rile_label_3way": "label"})

    split_rows = []

    rng = np.random.default_rng(seed)

    for label, g in docs.groupby("label"):
        doc_ids = g["doc_id"].tolist()
        rng.shuffle(doc_ids)

        n = len(doc_ids)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))

        train_docs = doc_ids[:n_train]
        val_docs = doc_ids[n_train:n_train + n_val]
        test_docs = doc_ids[n_train + n_val:]

        split_rows += [{"doc_id": d, "split": "train"} for d in train_docs]
        split_rows += [{"doc_id": d, "split": "validation"} for d in val_docs]
        split_rows += [{"doc_id": d, "split": "test"} for d in test_docs]

    split_df = pd.DataFrame(split_rows)
    return df.merge(split_df, on="doc_id", how="inner")


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
    parser.add_argument("--benchmark", default="data/processed/benchmark_v2/benchmark_v2.csv")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading benchmark:", args.benchmark)
    df = pd.read_csv(args.benchmark)

    df = df[df["rile_label_3way"].isin(LABEL2ID)].copy()
    df["label"] = df["rile_label_3way"].map(LABEL2ID)
    df["text"] = df["text"].astype(str)

    df = doc_level_split(df, seed=args.seed)

    print("\nRows by split:")
    print(df["split"].value_counts())

    print("\nDocuments by split:")
    print(df.groupby("split")["doc_id"].nunique())

    print("\nLabel distribution by split:")
    print(pd.crosstab(df["split"], df["rile_label_3way"]))

    split_path = out_dir / "split_manifest.csv"
    df[["benchmark_id", "chunk_id", "doc_id", "rile_label_3way", "label", "split"]].to_csv(split_path, index=False)

    train_df = df[df["split"] == "train"][["text", "label"]].reset_index(drop=True)
    val_df = df[df["split"] == "validation"][["text", "label"]].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
        )

    train_ds = Dataset.from_pandas(train_df).map(tokenize, batched=True)
    val_ds = Dataset.from_pandas(val_df).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        trust_remote_code=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    common_args = dict(
        output_dir=str(out_dir / "checkpoints"),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.06,
        logging_steps=20,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        bf16=torch.cuda.is_available(),
        report_to="none",
        seed=args.seed,
    )

    try:
        training_args = TrainingArguments(
            **common_args,
            eval_strategy="epoch",
            save_total_limit=2,
        )
    except TypeError:
        training_args = TrainingArguments(
            **common_args,
            evaluation_strategy="epoch",
            save_total_limit=2,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nTraining model:", args.model)
    trainer.train()

    print("\nEvaluating on held-out document-level test split...")

    test_ds = Dataset.from_pandas(test_df[["text", "label"]].reset_index(drop=True)).map(tokenize, batched=True)
    pred_output = trainer.predict(test_ds)

    logits = pred_output.predictions
    y_true = pred_output.label_ids
    y_pred = np.argmax(logits, axis=-1)

    test_df["pred_id"] = y_pred
    test_df["pred_label"] = test_df["pred_id"].map(ID2LABEL)
    test_df["gold_label"] = test_df["label"].map(ID2LABEL)
    test_df["correct"] = test_df["pred_label"] == test_df["gold_label"]

    chunk_accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    print("\nChunk-level test metrics:")
    print("Accuracy:", round(chunk_accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Weighted F1:", round(weighted_f1, 4))

    print("\nConfusion matrix:")
    labels_order = ["left", "center", "right"]
    print(pd.crosstab(test_df["gold_label"], test_df["pred_label"], rownames=["gold"], colnames=["pred"]))

    print("\nClassification report:")
    print(classification_report(
        test_df["gold_label"],
        test_df["pred_label"],
        labels=labels_order,
        zero_division=0,
    ))

    # Document-majority evaluation
    doc_rows = []
    for doc_id, g in test_df.groupby("doc_id"):
        gold = g["gold_label"].iloc[0]
        pred = g["pred_label"].value_counts().index[0]
        doc_rows.append({
            "doc_id": doc_id,
            "gold_label": gold,
            "predicted_label": pred,
            "n_chunks": len(g),
        })

    doc_df = pd.DataFrame(doc_rows)
    doc_df["correct"] = doc_df["gold_label"] == doc_df["predicted_label"]
    doc_acc = doc_df["correct"].mean()

    print("\nDocument-majority test accuracy:", round(doc_acc, 4))
    print("\nDocument-level confusion:")
    print(pd.crosstab(doc_df["gold_label"], doc_df["predicted_label"], rownames=["gold"], colnames=["pred"]))

    test_df.to_csv(out_dir / "test_predictions.csv", index=False)
    doc_df.to_csv(out_dir / "document_majority_predictions.csv", index=False)

    summary = {
        "model": args.model,
        "benchmark": args.benchmark,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "seed": args.seed,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "test_documents": int(test_df["doc_id"].nunique()),
        "chunk_accuracy": float(chunk_accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "document_majority_accuracy": float(doc_acc),
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved outputs to:", out_dir)


if __name__ == "__main__":
    main()
