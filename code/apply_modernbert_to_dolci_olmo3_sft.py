from pathlib import Path
import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

MODEL_DIR = Path("outputs/target_domain_binary_modernbert_combined/best_model")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input flattened JSONL")
    p.add_argument("--predictions-out", required=True, help="Output predictions JSONL, metadata only")
    p.add_argument("--candidates-out", required=True, help="Output political candidate JSONL, full rows")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests")
    return p.parse_args()


def flush_batch(batch, tokenizer, model, device, args, pred_f, cand_f, stats):
    if not batch:
        return

    texts = [x["text"] for x in batch]

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()

    for obj, prob in zip(batch, probs):
        prob_non = float(prob[0])
        prob_pol = float(prob[1])
        pred = 1 if prob_pol >= args.threshold else 0

        pred_obj = {
            "row_id": obj.get("row_id"),
            "dataset": obj.get("dataset"),
            "dataset_short_name": obj.get("dataset_short_name"),
            "dataset_source": obj.get("dataset_source", ""),
            "id": obj.get("id", ""),
            "text_chars": len(obj.get("text", "")),
            "pred_binary": pred,
            "prob_non_political": prob_non,
            "prob_political": prob_pol,
            "threshold": args.threshold,
        }

        pred_f.write(json.dumps(pred_obj, ensure_ascii=False) + "\n")

        if pred == 1:
            cand_obj = dict(obj)
            cand_obj["modernbert_binary_pred"] = pred
            cand_obj["modernbert_prob_non_political"] = prob_non
            cand_obj["modernbert_prob_political"] = prob_pol
            cand_obj["modernbert_threshold"] = args.threshold
            cand_f.write(json.dumps(cand_obj, ensure_ascii=False) + "\n")
            stats["political"] += 1
        else:
            stats["non_political"] += 1

        stats["written"] += 1


def main():
    args = parse_args()

    input_path = Path(args.input)
    pred_path = Path(args.predictions_out)
    cand_path = Path(args.candidates_out)

    pred_path.parent.mkdir(parents=True, exist_ok=True)
    cand_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Model:", MODEL_DIR)
    print("Input:", input_path)
    print("Predictions out:", pred_path)
    print("Candidates out:", cand_path)
    print("Device:", device)
    print("Batch size:", args.batch_size)
    print("Max length:", args.max_length)
    print("Threshold:", args.threshold)
    print("Limit:", args.limit)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    stats = {
        "read": 0,
        "written": 0,
        "empty_text": 0,
        "non_political": 0,
        "political": 0,
    }

    batch = []

    with input_path.open("r", encoding="utf-8") as f_in, \
         pred_path.open("w", encoding="utf-8") as pred_f, \
         cand_path.open("w", encoding="utf-8") as cand_f:

        for line in tqdm(f_in, desc="Classifying"):
            if args.limit is not None and stats["read"] >= args.limit:
                break

            stats["read"] += 1

            try:
                obj = json.loads(line)
            except Exception:
                continue

            text = str(obj.get("text", "")).strip()
            if not text:
                stats["empty_text"] += 1
                continue

            obj["text"] = text
            batch.append(obj)

            if len(batch) >= args.batch_size:
                flush_batch(batch, tokenizer, model, device, args, pred_f, cand_f, stats)
                batch = []

        flush_batch(batch, tokenizer, model, device, args, pred_f, cand_f, stats)

    print("\nDone.")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
