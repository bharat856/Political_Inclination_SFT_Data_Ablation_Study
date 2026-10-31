from pathlib import Path
from datasets import load_dataset
import json

DATASETS = [
    {
        "hf_name": "allenai/Dolci-Instruct-SFT-7B",
        "short_name": "dolci_instruct_sft_7b",
    },
    {
        "hf_name": "allenai/Dolci-Think-SFT-7B",
        "short_name": "dolci_think_sft_7b",
    },
]

OUT_DIR = Path("data/processed/dolci_olmo3_sft")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def messages_to_text(messages):
    parts = []

    for m in messages:
        role = str(m.get("role", "")).strip().upper()
        content = str(m.get("content", "")).strip()

        if not content:
            continue

        if role:
            parts.append(f"{role}:\n{content}")
        else:
            parts.append(content)

    return "\n\n".join(parts).strip()


for cfg in DATASETS:
    hf_name = cfg["hf_name"]
    short_name = cfg["short_name"]

    print("=" * 120)
    print("Loading:", hf_name)

    ds = load_dataset(hf_name, split="train")

    out_path = OUT_DIR / f"{short_name}_flattened.jsonl"
    sample_path = OUT_DIR / f"{short_name}_flattened_sample_1000.jsonl"

    n_written = 0
    n_empty = 0

    with out_path.open("w", encoding="utf-8") as f_out, sample_path.open("w", encoding="utf-8") as f_sample:
        for i, row in enumerate(ds):
            messages = row.get("messages", [])
            text = messages_to_text(messages)

            if not text:
                n_empty += 1
                continue

            obj = {
                "row_id": i,
                "dataset": hf_name,
                "dataset_short_name": short_name,
                "dataset_source": row.get("dataset_source", ""),
                "id": row.get("id", ""),
                "text": text,
                "messages": messages,
            }

            line = json.dumps(obj, ensure_ascii=False)
            f_out.write(line + "\n")

            if n_written < 1000:
                f_sample.write(line + "\n")

            n_written += 1

            if n_written % 100000 == 0:
                print(f"{short_name}: written {n_written:,}")

    print("Saved:", out_path)
    print("Saved sample:", sample_path)
    print("Rows written:", n_written)
    print("Empty skipped:", n_empty)

print("\nDone.")
