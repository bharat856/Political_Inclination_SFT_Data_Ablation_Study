from datasets import load_dataset
from pathlib import Path
import json
from tqdm import tqdm

DATASET = "allenai/tulu-3-sft-olmo-2-mixture"

OUT_DIR = Path("data/processed/tulu_olmo2_sft")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSONL = OUT_DIR / "tulu_olmo2_sft_flattened.jsonl"
OUT_SAMPLE = OUT_DIR / "tulu_olmo2_sft_flattened_sample_1000.jsonl"

print("Loading:", DATASET)
ds = load_dataset(DATASET, split="train")

def flatten_messages(messages):
    parts = []
    user_parts = []
    assistant_parts = []

    for m in messages:
        role = m.get("role", "")
        content = str(m.get("content", "")).strip()

        if not content:
            continue

        parts.append(f"{role.upper()}:\n{content}")

        if role == "user":
            user_parts.append(content)
        elif role == "assistant":
            assistant_parts.append(content)

    return {
        "full_text": "\n\n".join(parts),
        "user_text": "\n\n".join(user_parts),
        "assistant_text": "\n\n".join(assistant_parts),
    }

with OUT_JSONL.open("w", encoding="utf-8") as f_all, OUT_SAMPLE.open("w", encoding="utf-8") as f_sample:
    for i, ex in enumerate(tqdm(ds, total=len(ds))):
        messages = ex.get("messages", [])
        flat = flatten_messages(messages)

        row = {
            "row_id": i,
            "id": ex.get("id"),
            "source": ex.get("source"),
            "dataset": ex.get("dataset"),
            "num_messages": len(messages),
            "roles": [m.get("role") for m in messages],
            "full_text": flat["full_text"],
            "user_text": flat["user_text"],
            "assistant_text": flat["assistant_text"],
            "char_len_full": len(flat["full_text"]),
            "char_len_user": len(flat["user_text"]),
            "char_len_assistant": len(flat["assistant_text"]),
        }

        line = json.dumps(row, ensure_ascii=False)
        f_all.write(line + "\n")

        if i < 1000:
            f_sample.write(line + "\n")

print("Saved:")
print(OUT_JSONL)
print(OUT_SAMPLE)
