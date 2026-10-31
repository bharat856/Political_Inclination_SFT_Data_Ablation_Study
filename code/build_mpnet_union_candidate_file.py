from pathlib import Path
import json

INPUTS = {
    "instruct": Path(
        "outputs/dolci_olmo3_sft_union_mpnet_t04/instruct/"
        "dolci_instruct_sft_7b_union_t04_candidates_for_4way.jsonl"
    ),
    "think": Path(
        "outputs/dolci_olmo3_sft_union_mpnet_t04/think/"
        "dolci_think_sft_7b_union_t04_candidates_for_4way.jsonl"
    ),
}

OUTPUT = Path(
    "data/interventions/dolci_olmo3_sft_union_mpnet_t04/"
    "all_union_candidates_for_3way.jsonl"
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

counts = {}

with OUTPUT.open("w", encoding="utf-8") as out:
    for split, path in INPUTS.items():
        n = 0

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                obj = json.loads(line)

                if "row_id" not in obj:
                    raise KeyError(
                        f"Missing row_id in {path}: {obj.keys()}"
                    )

                obj["split"] = split
                obj["document_id"] = f"{split}:{int(obj['row_id'])}"

                out.write(
                    json.dumps(
                        obj,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                n += 1

        counts[split] = n

print("Counts:", counts)
print("Total:", sum(counts.values()))
print("Saved:", OUTPUT)
