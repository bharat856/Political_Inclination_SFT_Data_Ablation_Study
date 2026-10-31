import re
from pathlib import Path
import pandas as pd

IN = Path("outputs/tulu_olmo2_sft_political_classification_sample/sft_political_4way___fs_scratch_users_bharatpi_causal_political_content_hf_cache_unsloth_Llama_3_3_70B_Instruct_bnb_4bit.csv")
OUT = IN.with_name(IN.stem + "__reparsed.csv")

def reparse(raw):
    if pd.isna(raw):
        return None

    s = str(raw).strip()
    lower = s.lower()

    # Find explicit numbers anywhere.
    m = re.search(r"(?<!\d)[0123](?!\d)", s)
    if m:
        return int(m.group(0))

    # Textual fallbacks.
    if "non-political" in lower or "not politically" in lower or "not political" in lower:
        return 0
    if "center" in lower or "centre" in lower or "neutral" in lower or "balanced" in lower:
        return 2
    if "left" in lower or "progressive" in lower or "liberal" in lower:
        return 1
    if "right" in lower or "conservative" in lower:
        return 3

    return None

label_map = {
    0: "non_political",
    1: "left",
    2: "center_neutral",
    3: "right",
}

df = pd.read_csv(IN)

df["sft_political_label_id_reparsed"] = df["sft_political_raw_output"].apply(reparse)
df["sft_political_label_reparsed"] = df["sft_political_label_id_reparsed"].map(label_map)
df["parse_ok_reparsed"] = df["sft_political_label_id_reparsed"].notna()

df.to_csv(OUT, index=False)

print("Input:", IN)
print("Output:", OUT)
print("Original parsed:", int(df["parse_ok"].sum()), "/", len(df))
print("Reparsed:", int(df["parse_ok_reparsed"].sum()), "/", len(df))

print("\nReparsed label distribution:")
print(df["sft_political_label_reparsed"].value_counts(dropna=False))

print("\nStill unparsed examples:")
cols = ["row_id", "id", "source", "sft_political_raw_output", "full_text"]
still = df[df["parse_ok_reparsed"] == False][cols].head(30)
for _, r in still.iterrows():
    print("=" * 100)
    print("row_id:", r["row_id"])
    print("source:", r["source"])
    print("raw:", repr(r["sft_political_raw_output"]))
    print(str(r["full_text"])[:700].replace("\n", " "))
