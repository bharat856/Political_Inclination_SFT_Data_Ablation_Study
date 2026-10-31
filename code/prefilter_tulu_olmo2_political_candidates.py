import json
import re
from pathlib import Path
from collections import Counter

import pandas as pd
from tqdm import tqdm

IN = Path("data/processed/tulu_olmo2_sft/tulu_olmo2_sft_flattened.jsonl")
OUT_DIR = Path("data/processed/tulu_olmo2_sft")
OUT_CAND = OUT_DIR / "tulu_olmo2_sft_political_candidates.jsonl"
OUT_SAMPLE = OUT_DIR / "tulu_olmo2_sft_political_candidates_sample_1000.csv"
OUT_SUMMARY = OUT_DIR / "political_candidate_prefilter_summary.json"

KEYWORDS = [
    # general politics
    "politic", "political", "policy", "policies", "government", "governments",
    "election", "elections", "vote", "voting", "voter", "democracy", "democratic",
    "republican", "democrat", "liberal", "conservative", "left-wing", "right-wing",
    "leftist", "rightist", "progressive", "socialist", "communist", "capitalist",
    "nationalist", "populist", "fascist", "authoritarian",

    # institutions / actors
    "president", "prime minister", "parliament", "congress", "senate", "minister",
    "party", "parties", "campaign", "candidate", "lawmakers", "legislation",

    # issue areas
    "immigration", "migrant", "refugee", "border", "asylum",
    "tax", "taxation", "welfare", "healthcare", "medicare", "social security",
    "climate policy", "climate change", "green energy", "environmental regulation",
    "abortion", "gun control", "police", "criminal justice", "law and order",
    "foreign policy", "war", "military intervention", "sanctions",
    "civil rights", "lgbt", "gender equality", "racial justice",
    "minimum wage", "labor union", "trade union", "public education",

    # countries / geopolitics often political
    "trump", "biden", "obama", "putin", "zelensky", "netanyahu",
    "ukraine", "russia", "israel", "palestine", "gaza",
]

pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in KEYWORDS) + r")\b", re.IGNORECASE)

total = 0
matched = 0
source_counts = Counter()
keyword_counts = Counter()
lengths = []

sample_rows = []

with IN.open("r", encoding="utf-8") as fin, OUT_CAND.open("w", encoding="utf-8") as fout:
    for line in tqdm(fin, desc="Scanning"):
        total += 1
        row = json.loads(line)
        text = row.get("full_text", "")

        hits = sorted(set(m.group(0).lower() for m in pattern.finditer(text)))

        if not hits:
            continue

        matched += 1
        row["political_keyword_hits"] = hits
        row["num_political_keyword_hits"] = len(hits)

        source_counts[str(row.get("source"))] += 1
        lengths.append(row.get("char_len_full", 0))

        for h in hits:
            keyword_counts[h] += 1

        fout.write(json.dumps(row, ensure_ascii=False) + "\n")

        if len(sample_rows) < 1000:
            sample_rows.append({
                "row_id": row.get("row_id"),
                "id": row.get("id"),
                "source": row.get("source"),
                "num_messages": row.get("num_messages"),
                "keyword_hits": ", ".join(hits),
                "char_len_full": row.get("char_len_full"),
                "text_preview": text[:1000].replace("\n", " "),
            })

summary = {
    "input": str(IN),
    "output_candidates": str(OUT_CAND),
    "total_rows": total,
    "candidate_rows": matched,
    "candidate_percent": matched / total * 100 if total else 0,
    "top_candidate_sources": source_counts.most_common(50),
    "top_keywords": keyword_counts.most_common(100),
    "avg_candidate_char_len": sum(lengths) / len(lengths) if lengths else 0,
}

pd.DataFrame(sample_rows).to_csv(OUT_SAMPLE, index=False)
OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

print(json.dumps(summary, indent=2, ensure_ascii=False)[:8000])
print("\nSaved:")
print(OUT_CAND)
print(OUT_SAMPLE)
print(OUT_SUMMARY)
