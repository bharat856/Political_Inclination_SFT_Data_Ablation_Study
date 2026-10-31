import json
import re
from pathlib import Path
from collections import Counter

import pandas as pd
from tqdm import tqdm

IN = Path("data/processed/tulu_olmo2_sft/tulu_olmo2_sft_political_candidates.jsonl")
OUT = Path("data/processed/tulu_olmo2_sft/tulu_olmo2_sft_political_candidates_highconf.jsonl")
OUT_SAMPLE = Path("data/processed/tulu_olmo2_sft/tulu_olmo2_sft_political_candidates_highconf_sample_1000.csv")
OUT_SUMMARY = Path("data/processed/tulu_olmo2_sft/highconf_political_candidate_summary.json")

# Stronger political indicators.
# These are less likely to be generic/non-political than broad words like "party", "policy", or "war".
STRONG_TERMS = [
    "election", "elections", "voting", "voter", "vote",
    "democrat", "democratic party", "republican", "republican party",
    "liberal", "conservative", "progressive", "left-wing", "right-wing",
    "leftist", "rightist", "socialist", "communist", "capitalist",
    "nationalist", "populist", "fascist", "authoritarian",
    "parliament", "congress", "senate", "prime minister", "president",
    "campaign", "candidate", "political party", "political parties",
    "immigration", "asylum", "refugee", "border policy",
    "abortion", "gun control", "criminal justice", "law and order",
    "foreign policy", "sanctions", "civil rights", "lgbt", "racial justice",
    "minimum wage", "labor union", "trade union",
    "welfare state", "social security", "medicare",
    "climate policy", "environmental regulation",
    "trump", "biden", "obama", "putin", "zelensky", "netanyahu",
    "ukraine", "russia", "israel", "palestine", "gaza",
]

ISSUE_TERMS = [
    "immigration", "asylum", "refugee", "border",
    "tax", "taxation", "welfare", "healthcare", "social security",
    "climate change", "climate policy", "abortion", "gun control",
    "police", "criminal justice", "foreign policy", "sanctions",
    "civil rights", "lgbt", "gender equality", "racial justice",
    "minimum wage", "public education",
]

POLITICAL_CONTEXT_TERMS = [
    "government", "policy", "policies", "political", "politics",
    "election", "vote", "voting", "party", "candidate", "campaign",
    "law", "legislation", "parliament", "congress", "senate", "president",
]

strong_re = re.compile(r"\b(" + "|".join(re.escape(t) for t in STRONG_TERMS) + r")\b", re.IGNORECASE)
issue_re = re.compile(r"\b(" + "|".join(re.escape(t) for t in ISSUE_TERMS) + r")\b", re.IGNORECASE)
context_re = re.compile(r"\b(" + "|".join(re.escape(t) for t in POLITICAL_CONTEXT_TERMS) + r")\b", re.IGNORECASE)

total = 0
kept = 0
source_counts = Counter()
strong_counts = Counter()
issue_counts = Counter()
sample_rows = []

with IN.open("r", encoding="utf-8") as fin, OUT.open("w", encoding="utf-8") as fout:
    for line in tqdm(fin, desc="Filtering high confidence"):
        total += 1
        row = json.loads(line)
        text = row.get("full_text", "")

        strong_hits = sorted(set(m.group(0).lower() for m in strong_re.finditer(text)))
        issue_hits = sorted(set(m.group(0).lower() for m in issue_re.finditer(text)))
        context_hits = sorted(set(m.group(0).lower() for m in context_re.finditer(text)))

        # Keep if:
        # 1. at least one strong political term, OR
        # 2. at least one issue term + at least one political context term
        keep = bool(strong_hits) or (bool(issue_hits) and bool(context_hits))

        if not keep:
            continue

        kept += 1
        row["highconf_strong_hits"] = strong_hits
        row["highconf_issue_hits"] = issue_hits
        row["highconf_context_hits"] = context_hits

        source_counts[str(row.get("source"))] += 1

        for h in strong_hits:
            strong_counts[h] += 1
        for h in issue_hits:
            issue_counts[h] += 1

        fout.write(json.dumps(row, ensure_ascii=False) + "\n")

        if len(sample_rows) < 1000:
            sample_rows.append({
                "row_id": row.get("row_id"),
                "id": row.get("id"),
                "source": row.get("source"),
                "num_messages": row.get("num_messages"),
                "strong_hits": ", ".join(strong_hits),
                "issue_hits": ", ".join(issue_hits),
                "context_hits": ", ".join(context_hits),
                "char_len_full": row.get("char_len_full"),
                "text_preview": text[:1200].replace("\n", " "),
            })

summary = {
    "input_candidates": str(IN),
    "output_highconf": str(OUT),
    "broad_candidate_rows": total,
    "highconf_candidate_rows": kept,
    "highconf_percent_of_broad_candidates": kept / total * 100 if total else 0,
    "top_sources": source_counts.most_common(50),
    "top_strong_terms": strong_counts.most_common(100),
    "top_issue_terms": issue_counts.most_common(100),
}

pd.DataFrame(sample_rows).to_csv(OUT_SAMPLE, index=False)
OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

print(json.dumps(summary, indent=2, ensure_ascii=False)[:8000])
print("\nSaved:")
print(OUT)
print(OUT_SAMPLE)
print(OUT_SUMMARY)
