from pathlib import Path
import json

manifest_path = Path("data/interventions/dolci_olmo3_sft/intervention_manifest.json")
token_path = Path("data/interventions/dolci_olmo3_sft/intervention_token_removal_stats.json")
out_path = Path("data/interventions/dolci_olmo3_sft/final_intervention_report.json")

manifest = json.loads(manifest_path.read_text())
token_stats = json.loads(token_path.read_text())

report = {
    "dataset": {
        "name": "Dolci OLMo-3 SFT",
        "total_rows": token_stats["total_rows"],
        "total_tokens": token_stats["total_tokens"],
    },
    "variants": {}
}

for variant, row_info in manifest.items():
    tok_info = token_stats["variants"][variant]
    report["variants"][variant] = {
        **row_info,
        "removed_tokens": tok_info["removed_tokens"],
        "removed_row_pct": tok_info["removed_row_pct"],
        "removed_token_pct": tok_info["removed_token_pct"],
        "avg_tokens_removed_row": tok_info["avg_tokens_removed_row"],
        "token_by_split": tok_info["by_split"],
    }

out_path.write_text(json.dumps(report, indent=2))
print("Saved:", out_path)
print(json.dumps(report, indent=2))
