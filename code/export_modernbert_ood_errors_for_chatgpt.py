from pathlib import Path
import pandas as pd

IN = Path("outputs/modernbert_ood_test/modernbert_ood_predictions.csv")
OUT = Path("outputs/modernbert_ood_test/error_analysis/modernbert_ood_errors_for_chatgpt.txt")
OUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN)

df["error_type"] = "correct"

df.loc[
    (df["gold_binary"] == 0) & (df["modernbert_pred_binary"] == 1),
    "error_type",
] = "false_positive"

df.loc[
    (df["gold_binary"] == 1) & (df["modernbert_pred_binary"] == 0),
    "error_type",
] = "false_negative"

errors = df[df["error_type"] != "correct"].copy()
errors = errors.sort_values(["error_type", "prob_political"], ascending=[True, True])


def clean_label(x):
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if s.lower() in {"nan", "none"}:
        return ""
    return s


errors["manual_label_norm"] = errors["manual_label"].apply(clean_label)

lines = []
lines.append("Please relabel these examples using exactly one label:")
lines.append("0 = non_political")
lines.append("1 = left / progressive political")
lines.append("2 = center_neutral / factual political")
lines.append("3 = right / conservative political")
lines.append("U = unclear / unsure")
lines.append("")
lines.append("Important annotation rule:")
lines.append("- If political words are only decorative around a math/coding/generic task, label 0.")
lines.append("- If the actual variable/topic is policy, government, voter behavior, war, foreign affairs, public budget, or political institutions, label 2 unless it has clear left/right framing.")
lines.append("- If it frames free markets, nationalism, law/order, anti-bureaucracy, traditional values, or conservative policy positively, label 3.")
lines.append("- If it frames welfare, equality, redistribution, labor rights, minority rights, climate action, or progressive policy positively, label 1.")
lines.append("")
lines.append("Return only this TSV format:")
lines.append("row_id<TAB>final_label<TAB>short_note")
lines.append("")
lines.append("=" * 120)

for i, (_, r) in enumerate(errors.iterrows(), start=1):
    row_id = r.get("row_id", "")
    manual_label = r.get("manual_label_norm", "")
    error_type = r.get("error_type", "")
    prob_political = r.get("prob_political", "")
    pred = r.get("modernbert_pred_binary", "")
    gold = r.get("gold_binary", "")
    notes = r.get("manual_notes", "")
    text = str(r.get("text_preview", "")).replace("\r", " ").strip()

    lines.append("")
    lines.append(f"ITEM {i}")
    lines.append(f"ROW_ID: {row_id}")
    lines.append(f"CURRENT_MANUAL_LABEL: {manual_label}")
    lines.append(f"GOLD_BINARY: {gold}")
    lines.append(f"MODERNBERT_PRED_BINARY: {pred}")
    lines.append(f"ERROR_TYPE: {error_type}")
    lines.append(f"PROB_POLITICAL: {prob_political}")
    lines.append(f"MANUAL_NOTES: {notes}")
    lines.append("TEXT:")
    lines.append(text[:2500])
    lines.append("-" * 120)

OUT.write_text("\n".join(lines), encoding="utf-8")

print("Saved:", OUT)
print("Rows exported:", len(errors))
print(errors["error_type"].value_counts())
print("\nManual labels in exported errors:")
print(errors["manual_label_norm"].value_counts(dropna=False))
