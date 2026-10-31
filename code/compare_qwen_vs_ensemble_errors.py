from pathlib import Path
import pandas as pd

BASE = Path("outputs/validation_tanise/ensemble_v4")

qwen = pd.read_csv(BASE / "qwen_predictions.csv")

ensemble = pd.read_csv(
    BASE / "qwen_llama_gemma" / "ensemble_predictions.csv"
)

# align by row order
qwen = qwen.reset_index(drop=True)
ensemble = ensemble.reset_index(drop=True)

gold = qwen["gold_label"].astype(str).str.lower()

qwen_pred = qwen["parsed_prediction_label"].astype(str).str.lower()
ens_pred = ensemble["ensemble_prediction"].astype(str).str.lower()

df = pd.DataFrame({
    "text": qwen["text"],
    "gold": gold,
    "qwen": qwen_pred,
    "ensemble": ens_pred,
})

df["qwen_correct"] = df.qwen == df.gold
df["ensemble_correct"] = df.ensemble == df.gold

both_correct = df[df.qwen_correct & df.ensemble_correct]
both_wrong = df[~df.qwen_correct & ~df.ensemble_correct]
fixed = df[~df.qwen_correct & df.ensemble_correct]
regressed = df[df.qwen_correct & ~df.ensemble_correct]

OUT = BASE / "qwen_llama_gemma"
OUT.mkdir(exist_ok=True, parents=True)

summary = pd.DataFrame({
    "category": [
        "both_correct",
        "both_wrong",
        "fixed_by_ensemble",
        "regressed_by_ensemble",
    ],
    "count": [
        len(both_correct),
        len(both_wrong),
        len(fixed),
        len(regressed),
    ],
})

summary.to_csv(OUT / "ensemble_vs_qwen_summary.csv", index=False)

fixed.to_csv(
    OUT / "ensemble_fixed_examples.csv",
    index=False,
)

regressed.to_csv(
    OUT / "ensemble_regressed_examples.csv",
    index=False,
)

print(summary)

print()
print("Net improvement:",
      len(fixed) - len(regressed))
