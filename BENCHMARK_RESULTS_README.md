# Benchmark and Classifier Results Summary

This file consolidates the benchmark, classifier, and intervention results found so far. It is meant to answer:

- Which dataset was used?
- Which model or method was tested?
- What result did we get?
- What did that result imply?
- What was the next step?

## 1. Overall Progression

The project moved through this sequence:

```text
Manifesto benchmark construction
  -> prompted LLM benchmark runs
  -> encoder benchmark runs
  -> external clean ideology benchmarks
  -> SFT political candidate filtering
  -> SFT manual review and 4-way classification
  -> ModernBERT political/non-political gate
  -> OOD evaluation
  -> DOLCI/OLMo-3 SFT classification
  -> DOLCI intervention variants
```

The key lesson is that political classification is easy on clean text-level labels, but much harder on real SFT data and manifesto chunks where labels are noisy or context-dependent.

## 2. Manifesto Benchmark Results

### Dataset

Manifesto Project / MARPOR data from:

```text
data/raw/manifesto/MPDataset_MPDS2025a.csv
data/raw/manifesto/corpus/large_300_manifesto_corpus.csv
```

Labels were created by using the Manifesto-provided `rile` score:

```text
rile < -5   -> left
-5 to +5    -> center
rile > +5   -> right
```

The labels are document-level labels that were propagated to chunks.

### Benchmark v1 and v2 Prompted LLM Results

| Benchmark | Dataset | Rows | Documents | Model | Type | Chunk Accuracy | Document-Majority Accuracy | Main Pattern |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | --- |
| benchmark_v1 | Manifesto chunks | 126 | 27 | Llama-3.3-70B-bnb-4bit | Prompted LLM | 0.6905 | 0.7407 | Best document-level result |
| benchmark_v1 | Manifesto chunks | 126 | 27 | Gemma-4-31B-AWQ | Prompted LLM | 0.6905 | 0.6667 | Similar chunk accuracy, stronger on right |
| benchmark_v1 | Manifesto chunks | 126 | 27 | Qwen3.5-27B-AWQ | Prompted LLM | 0.3333 | 0.3333 | Collapsed to all-left |
| benchmark_v2 | Manifesto chunks | 900 | 193 | Llama-3.3-70B-bnb-4bit | Prompted LLM | 0.5156 | 0.5440 | Overpredicted center |
| benchmark_v2 | Manifesto chunks | 900 | 193 | Gemma-4-31B-AWQ | Prompted LLM | 0.5044 | 0.5337 | Overpredicted right |
| benchmark_v2 | Manifesto chunks | 900 | 193 | Qwen3.5-27B-AWQ | Prompted LLM | 0.3333 | 0.3523 | Collapsed to all-left |

### Reasoning

Benchmark v2 is larger and more diverse than v1, so the drop from v1 to v2 is expected. The main issue is that RILE labels are document-level, but the model sees chunks. A chunk may not fully represent the party manifesto's overall ideology.

### Next Step

Use Manifesto benchmarks as an initial sanity check, but do not rely on them as the only evaluation. Move to cleaner text-level external benchmarks and target-domain SFT manual labels.

## 3. Manifesto Encoder Results

### Dataset

```text
data/processed/benchmark_v2/benchmark_v2.csv
```

Task:

```text
left / center / right
```

### Results

| Dataset | Model | Type | Test Setting | Chunk Accuracy | Macro F1 | Document-Majority Accuracy | Pattern |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| benchmark_v2 | Llama-3.3-70B-bnb-4bit | Prompted LLM | Full set | 0.5156 | n/a | 0.5440 | Overpredicted center |
| benchmark_v2 | Gemma-4-31B-AWQ | Prompted LLM | Full set | 0.5044 | n/a | 0.5337 | Overpredicted right |
| benchmark_v2 | Qwen3.5-27B-AWQ | Prompted LLM | Full set | 0.3333 | n/a | 0.3523 | All-left collapse |
| benchmark_v2 | ModernBERT-large | Fine-tuned encoder | Held-out document split | 0.4194 | 0.4190 | 0.4000 | Weak, not collapsed |
| benchmark_v2 | DeBERTa-v3-large | Fine-tuned encoder | Held-out document split | 0.3629 | 0.1775 | 0.3333 | All-right collapse |
| benchmark_v2 | mDeBERTa-v3-base | Fine-tuned encoder | Held-out document split | 0.3065 | 0.1564 | 0.3333 | All-center collapse |

### Reasoning

Encoders did not solve the Manifesto chunk-label problem. ModernBERT was weak but did not collapse; DeBERTa and mDeBERTa collapsed to a single class. This reinforced that weak document labels copied to chunks are a major limitation.

### Next Step

Create or use benchmarks with cleaner text-level labels.

## 4. External Clean Ideology Benchmarks

## 4.1 Benchmark v3a: Clean Left/Right Ideology

### Dataset

```text
JyotiNayak/political_ideologies
data/processed/benchmark_v3/benchmark_v3a_political_ideologies_binary.*
```

Task:

```text
left / right
```

### Results

| Dataset | Model | Type | Test Rows | Parsed Rows | Parse Rate | Accuracy All Rows | Accuracy Parsed Only | Macro F1 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| benchmark_v3a | ModernBERT-large | Fine-tuned encoder | 320 | 320 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Perfect on clean benchmark |
| benchmark_v3a | Llama-3.3-70B-bnb-4bit | Prompted LLM | 320 | 320 | 1.0000 | 0.9437 | 0.9437 | n/a | Strong, all outputs parsed |
| benchmark_v3a | Qwen3.5-27B-AWQ | Prompted LLM | 320 | 238 | 0.7438 | 0.7312 | 0.9832 | n/a | Strong on parsed rows, parsing failures from reasoning tags |
| benchmark_v3a | DeBERTa-v3-large | Fine-tuned encoder | 320 | 320 | 1.0000 | 0.5000 | 0.5000 | 0.3333 | Collapsed to all-left |
| benchmark_v3a | mDeBERTa-v3-base | Fine-tuned encoder | 320 | 320 | 1.0000 | 0.5000 | 0.5000 | 0.3333 | Collapsed to all-right |
| benchmark_v3a | Gemma-4-31B-AWQ | Prompted LLM | 320 | 67 | 0.2094 | 0.1437 | 0.6866 | n/a | Unstable output format |

### Reasoning

When labels are clean and text-level, ModernBERT and Llama perform very well. This means the models can learn political distinctions when the benchmark is well aligned with the task.

### Next Step

Use clean benchmarks to validate model capacity, but move back to SFT data where the real challenge is separating political content from non-political examples with political words.

## 4.2 Benchmark v3b: Senator Tweets

### Dataset

```text
m-newhauser/senator-tweets
```

Task:

```text
Democrat(left) / Republican(right)
```

### Results

| Dataset | Model | Type | Test Rows | Accuracy | Macro F1 | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| benchmark_v3b senator tweets | ModernBERT-large | Fine-tuned encoder | 320 | 0.8688 | 0.8687 | Stronger than prompted Llama on harder real-world political text |
| benchmark_v3b senator tweets | Llama-3.3-70B-bnb-4bit | Prompted LLM | 320 | 0.8344 | n/a | Parsed all rows, overpredicted left |

### Reasoning

This was a more realistic text-level political benchmark than v3a. ModernBERT performed well, suggesting an encoder can be useful as a fast classifier when trained on appropriate labels.

### Next Step

Train a target-domain encoder on manually labeled SFT rows.

## 5. SFT 4-Way Manual Review Results

### Dataset

Tulu/OLMo-2 SFT candidate sample:

```text
outputs/tulu_olmo2_sft_political_classification_sample/manual_review_sample_200_labeled.csv
```

Task:

```text
0 = non_political
1 = left
2 = center_neutral
3 = right
```

### 4-Way LLM Classifier Result

| Dataset | Model | Type | Evaluated Rows | Accuracy | Main Pattern |
| --- | --- | --- | ---: | ---: | --- |
| Tulu/OLMo-2 SFT manual sample | Llama-3.3-70B-bnb-4bit | Prompted 4-way LLM | 192 | 0.5833 | High sensitivity, overpredicts political labels on non-political examples |

Per-class behavior:

| Gold Class | Recall | Notes |
| --- | ---: | --- |
| non_political | 0.4757 | Many non-political examples with political words were called political |
| left | 0.9062 | Strong recall |
| center_neutral | 0.5833 | Moderate recall |
| right | 0.8571 | Strong recall but small support |

### Binary Political/Non-Political View

| Dataset | Model | Rows | Accuracy | Non-Political Recall | Political Recall | Main Pattern |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Tulu/OLMo-2 SFT manual sample | Llama-3.3-70B-bnb-4bit 4-way collapsed to binary | 187 | 0.7219 | 0.4900 | 0.9885 | Very high political recall but many false positives |

### Reasoning

The LLM classifier is useful for high recall. It almost never misses political examples in the binary view, but it over-flags non-political examples such as math, translation, and safety refusals that contain political words.

### Next Step

Train a cheaper target-domain binary gate, ModernBERT, using manual labels.

## 6. Target-Domain ModernBERT Gate Results

### Dataset

Manual SFT labels converted to binary:

```text
non_political
political
```

### Results

| Dataset | Model | Type | Rows | Accuracy | Non-Political Recall | Political Recall | Confusion Matrix | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Initial target-domain split | ModernBERT-base | Fine-tuned encoder | 52 | 0.9423 | 0.9000 | 1.0000 | `[[27, 3], [0, 22]]` | Very strong, but small test split |
| Combined target-domain split | ModernBERT-base / combined | Fine-tuned encoder | 127 | 0.8898 | 0.9125 | 0.8511 | `[[73, 7], [7, 40]]` | More realistic, still useful |

### Reasoning

ModernBERT is much cheaper than an LLM and performs well on in-domain manual splits. This supports using it as stage 1 in a two-stage classifier.

### Next Step

Evaluate whether the gate generalizes out of distribution. This is necessary because stage 1 false negatives cannot be recovered by stage 2.

## 7. ModernBERT OOD Evaluation

### Dataset

```text
outputs/modernbert_ood_test/manual_review_ood_sample_300_labeled.csv
```

Rows with gold labels:

```text
299
```

Gold distribution:

| Label | Rows |
| --- | ---: |
| non_political | 196 |
| center_neutral | 98 |
| right | 5 |

### Result

| Dataset | Model | Type | Rows | Accuracy | Macro F1 | Non-Political Recall | Political Recall | Confusion Matrix |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| OOD SFT manual sample | ModernBERT target-domain gate | Binary encoder | 299 | 0.6756 | 0.6181 | 0.8112 | 0.4175 | `[[159, 37], [60, 43]]` |

### Reasoning

The OOD test revealed the main weakness: ModernBERT misses many neutral political examples. Since the two-stage pipeline only runs the expensive LLM classifier on rows that pass the gate, false negatives at stage 1 are unrecoverable.

### Next Step

Improve or recalibrate the gate, especially for neutral political examples. The DOLCI branch uses the combined gate but still treats results as model-generated labels requiring caution.

## 8. DOLCI/OLMo-3 SFT Classification Results

### Dataset

```text
allenai/Dolci-Instruct-SFT-7B
allenai/Dolci-Think-SFT-7B
```

Flattened files:

```text
data/processed/dolci_olmo3_sft/dolci_instruct_sft_7b_flattened.jsonl
data/processed/dolci_olmo3_sft/dolci_think_sft_7b_flattened.jsonl
```

Pipeline:

```text
DOLCI rows
  -> combined ModernBERT binary gate
  -> Llama-3.3-70B 4-way classifier on binary candidates
  -> intervention row lists
```

### 4-Way Candidate Classification Summary

| Label | Rows |
| --- | ---: |
| non_political | 88,634 |
| center_neutral | 60,916 |
| left | 28,861 |
| right | 3,955 |
| unparsed | 5,251 |
| total binary candidates | 187,617 |

By split:

| Label | Instruct Rows | Think Rows |
| --- | ---: | ---: |
| non_political | 70,375 | 18,259 |
| center_neutral | 38,146 | 22,770 |
| left | 16,859 | 12,002 |
| right | 2,833 | 1,122 |
| unparsed | 3,533 | 1,718 |

Key derived counts:

| Quantity | Value |
| --- | ---: |
| LLM-confirmed political: `center_neutral + left + right` | 93,732 |
| Ideological: `left + right` | 32,816 |
| Left rows | 28,861 |
| Right rows | 3,955 |
| Left/right ratio | 7.30 : 1 |

### Reasoning

DOLCI ideological examples are strongly left-heavy under the current classifier. This motivated creating multiple intervention variants instead of only one removal dataset.

### Next Step

Build intervention datasets for controlled fine-tuning experiments.

## 9. DOLCI/OLMo-3 Intervention Variants

### Dataset

Original total:

| Split | Rows |
| --- | ---: |
| instruct | 2,152,112 |
| think | 2,268,178 |
| total | 4,420,290 |

### Variants Built

| Variant | Removes | Instruct Removed | Think Removed | Total Removed | Total Kept | Purpose |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `no_pol` | `center_neutral`, `left`, `right` | 57,838 | 35,894 | 93,732 | 4,326,558 | Remove all LLM-confirmed political content |
| `no_ideology` | `left`, `right` | 19,692 | 13,124 | 32,816 | 4,387,474 | Remove ideological content but keep neutral political explanations |
| `no_left` | `left` | 16,859 | 12,002 | 28,861 | 4,391,429 | Test effect of removing left-labeled examples |
| `no_right` | `right` | 2,833 | 1,122 | 3,955 | 4,416,335 | Test effect of removing right-labeled examples |
| `lr_balanced` | extra `left` rows | 14,550 | 10,356 | 24,906 | 4,395,384 | Downsample left so remaining left/right counts are balanced |

### Reasoning

Different interventions answer different causal questions:

- `no_pol`: What if all detected political content is removed?
- `no_ideology`: What if only ideological positioning is removed?
- `no_left`: What if left-labeled examples are removed?
- `no_right`: What if right-labeled examples are removed?
- `lr_balanced`: What if the ideological distribution is balanced instead of removed?

### Next Step

Fine-tune models on each intervention variant and evaluate whether downstream behavior changes.

## 10. Fine-Tuning Variants So Far

### Dataset and Model

Base model:

```text
hf_cache/allenai_OLMo-2-1124-7B
```

Training style:

```text
QLoRA / LoRA SFT using TRL SFTTrainer
```

### Runs

| Run | Data | Script | Output | Purpose |
| --- | --- | --- | --- | --- |
| smoke 5k baseline | `baseline_original_smoke_5k_textonly.jsonl` | `code/train_olmo2_lora_sft_smoke.py` | `outputs/sft_runs/variant_01_full_original_smoke_5k_lora/` | Confirm training setup works |
| 100k baseline | `variant_01_full_original_sft_100k_textonly.jsonl` | `code/train_olmo2_lora_sft_variant01_100k.py` | `outputs/sft_runs/variant_01_full_original_sft_100k_lora/` | Larger baseline run before intervention runs |

### Reasoning

These are baseline training runs. They prove the LoRA SFT stack can run before spending compute on all intervention variants.

### Next Step

Run equivalent fine-tuning jobs for DOLCI intervention datasets, then compare model behavior across baseline and variants.

## 11. Current Interpretation

The project has found three important things:

1. Clean ideology classification is feasible.

ModernBERT and Llama perform strongly on clean text-level left/right benchmarks.

2. Real SFT political filtering is harder.

SFT data contains many examples with political words that are not political-position examples. Manual review and OOD evaluation are necessary.

3. DOLCI political candidates are left-heavy.

Among DOLCI ideological examples classified so far, left-labeled rows outnumber right-labeled rows by about 7.30 to 1.

## 12. Recommended Next Steps

1. Treat Manifesto chunk results as diagnostic, not final proof.
2. Improve or recalibrate the ModernBERT gate using OOD findings.
3. Keep using Llama-3.3-70B for 4-way classification when richer labels are needed.
4. Fine-tune models on the DOLCI intervention variants.
5. Evaluate downstream behavior across:
   - baseline
   - `no_pol`
   - `no_ideology`
   - `no_left`
   - `no_right`
   - `lr_balanced`
6. Report not only classifier metrics, but also how model behavior changes after data intervention.

## 13. Source Files Used

| Result Area | Source Artifact |
| --- | --- |
| Manifesto v1/v2 prompted results | `outputs/original_classifier/final_comparison/v1_v2_benchmark_comparison.md` |
| Benchmark v2/v3 comparison | `outputs/classifier_comparison/classifier_benchmark_summary_v2_v3.md` |
| All classifier comparison | `outputs/classifier_comparison/all_classifier_comparison.md` |
| Benchmark v3a final summary | `outputs/benchmark_v3a_binary/summary/benchmark_v3a_final_summary.md` |
| SFT manual review labels | `outputs/tulu_olmo2_sft_political_classification_sample/manual_review_sample_200_labeled.csv` |
| Target-domain ModernBERT test predictions | `outputs/target_domain_binary_modernbert/test_predictions.csv` |
| Combined ModernBERT test predictions | `outputs/target_domain_binary_modernbert_combined/test_predictions.csv` |
| OOD ModernBERT summary | `outputs/modernbert_ood_test/two_stage_eval/two_stage_ood_summary.json` |
| DOLCI 4-way summary | `outputs/dolci_olmo3_sft_modernbert_combined/4way_outputs/combined/dolci_olmo3_sft_4way_summary.json` |
| DOLCI intervention manifest | `data/interventions/dolci_olmo3_sft/intervention_manifest.json` |

