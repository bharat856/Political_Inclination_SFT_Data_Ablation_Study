# Project Presentation Report: Political Content in SFT Data

## 1. One-Minute Summary

This project studies how much political content exists inside large instruction-tuning datasets, how to identify it reliably, and how to create controlled data interventions without blindly deleting useful examples.

The work progressed in four stages:

1. Build and evaluate political classification benchmarks.
2. Test prompted LLM classifiers and encoder classifiers.
3. Apply a two-stage classifier to SFT data: first political vs non-political, then ideology label.
4. Build DOLCI/OLMo-3 SFT intervention datasets that remove or rebalance political examples.

The current result is a working pipeline that can scan millions of SFT rows, label candidate political examples, quantify token impact, and generate intervention JSONL datasets for downstream fine-tuning experiments.

## 1A. Obsidian Progress Tracker

### Done Till Now

- [x] Selected the starting political-text dataset: Manifesto Project / MARPOR / CMP data from [manifesto-project.wzb.eu](https://manifesto-project.wzb.eu/).
- [x] Downloaded/exported Manifesto metadata and corpus text into the local project workspace.
- [x] Used Manifesto metadata, especially RILE scores, to build left/center/right benchmark labels.
- [x] Split manifesto documents into meaningful text chunks instead of classifying only whole long documents.
- [x] Created stricter chunking scripts so chunks are more comparable in length and easier to evaluate.
- [x] Built benchmark v1 as a small pilot political-orientation benchmark.
- [x] Built benchmark v2 as a larger and more diverse Manifesto benchmark.
- [x] Added content/ideology labels to benchmark rows: left, center, right.
- [x] Used the RILE-style left/right scaling idea from the Manifesto literature and the two reference papers:
  - [Nikolaev et al. 2023, EMNLP: Multilingual estimation of political-party positioning](https://aclanthology.org/2023.emnlp-main.591.pdf)
  - [NLP4DH 2025 paper](https://aclanthology.org/2025.nlp4dh-1.38.pdf)
- [x] Implemented the core RILE interpretation:

```text
RILE = (R - L) / (R + L + O)
```

where `R` is the count/percentage of right-leaning statements, `L` is the count/percentage of left-leaning statements, and `O` is the count/percentage of other statements.

- [x] Tested prompted LLM classifiers on Manifesto benchmarks.
- [x] Compared Llama, Gemma, and Qwen-style prompted classifiers.
- [x] Found that Llama and Gemma worked better than Qwen on the Manifesto setup, while Qwen often collapsed to one class.
- [x] Built additional public political benchmarks beyond Manifesto, including clean left/right ideology data and senator tweets.
- [x] Trained/evaluated encoder classifiers such as ModernBERT and DeBERTa variants.
- [x] Inspected the Tulu/OLMo-2 SFT dataset.
- [x] Flattened SFT conversations into row-level text examples.
- [x] Built broad keyword filters to find likely political SFT candidates.
- [x] Built high-confidence political candidate filters.
- [x] Created manual review samples for SFT political classification.
- [x] Manually labeled SFT examples into non-political, left, center/neutral, and right categories.
- [x] Evaluated the 4-way LLM SFT political classifier against manual labels.
- [x] Trained a target-domain ModernBERT binary gate for political vs non-political SFT examples.
- [x] Evaluated ModernBERT on in-domain and OOD manual samples.
- [x] Found an important limitation: ModernBERT can miss neutral political examples out of distribution.
- [x] Applied the pipeline to DOLCI/OLMo-3 SFT data.
- [x] Classified DOLCI binary candidates with the 4-way political classifier.
- [x] Computed DOLCI token statistics by label.
- [x] Created DOLCI intervention row lists.
- [x] Created DOLCI intervention JSONL datasets: `no_pol`, `no_ideology`, `no_left`, `no_right`, and `lr_balanced`.
- [x] Ran baseline/smoke LoRA SFT experiments to verify the training setup.

### Current Phase

The project is currently at the transition point between dataset construction/classification and downstream intervention evaluation.

The main completed artifact is not just a classifier. It is an end-to-end pipeline:

```text
Manifesto benchmarks
  -> classifier evaluation
  -> SFT candidate filtering
  -> manual labels
  -> ModernBERT political gate
  -> LLM 4-way ideology labels
  -> DOLCI token statistics
  -> intervention JSONL datasets
```

### Next Phase: Running Classifier on SFT Dataset

- [ ] Decide the exact SFT dataset/version to use for the next full run.
- [ ] Run the political/non-political gate over the full target SFT dataset.
- [ ] Run the 4-way classifier only on rows that pass the candidate/gate stage.
- [ ] Save full prediction files with row IDs, source dataset, text preview, raw model output, parsed label, and confidence/probability where available.
- [ ] Compute label distributions by source dataset.
- [ ] Compute token statistics by label and by source.
- [ ] Manually review a new OOD sample from classifier outputs.
- [ ] Use the OOD manual labels to recalibrate the ModernBERT threshold or retrain the gate.
- [ ] Decide whether to optimize for higher recall or higher precision.
- [ ] Re-run intervention row-list creation after final classifier settings are chosen.
- [ ] Train LoRA models on each intervention dataset.
- [ ] Evaluate downstream model behavior on political prompts, neutrality, factual political questions, refusal/safety behavior, and ideology-sensitive generations.
- [ ] Compare model behavior across baseline, no-politics, no-ideology, no-left, no-right, and left/right-balanced variants.
- [ ] Write the final results section explaining whether data interventions changed downstream behavior.

### What Is Left To Do

The biggest remaining work is downstream validation. The current pipeline can identify and remove/rebalance political examples, but the final research question requires testing whether models trained on those intervention datasets actually behave differently.

Priority order:

1. Improve the stage-1 political gate using the OOD findings.
2. Finalize the SFT dataset and classifier threshold.
3. Run full classification and token accounting.
4. Train LoRA intervention models.
5. Evaluate behavioral differences.
6. Convert results into thesis/report figures and tables.

## 2. Research Question

The central question is:

Can we identify political examples inside a large SFT mixture, understand their ideological balance, and build interventions that reduce or rebalance political content without removing too much useful data?

This matters because instruction-tuning data can influence model behavior. If political examples are imbalanced or overrepresented, downstream models may learn unwanted political tendencies. But removing all text with political keywords is too crude because many examples only mention political terms in translation, math, history, safety, or factual contexts.

## 3. Label Spaces Used

Different stages use different labels:

| Task | Labels |
| --- | --- |
| Manifesto ideology | left, center, right |
| Numeric ideology | 1 = left, 2 = center/neutral, 3 = right |
| SFT 4-way classifier | non_political, left, center_neutral, right |
| ModernBERT gate | non_political, political |
| Binary public benchmarks | left, right |

The important conceptual distinction is between political topic and political position:

Political topic means the text discusses elections, parties, policy, government, ideology, history, etc.

Political position means the example takes, asks for, or contains a clear ideological stance such as left or right. Neutral political explanations are often center_neutral.

## 4. Stage 1: Building Benchmark Datasets

The project first created external benchmarks so classifiers could be evaluated before applying them to SFT data.

### Manifesto Benchmarks

Manifesto Project data was used to create left/center/right examples. The Manifesto Project provides document-level ideological scores, and the scripts chunk manifesto documents into smaller text examples.

Important scripts:

| Script | Purpose |
| --- | --- |
| `select_manifestos_for_large_download.py` | Selects balanced manifesto documents using RILE ideology scores |
| `create_manifesto_large_chunks.py` | Creates chunks from a larger 300-document manifesto corpus |
| `build_benchmark_v1.py` | Builds the first small benchmark |
| `create_benchmark_v2_from_large_chunks.py` | Builds a larger and harder benchmark |

Important benchmark sizes:

| Benchmark | Rows | Documents | Description |
| --- | ---: | ---: | --- |
| benchmark_v1 | 126 | 27 | Small pilot benchmark |
| benchmark_v2 | 900 | 193 | Larger, more diverse benchmark |

### External Binary Benchmarks

The project also created non-manifesto benchmarks:

| Benchmark | Source | Task |
| --- | --- | --- |
| benchmark_v3a | `JyotiNayak/political_ideologies` | left vs right |
| benchmark_v3b | senator tweets | Democrat(left) vs Republican(right) |
| AllSides-style benchmark | news/media bias data | left/center/right-style media bias |

These were useful because manifesto chunks have a weakness: document-level labels are copied to chunks, so a chunk may not itself express the whole document's ideology.

## 5. Stage 2: Prompted LLM Classifiers

Several LLMs were tested as political classifiers:

| Model | Role |
| --- | --- |
| Llama-3.3-70B-bnb-4bit | Main prompted classifier |
| Gemma-4-31B-AWQ | Comparison model |
| Qwen3.5-27B-AWQ | Comparison model |

### Manifesto Benchmark Results

| Benchmark | Model | Chunk Accuracy | Document-Majority Accuracy | Pattern |
| --- | --- | ---: | ---: | --- |
| v1 | Llama-3.3-70B | 0.6905 | 0.7407 | Best document-level result |
| v1 | Gemma-4-31B | 0.6905 | 0.6667 | Similar chunk accuracy |
| v1 | Qwen3.5-27B | 0.3333 | 0.3333 | Collapsed to all-left |
| v2 | Llama-3.3-70B | 0.5156 | 0.5440 | Overpredicted center |
| v2 | Gemma-4-31B | 0.5044 | 0.5337 | Overpredicted right |
| v2 | Qwen3.5-27B | 0.3333 | 0.3523 | Collapsed to all-left |

Interpretation:

Benchmark v2 was harder and more realistic than v1. The drop from v1 to v2 suggests that the task is noisy, especially because document-level RILE labels are imperfect when applied to individual chunks. Llama was the best prompted classifier overall, but not strong enough to rely on alone for every stage.

### Clean Binary Benchmark Results

On cleaner left/right ideology data, classifiers performed much better:

| Dataset | Model | Test Rows | Accuracy | Notes |
| --- | --- | ---: | ---: | --- |
| v3a clean ideology | ModernBERT-large | 320 | 1.0000 | Perfect on clean benchmark |
| v3a clean ideology | Llama-3.3-70B | 320 | 0.9437 | Strong prompted result |
| v3b senator tweets | ModernBERT-large | 320 | 0.8688 | Best on harder real-world text |
| v3b senator tweets | Llama-3.3-70B | 320 | 0.8344 | Strong but overpredicted left |

Interpretation:

The classifier can work well when labels are clean and text-level. The harder problem is not simply political ideology classification; it is identifying political content inside messy SFT examples.

## 6. Stage 3: Tulu/OLMo-2 SFT Data Inspection

The next stage moved from benchmarks to the actual target domain: instruction-tuning data.

Dataset inspected:

`allenai/tulu-3-sft-olmo-2-mixture`

Important counts:

| Item | Count |
| --- | ---: |
| Total rows | 939,344 |
| Most common role pattern | user, assistant |
| Rows with user-assistant only | 896,081 |
| Percent user-assistant only | 95.39% |

The dataset contains many sources, including math, code, WildChat, Aya, FLAN, safety, and PersonaHub-style data. This variety is why a simple keyword search is not enough.

## 7. Stage 4: Candidate Filtering for Tulu/OLMo-2

The first SFT filtering stage used broad political keyword matching to maximize recall.

| Stage | Rows | Percent |
| --- | ---: | ---: |
| Total SFT rows | 939,344 | 100.00% |
| Broad political candidates | 107,720 | 11.47% of total |
| High-confidence candidates | 44,159 | 40.99% of broad candidates |

Common broad keywords included:

political, government, party, policy, war, healthcare, campaign, candidate, president, border, tax, election, climate change, democratic, progressive, conservative.

Common high-confidence terms included:

campaign, candidate, president, election, progressive, conservative, vote, immigration, republican, civil rights, Trump, foreign policy, abortion, right-wing, left-wing.

Interpretation:

The broad pass intentionally catches too much. Many false positives are expected because examples can contain political terms without being political-position examples. This motivates manual labeling and a learned classifier.

## 8. Stage 5: Manual Labeling and 4-Way LLM Classifier Evaluation

A manually reviewed sample was created from SFT political candidates.

Manual review file:

`outputs/tulu_olmo2_sft_political_classification_sample/manual_review_sample_200_labeled.csv`

The 4-way LLM classifier labels examples as:

non_political, left, center_neutral, right.

On 192 labeled and parsed rows:

| Metric | Value |
| --- | ---: |
| 4-way accuracy | 0.5833 |
| Non-political recall | 0.4757 |
| Left recall | 0.9062 |
| Center-neutral recall | 0.5833 |
| Right recall | 0.8571 |

Confusion pattern:

The model often over-called political labels on non-political examples. For example, translation tasks, math word problems, or safety refusals with political words could be predicted as left/right/center even when the manual label was non_political.

Binary version of the same evaluation:

| Metric | Value |
| --- | ---: |
| Rows | 187 |
| Accuracy | 0.7219 |
| Political recall | 0.9885 |
| Non-political recall | 0.4900 |

Interpretation:

The prompted LLM is very sensitive: it rarely misses political examples, but it over-classifies many non-political examples as political. This is useful for high recall but not enough for clean filtering.

## 9. Stage 6: Target-Domain ModernBERT Political Gate

To make filtering cheaper and more consistent, a binary ModernBERT classifier was trained on manually labeled target-domain SFT examples.

Task:

non_political vs political.

Initial test split:

| Metric | Value |
| --- | ---: |
| Rows | 52 |
| Accuracy | 0.9423 |
| Non-political recall | 0.9000 |
| Political recall | 1.0000 |

Combined/later test split:

| Metric | Value |
| --- | ---: |
| Rows | 127 |
| Accuracy | 0.8898 |
| Non-political recall | 0.9125 |
| Political recall | 0.8511 |

Interpretation:

ModernBERT performed well on in-domain test splits. It became stage 1 of the two-stage pipeline:

SFT row -> ModernBERT political/non-political gate -> LLM 4-way ideology classifier on predicted political rows.

## 10. Stage 7: OOD Evaluation of ModernBERT

A separate 300-row OOD manual sample was used to test generalization.

OOD file:

`outputs/modernbert_ood_test/manual_review_ood_sample_300_labeled.csv`

Rows with gold labels: 299.

Gold 4-way distribution:

| Label | Rows |
| --- | ---: |
| non_political | 196 |
| center_neutral | 98 |
| right | 5 |

Stage 1 predictions:

| Predicted Label | Rows |
| --- | ---: |
| non_political | 219 |
| political | 80 |

ModernBERT OOD binary metrics:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.6756 |
| Macro F1 | 0.6181 |
| Non-political precision | 0.7260 |
| Non-political recall | 0.8112 |
| Political precision | 0.5375 |
| Political recall | 0.4175 |

Confusion matrix:

| Gold \ Predicted | Non-political | Political |
| --- | ---: | ---: |
| Non-political | 159 | 37 |
| Political | 60 | 43 |

Important finding:

The OOD test exposed a recall problem. ModernBERT rejected many neutral political examples as non-political. Since the two-stage system only sends stage-1 positives to the 4-way classifier, stage 2 cannot recover examples that stage 1 misses.

This is one of the most important research findings so far.

## 11. Stage 8: DOLCI/OLMo-3 SFT Pipeline

The most recent work applies a similar idea to DOLCI/OLMo-3 SFT data.

Two DOLCI splits were processed:

| Split | Rows |
| --- | ---: |
| instruct | 2,152,112 |
| think | 2,268,178 |
| Total | 4,420,290 |

The binary candidate set contained 187,617 rows.

LLM 4-way classification of those candidates produced:

| Label | Rows |
| --- | ---: |
| non_political | 88,634 |
| center_neutral | 60,916 |
| left | 28,861 |
| right | 3,955 |
| unparsed | 5,251 |

Confirmed political examples:

| Group | Rows |
| --- | ---: |
| LLM-confirmed political: center_neutral + left + right | 93,732 |
| Ideological: left + right | 32,816 |
| Left | 28,861 |
| Right | 3,955 |
| Left/right ratio | 7.30 : 1 |

Interpretation:

Among examples classified as ideological, left-labeled examples are much more frequent than right-labeled examples. This motivates balancing interventions.

## 12. DOLCI Token Statistics

A token-count pass was run with the Llama-3.3 tokenizer.

Total:

| Item | Count |
| --- | ---: |
| Rows | 4,420,290 |
| Tokens | 24,105,484,423 |

Overall label distribution:

| Label | Rows | Tokens | Row Percent | Token Percent | Avg Tokens/Row |
| --- | ---: | ---: | ---: | ---: | ---: |
| not_candidate | 4,232,673 | 23,943,393,194 | 95.76% | 99.33% | 5,656.80 |
| non_political | 88,634 | 76,462,908 | 2.01% | 0.32% | 862.68 |
| center_neutral | 60,916 | 53,138,151 | 1.38% | 0.22% | 872.32 |
| left | 28,861 | 21,683,009 | 0.65% | 0.09% | 751.29 |
| right | 3,955 | 5,289,204 | 0.09% | 0.02% | 1,337.35 |
| unparsed | 5,251 | 5,517,957 | 0.12% | 0.02% | 1,050.84 |

Key takeaway:

Political examples are a small fraction of the full DOLCI corpus by tokens. Removing all confirmed political examples removes 93,732 rows, but these rows are much shorter on average than the long non-candidate think examples, so the token share removed is small.

## 13. Stage 9: DOLCI Intervention Datasets

The project created row lists and full JSONL intervention variants.

### Row Removal Lists

| Intervention | Rows Removed | Meaning |
| --- | ---: | --- |
| remove_no_pol | 93,732 | Remove center_neutral, left, and right |
| remove_no_ideology | 32,816 | Remove only left and right |
| remove_left | 28,861 | Remove left examples |
| remove_right | 3,955 | Remove right examples |
| remove_extra_left_for_lr_balance | 24,906 | Remove enough left rows to balance left/right |

### JSONL Variants

| Variant | Total Kept | Total Removed | Purpose |
| --- | ---: | ---: | --- |
| no_pol | 4,326,558 | 93,732 | Remove all LLM-confirmed political topic/position examples |
| no_ideology | 4,387,474 | 32,816 | Remove only ideological left/right examples |
| no_left | 4,391,429 | 28,861 | Remove left-labeled examples |
| no_right | 4,416,335 | 3,955 | Remove right-labeled examples |
| lr_balanced | 4,395,384 | 24,906 | Downsample left so left and right are balanced |

These files are saved under:

`data/interventions/dolci_olmo3_sft/jsonl/`

Interpretation:

This gives several experimental conditions. Instead of asking only "what happens if we remove politics?", we can ask finer questions:

- What happens if all political examples are removed?
- What happens if only ideological examples are removed?
- What happens if left examples are removed?
- What happens if right examples are removed?
- What happens if left/right counts are balanced?

## 14. LoRA SFT Runs

The project also includes OLMo-2 LoRA SFT runs.

Existing run artifacts:

| Run | Purpose |
| --- | --- |
| `variant_01_full_original_smoke_5k_lora` | Smoke test on a 5k subset |
| `variant_01_full_original_sft_100k_lora` | Larger 100k-example baseline run |

These runs establish that the training setup can execute before launching more expensive intervention runs.

## 15. What Has Been Achieved

The project now has:

1. Benchmark datasets for political classification.
2. Prompted LLM classifier comparisons.
3. Encoder classifier experiments.
4. Tulu/OLMo-2 SFT inspection and candidate filtering.
5. Manual labels for SFT political classification.
6. A target-domain ModernBERT political gate.
7. OOD analysis showing a real generalization problem.
8. DOLCI/OLMo-3 candidate classification.
9. Token-level estimates of how much data each label represents.
10. Intervention JSONL datasets for controlled fine-tuning experiments.

## 16. Main Findings

### Finding 1: Manifesto chunk classification is noisy

Performance drops from benchmark v1 to benchmark v2. This is probably because document-level ideology labels do not always match individual chunks.

### Finding 2: Clean text-level ideology classification is much easier

ModernBERT-large reached 1.0 accuracy on the clean v3a benchmark and 0.8688 on senator tweets. This means the models can learn political distinctions when labels match the text well.

### Finding 3: Keyword filtering has high recall but low precision

The broad keyword pass found 107,720 Tulu candidates, but many were math, translation, coding, or safety examples with political terms. This is expected and motivates learned filtering.

### Finding 4: Prompted LLM classification is sensitive

The 4-way LLM classifier catches many political examples but overpredicts politics on non-political rows.

### Finding 5: ModernBERT works in-domain but struggles OOD

ModernBERT looked strong on held-out target-domain splits, but OOD evaluation showed low political recall, especially for neutral political examples.

### Finding 6: DOLCI ideological examples are left-heavy

Among DOLCI ideological examples, left rows were 28,861 and right rows were 3,955, a left/right ratio of about 7.30:1.

### Finding 7: Political rows are a small token fraction

Even though tens of thousands of political rows are found, they are a small fraction of the 24.1B-token DOLCI corpus.

## 17. Limitations

1. Manual labels are limited in size.

The SFT manual review sample is useful but small. More labels would improve confidence.

2. Manifesto labels are weak at chunk level.

RILE scores are document-level labels. A chunk from a right-leaning manifesto can contain a neutral or left-sounding sentence, and vice versa.

3. Prompted LLM labels are not ground truth.

The DOLCI intervention labels come from model classification. They should be treated as high-quality heuristic labels, not perfect human labels.

4. Stage 1 errors are unrecoverable in the two-stage pipeline.

If ModernBERT labels a political row as non-political, the 4-way classifier never sees it.

5. "Political" is inherently ambiguous.

Some rows are historical, factual, safety-related, or translation tasks. Reasonable annotators may disagree.

6. Intervention effects still need downstream evaluation.

The data interventions have been created, but the final question is how fine-tuned models behave after training on these variants.

## 18. How to Explain the Method in a Presentation

Use this simple storyline:

First, I built political classification benchmarks because I needed a way to measure classifiers before using them on SFT data.

Second, I tested prompted LLMs and encoder models. I found that clean text-level labels work well, but manifesto chunk labels are noisy.

Third, I inspected the SFT mixture and used keyword filtering only as a recall-oriented first pass.

Fourth, I manually labeled SFT examples and trained a target-domain ModernBERT binary gate.

Fifth, I evaluated the gate on OOD data and found that it can miss neutral political examples, which is an important limitation.

Finally, I applied the pipeline to DOLCI/OLMo-3 data, counted tokens by label, and created intervention datasets for future LoRA fine-tuning comparisons.

## 19. Likely Questions and Answers

### Q1. What is the main contribution of your work?

The main contribution is an end-to-end pipeline for identifying political examples in SFT data, measuring ideological imbalance, and creating controlled intervention datasets. It goes beyond simple keyword removal by combining candidate filtering, manual review, learned classifiers, LLM classification, token accounting, and JSONL dataset generation.

### Q2. Why not just remove all examples with political keywords?

Because many examples with political words are not political training examples. For example, translation tasks, math word problems, factual history, coding examples, and safety refusals may mention "president", "election", or "policy" without teaching a political position. Removing all keyword matches would delete too much useful data.

### Q3. Why use a two-stage classifier?

The two-stage design is efficient. ModernBERT is cheap and fast for political/non-political filtering. The LLM 4-way classifier is more expensive but can provide richer labels, so it is only applied to rows that pass the binary gate or candidate filter.

### Q4. What did the OOD test show?

It showed that the ModernBERT gate does not generalize perfectly. On 299 OOD labeled rows, political recall was only 0.4175. This means it missed many political examples, especially neutral political ones. That is important because stage 2 cannot fix stage 1 false negatives.

### Q5. Why is left/right imbalance important?

If an SFT corpus contains far more left-labeled than right-labeled ideological examples, fine-tuning may expose the model to an imbalanced political distribution. In DOLCI, the ideological subset had 28,861 left rows and 3,955 right rows, about 7.30 left rows per right row.

### Q6. Does "left" mean the model will become left-wing?

Not directly. The label describes the content of training examples, not automatically the model's behavior. The next step is to fine-tune on intervention variants and evaluate whether model outputs change.

### Q7. How much data is removed in the strongest intervention?

The no_pol intervention removes 93,732 rows from 4,420,290 total rows. It keeps 4,326,558 rows. By row count this is about 2.12% of the corpus. By token count, confirmed political labels are a smaller share because many non-candidate think examples are very long.

### Q8. What is the difference between no_pol and no_ideology?

no_pol removes center_neutral, left, and right examples. It removes all LLM-confirmed political examples.

no_ideology removes only left and right examples. It keeps neutral political explanations.

### Q9. Why keep center_neutral in some interventions?

Neutral political explanations may be useful for factual knowledge and civic reasoning. Removing only left/right ideology lets us test whether ideological positioning matters separately from political factual content.

### Q10. What is lr_balanced?

lr_balanced removes extra left-labeled rows until left and right counts are balanced. It does not remove all ideology; it tries to equalize ideological representation.

### Q11. Are the labels perfect?

No. The labels are a mix of manual labels, model predictions, and heuristic candidate filters. The project explicitly evaluates errors and treats label noise as part of the research problem.

### Q12. What would you do next?

The next steps are:

1. Increase manual labels, especially OOD neutral political examples.
2. Improve or recalibrate the ModernBERT gate for higher political recall.
3. Run LoRA fine-tuning on the intervention JSONLs.
4. Evaluate downstream model behavior on political neutrality, factuality, refusal behavior, and ideology-sensitive prompts.

## 20. Important Files to Mention

| File or Directory | Why it matters |
| --- | --- |
| `CHRONOLOGICAL_PROJECT_README.md` | High-level project story |
| `code/README.md` | Script guide |
| `outputs/classifier_comparison/classifier_benchmark_summary_v2_v3.md` | Benchmark comparison |
| `outputs/original_classifier/final_comparison/v1_v2_benchmark_comparison.md` | Manifesto classifier comparison |
| `data/processed/tulu_olmo2_sft/inspection_summary.json` | Tulu dataset inspection |
| `data/processed/tulu_olmo2_sft/political_candidate_prefilter_summary.json` | Broad keyword candidate counts |
| `data/processed/tulu_olmo2_sft/highconf_political_candidate_summary.json` | High-confidence candidate counts |
| `outputs/modernbert_ood_test/two_stage_eval/two_stage_ood_summary.json` | OOD evaluation |
| `outputs/dolci_olmo3_sft_modernbert_combined/4way_outputs/combined/dolci_olmo3_sft_4way_summary.json` | DOLCI 4-way classification summary |
| `data/interventions/dolci_olmo3_sft/token_stats_by_label.json` | DOLCI token statistics |
| `data/interventions/dolci_olmo3_sft/intervention_manifest.json` | Final intervention JSONL manifest |

## 21. Short Script You Can Say Out Loud

My project is about measuring and controlling political content in instruction-tuning data. I started by building political classification benchmarks from manifesto and public political datasets, because I needed to understand how reliable different classifiers were before applying them to SFT data.

The benchmark results showed that clean text-level ideology classification can work well, but manifesto chunk classification is noisy because the labels are document-level. After that, I moved to the actual SFT setting. I inspected the Tulu/OLMo mixture, used broad keyword filtering to get high-recall political candidates, manually labeled samples, and trained a ModernBERT binary political gate.

The important finding from OOD evaluation is that the gate works reasonably in-domain but misses some neutral political examples out of distribution. That means a two-stage system has to be designed carefully because stage 1 false negatives cannot be recovered later.

Most recently, I applied the pipeline to DOLCI/OLMo-3 SFT data. I classified 187,617 binary candidates with a 4-way classifier and found 93,732 LLM-confirmed political examples. Among ideological examples, the data was left-heavy: 28,861 left examples versus 3,955 right examples. I then computed token statistics and generated intervention datasets that remove all political content, remove only ideology, remove left, remove right, or balance left and right. These datasets are ready for controlled LoRA fine-tuning experiments.
