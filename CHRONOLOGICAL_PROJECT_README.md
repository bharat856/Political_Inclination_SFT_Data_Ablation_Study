# Chronological Project README

This project is a research pipeline for understanding and controlling political
content in instruction-tuning data. The core question is:

> Can we identify political examples inside a large SFT mixture, understand
> their ideological balance, and build interventions that reduce or rebalance
> political content without blindly removing useful data?

The project grows in stages. First, it builds external political benchmarks so
we can measure classifiers. Then it inspects and filters the Tulu/OLMo SFT
mixture. Then it manually labels samples, trains a target-domain ModernBERT
political-content gate, runs a second-stage 4-way ideology classifier, evaluates
OOD behavior, and finally creates SFT intervention datasets and LoRA training
runs. The newer active branch applies the same detection/intervention idea to
OLMo-3 DOLCI SFT data, treating `Dolci-Instruct-SFT-7B` and
`Dolci-Think-SFT-7B` as separate SFT splits throughout flattening,
classification, token accounting, and intervention JSONL creation.

## Environment

Run most Python scripts from the project root:

```bash
cd /fs/scratch/users/bharatpi/causal_political_content
```

The main virtual environment is:

```bash
source envs/classifier/bin/activate
```

Or call Python directly:

```bash
envs/classifier/bin/python code/<script>.py
```

Many scripts assume local caches and outputs under:

- `data/raw/`
- `data/processed/`
- `outputs/`
- `cache/huggingface/`
- `hf_cache/`

## Label Conventions

The project uses several related label spaces:

| Task | Labels |
| --- | --- |
| Manifesto 3-way ideology | `left`, `center`, `right` |
| Numeric 3-way ideology | `1 = left`, `2 = center/neutral`, `3 = right` |
| SFT 4-way political label | `0 = non_political`, `1 = left`, `2 = center_neutral`, `3 = right` |
| ModernBERT political gate | `0 = non_political`, `1 = political` |
| Binary ideology benchmarks | usually `left` / `right` |

## Chronological Story

### 1. Establish Political Classification Benchmarks

The first problem is measurement. Before changing SFT data, we need benchmark
tasks where political orientation can be evaluated. This part uses Manifesto
Project data and public political-bias datasets.

The Manifesto work creates left/center/right examples from party manifestos.
The external v3 benchmarks add binary ideology tasks from non-manifesto sources,
which helps test whether classifiers generalize beyond one domain.

### 2. Run Prompted LLM Classifiers

After building benchmarks, the project runs LLMs as political classifiers. These
experiments test prompt design, parsing reliability, model agreement, and whether
large instruction models can classify political text consistently.

This stage motivates the move from a simple 3-way political prompt to a 4-way
prompt that explicitly includes `non_political`. That distinction matters because
SFT mixtures contain many coding, math, table, translation, and safety examples
that may mention political words without being politically classifiable.

### 3. Train Encoder Classifiers

Prompted LLMs are expensive. The next stage trains encoder classifiers for
faster, repeatable scoring. There are two flavors:

- Manifesto/domain benchmark encoders for left/center/right or left/right tasks.
- A target-domain ModernBERT binary gate trained on manually labeled SFT samples.

The ModernBERT gate becomes stage 1 of the two-stage classifier:

```text
SFT example
  -> Stage 1: ModernBERT political/non-political gate
  -> Stage 2: LLM 4-way ideology classifier on examples that pass the gate
```

### 4. Inspect and Flatten the Tulu/OLMo SFT Mixture

The project then turns to the actual target domain: the Tulu/OLMo SFT mixture.
The mixture is inspected, flattened into row-level conversations, and filtered
with political keyword heuristics to create candidate pools.

This is intentionally broad at first. The goal is recall: find possible
political examples, then narrow them with model classifiers and manual labels.

### 5. Manual Labeling and Training Data Creation

Manual review is used to establish ground truth for SFT examples. The labels
distinguish truly non-political tasks from neutral political explanations and
left/right ideological content.

The initial labeled sample is used both to evaluate the 4-way LLM classifier and
to train the target-domain ModernBERT political gate.

### 6. Apply ModernBERT and Evaluate OOD Behavior

After training the binary gate, the project applies ModernBERT across candidate
SFT rows and separately tests it on a new 300-row OOD manual sample.

This stage revealed an important behavior: the gate can over-filter neutral
political examples. That matters because a two-stage classifier cannot recover
political examples that stage 1 rejects.

### 7. Analyze Errors and Inspect OOD Failures

The OOD analysis tools split errors into false positives and false negatives,
then support manual review of the most consequential mistakes. This helps answer
whether the model is wrong, the label guideline is ambiguous, or the task needs
a different threshold/architecture.

### 8. Build Tulu/OLMo-2 SFT Interventions

Once political-position examples are detected, the project creates intervention
datasets:

- remove all detected political-position rows
- rebalance left/right examples
- downsample overrepresented groups
- upsample underrepresented groups

These produce full SFT JSONL variants that can be used for controlled LoRA
fine-tuning experiments.

### 9. Train OLMo-2 LoRA Variants

This stage trains OLMo-2 LoRA variants on baseline or intervention data.
The long-term objective is to compare how different SFT data interventions
change downstream behavior.

### 10. Extend the Pipeline to OLMo-3 DOLCI SFT

The newer branch moves from the earlier Tulu/OLMo-2 mixture to OLMo-3 DOLCI SFT
datasets:

- `allenai/Dolci-Instruct-SFT-7B`
- `allenai/Dolci-Think-SFT-7B`

These are handled as two splits, `instruct` and `think`. They are flattened into
row-level JSONL, passed through the combined ModernBERT gate, prepared for the
LLM 4-way classifier, and then joined into a combined DOLCI classification file.

The combined 4-way labels are used to build row-removal lists:

- `no_pol`: remove `center_neutral`, `left`, and `right`
- `no_ideology`: remove only `left` and `right`
- `no_left`: remove left-labeled rows
- `no_right`: remove right-labeled rows
- `lr_balanced`: remove extra left rows so left/right counts are balanced

The DOLCI branch also adds token accounting. It counts tokens by label and split
using the Llama-3.3 tokenizer, and separately estimates how many tokens each
intervention variant removes from the full `instruct` + `think` corpus.

### 11. Build OLMo-3 DOLCI Intervention JSONLs

After row lists are created, the project streams through both DOLCI flattened
SFT files and writes intervention JSONL variants under:

```text
data/interventions/dolci_olmo3_sft/jsonl/
```

Each variant keeps separate output files for the `instruct` and `think` splits,
so later training/evaluation can preserve or explicitly control the original
DOLCI split identity.

## Current Two-Stage Classifier Path

The earlier Tulu/OLMo-2 OOD path is:

```text
outputs/modernbert_ood_test/manual_review_ood_sample_300_labeled.csv
  -> code/evaluate_modernbert_on_ood_sample.py
  -> outputs/modernbert_ood_test/modernbert_ood_predictions.csv
  -> code/evaluate_two_stage_ood.py
  -> outputs/modernbert_ood_test/two_stage_eval/
```

To complete full two-stage OOD scoring, run the 4-way classifier on the same OOD
rows and pass that output into `evaluate_two_stage_ood.py` with `--stage2`.

The newer OLMo-3 DOLCI path is:

```text
allenai/Dolci-Instruct-SFT-7B
allenai/Dolci-Think-SFT-7B
  -> code/download_inspect_olmo3_dolci_sft.py
  -> code/flatten_olmo3_dolci_sft.py
  -> code/apply_modernbert_to_dolci_olmo3_sft.py
  -> code/prepare_dolci_candidates_for_4way.py
  -> code/run_sft_political_4way_classifier.py
  -> outputs/dolci_olmo3_sft_modernbert_combined/4way_outputs/combined/
  -> code/create_dolci_olmo3_intervention_row_lists.py
  -> code/compute_dolci_olmo3_token_stats_by_label.py
  -> code/compute_dolci_intervention_token_removal_stats.py
  -> code/build_dolci_olmo3_intervention_jsonl.py
  -> data/interventions/dolci_olmo3_sft/jsonl/
```

## Python Files in Chronological Order

The sections below list every Python file in `apps/` and `code/`, ordered by the
pipeline stage where it naturally belongs.

## Stage 0: Shared Utilities and Prompt Definitions

These files are used by multiple later stages.

| File | Role |
| --- | --- |
| `code/benchmark_utils.py` | Shared helper functions for project path resolution, JSON writing, timestamping, model-name sanitization, label normalization, and parsing numeric/label outputs from models. |
| `code/benchmark_prompts.py` | Stores prompt templates for Manifesto benchmark classification and exposes `build_prompt()` for runners. |
| `code/test_model_load.py` | Small smoke test for checking whether a local or Hugging Face causal LM can load and generate. Useful before launching expensive runs. |

## Stage 1: Manifesto Data Preparation

These scripts build the first controlled political benchmark from Manifesto
Project data.

| File | Role |
| --- | --- |
| `code/select_manifestos_for_large_download.py` | Reads Manifesto metadata, maps RILE scores into left/center/right labels, and samples a balanced list of manifestos for later corpus download. |
| `code/chunk_manifestos.py` | Splits a smaller downloaded Manifesto text sample into sentence-aware chunks. It is an early chunking script. |
| `code/chunk_manifestos_strict.py` | Creates stricter word-count-based chunks from Manifesto text, giving more consistent benchmark rows. |
| `code/create_manifesto_large_chunks.py` | Builds overlapping chunks from the larger 300-document Manifesto corpus and writes large chunk-level data plus metadata. |
| `code/analyze_chunk_similarity.py` | Computes TF-IDF cosine similarity among chunks from the same document to detect repeated or overly similar chunks. |

## Stage 2: Manifesto Benchmark Construction

These scripts convert chunks into benchmark CSVs.

| File | Role |
| --- | --- |
| `code/build_benchmark_v1.py` | Creates benchmark v1 from prepared chunks, normalizes labels, balances classes if requested, adds prompt metadata, and can create splits. |
| `code/create_benchmark_v2_from_large_chunks.py` | Creates a larger balanced benchmark from the 300-document chunk pool, using similarity filtering to choose diverse chunks. |

## Stage 3: Manifesto Prompted Classifier Runs

These scripts run LLMs as left/center/right political classifiers on Manifesto
benchmarks.

| File | Role |
| --- | --- |
| `code/run_benchmark_v1.py` | General benchmark v1 runner supporting vLLM or Transformers backends, multiple models, resume mode, parsing, and error JSONs. |
| `code/run_llama_manifesto_classifier.py` | Earlier row-by-row Manifesto classifier. It includes dry-run and single-prompt test paths and is useful for debugging prompt/model behavior. |
| `code/run_original_classifier_on_benchmark.py` | Runs the original 3-way political-language prompt on a benchmark and saves parsed predictions. |

## Stage 4: Manifesto Evaluation and Error Analysis

These scripts summarize benchmark performance and inspect model disagreements.

| File | Role |
| --- | --- |
| `code/evaluate_benchmark_v1.py` | Evaluates benchmark v1 prediction CSVs and writes summary metrics, per-class metrics, confusion matrices, and a report. |
| `code/evaluate_benchmark_v2_document_majority.py` | Aggregates chunk-level benchmark v2 predictions into document-level majority predictions. |
| `code/analyze_benchmark_v2_errors.py` | Joins Llama, Gemma, and Qwen benchmark v2 outputs and writes detailed agreement/error subsets. |
| `code/analyze_benchmark_v2_errors_light.py` | Lighter benchmark v2 error-analysis variant that writes compact summaries and samples. |
| `code/create_final_benchmark_comparison.py` | Combines v1 and v2 original-classifier results into final comparison CSV/Markdown outputs. |
| `code/create_all_classifier_comparison.py` | Manually assembles a high-level comparison table across prompted LLM and encoder classifiers. |
| `code/create_classifier_benchmark_summary_v2_v3.py` | Summarizes classifier results across benchmark v2 and v3 into CSV and Markdown. |

## Stage 5: External Political Benchmark Discovery and Creation

These files search for and build non-Manifesto benchmarks, mainly for
left/right ideology classification.

| File | Role |
| --- | --- |
| `code/try_lcr_dataset_candidates.py` | Checks candidate Hugging Face datasets that might support left/center/right classification. |
| `code/inspect_political_datasets.py` | Loads and prints structure/examples from several political datasets to decide whether they are usable. |
| `code/inspect_news_political_bias.py` | Inspects a news political-bias dataset candidate. |
| `code/inspect_newsmediabias.py` | Inspects a media-bias dataset candidate. |
| `code/inspect_allsides_streaming.py` | Streams examples from an AllSides-style dataset for structure and label inspection. |
| `code/create_benchmark_v3a_political_ideologies.py` | Builds benchmark v3a from `JyotiNayak/political_ideologies`, mapping labels into left/right. |
| `code/create_benchmark_v3b_senator_tweets_binary.py` | Builds a binary benchmark from senator tweets, mapping Democrat/Republican party to left/right. |
| `code/create_benchmark_v3b_allsides.py` | Builds an AllSides-style news benchmark with political-bias labels. |

## Stage 6: External Binary Classifier Runs and Summaries

These scripts evaluate models on binary external political benchmarks.

| File | Role |
| --- | --- |
| `code/run_binary_political_classifier.py` | Runs a numeric left/right prompt over binary benchmarks and saves parsed predictions. |
| `code/run_binary_political_classifier_v2.py` | Runs a stricter left/right label prompt with improved parsing and reporting. |
| `code/reparse_binary_outputs.py` | Re-parses messy binary LLM outputs after the fact and writes `__reparsed.csv` files. |
| `code/create_benchmark_v3a_summary.py` | Summarizes prompted binary classifier outputs for benchmark v3a. |
| `code/create_benchmark_v3a_final_summary.py` | Combines prompted LLM and encoder results for benchmark v3a into final summary tables. |

## Stage 7: Encoder Training on Benchmark Data

These files train or audit encoder models for benchmark tasks.

| File | Role |
| --- | --- |
| `code/train_encoder_manifesto_classifier.py` | Trains a 3-way encoder classifier on Manifesto benchmark v2 with document-level splitting and saves chunk/document predictions. |
| `code/train_encoder_benchmark_v3a_binary.py` | Trains a binary left/right encoder on v3-style benchmark splits and writes test predictions plus summary metrics. |

## Stage 8: Inspect and Flatten the Tulu/OLMo SFT Mixture

These scripts move from external benchmarks to the target SFT data.

| File | Role |
| --- | --- |
| `code/inspect_tulu_olmo2_sft.py` | Streams quick examples from the Tulu/OLMo SFT mixture for initial inspection. |
| `code/inspect_tulu_olmo2_sft_full.py` | Loads the full mixture and writes source, dataset, role-pattern, and turn-count distributions. |
| `code/flatten_tulu_olmo2_sft.py` | Converts multi-message SFT examples into flat JSONL rows with combined text and sample files. |

## Stage 9: Create Candidate Political SFT Pools

These scripts find likely political examples in the flattened SFT mixture.

| File | Role |
| --- | --- |
| `code/prefilter_tulu_olmo2_political_candidates.py` | Uses broad political keyword filters to create a candidate pool from flattened SFT examples. |
| `code/create_high_confidence_political_candidates.py` | Applies stronger keyword/category rules to create a smaller high-confidence candidate pool. |
| `code/create_tulu_sft_highconf_classification_sample.py` | Samples high-confidence candidates into CSV/JSONL files for classifier runs and manual review. |

## Stage 10: 4-Way SFT Political Classification

This stage asks an LLM to classify SFT conversations into non-political, left,
center/neutral, or right.

| File | Role |
| --- | --- |
| `code/run_sft_political_4way_classifier.py` | Runs the 4-way SFT political prompt using vLLM and saves raw outputs plus parsed labels. |
| `code/reparse_sft_4way_outputs.py` | Re-parses 4-way classifier raw outputs with numeric and textual fallbacks. |
| `code/compare_original_vs_our_prompt.py` | Compares the older 3-way prompt against the newer 4-way prompt on a labeled file. |
| `code/compare_original_vs_our_prompt_gpus23.py` | GPU-specific variant of the prompt comparison script. |
| `code/compare_original_prompt_three_models.py` | Runs the original 3-way prompt across three configured models for comparison. |

## Stage 11: Manual Review Tooling

Manual labels are central because heuristic filters and LLM classifiers are not
trusted as ground truth.

| File | Role |
| --- | --- |
| `code/create_sft_4way_manual_review_sample.py` | Samples classifier outputs into a manual-review CSV, usually stratified to inspect model predictions. |
| `code/export_manual_review_to_txt.py` | Exports manual-review rows into a readable text batch for external labeling or review. |
| `code/import_manual_labels_from_tsv.py` | Merges externally prepared TSV labels back into the review CSV. |
| `apps/manual_sft_labeler.py` | Streamlit app for labeling SFT examples as `0/1/2/3/U`, saving progress into the labeled CSV. |
| `code/evaluate_manual_review_sample.py` | Evaluates 4-way classifier predictions against manual labels and writes error cases. |
| `code/create_binary_manual_eval.py` | Converts 4-way manual labels/predictions into a binary political/non-political or left/right evaluation view. |
| `code/evaluate_binary_manual_review.py` | Evaluates the binary manual-review file. |

## Stage 12: Train the Target-Domain ModernBERT Gate

This stage creates the first part of the two-stage classifier: a fast binary
political/non-political gate trained on manually labeled SFT examples.

| File | Role |
| --- | --- |
| `code/create_target_domain_binary_train.py` | Converts the manually labeled 4-way review file into `target_domain_binary_manual_train.csv`, mapping `0` to non-political and `1/2/3` to political. |
| `code/check_modernbert_train_val_split.py` | Audits the target-domain training file and expected train/validation split assumptions before training. |
| `code/train_encoder_target_domain_binary.py` | Trains `answerdotai/ModernBERT-base` as a binary political-content classifier and saves `outputs/target_domain_binary_modernbert/best_model`. |
| `code/apply_target_domain_binary_modernbert.py` | Applies the trained ModernBERT gate to SFT political candidates and writes probabilities/predictions. |

## Stage 13: Two-Stage Prompting After ModernBERT Filtering

This stage evaluates the full idea of using a cheap gate before an expensive
LLM ideology classifier.

| File | Role |
| --- | --- |
| `code/compare_prompts_after_modernbert_filter.py` | Merges manual labels with ModernBERT predictions, runs prompts only on rows that pass the gate, and evaluates final two-stage outcomes. |

## Stage 14: OOD Manual Sample and ModernBERT OOD Evaluation

This stage tests whether the ModernBERT gate trained on one labeled sample
generalizes to a new manually labeled OOD sample.

| File | Role |
| --- | --- |
| `code/create_modernbert_ood_manual_sample.py` | Samples 300 new SFT candidate rows excluding the earlier manual-review row IDs, creating an OOD manual-labeling file. |
| `code/evaluate_modernbert_on_ood_sample.py` | Runs the trained ModernBERT gate on the 300-row labeled OOD sample and reports binary metrics. |
| `code/analyze_modernbert_ood_errors.py` | Splits ModernBERT OOD predictions into correct, false-positive, and false-negative CSVs plus focused high-confidence error samples. |
| `apps/modernbert_ood_error_inspector.py` | Streamlit app for inspecting ModernBERT OOD errors and recording review decisions/notes. |
| `code/evaluate_two_stage_ood.py` | Evaluates the OOD sample in stage-1-only mode, or full two-stage mode when a 4-way stage-2 prediction file is supplied. |

## Stage 15: Build Tulu/OLMo-2 SFT Intervention Datasets

After detecting political-position examples, this stage builds modified SFT
datasets for causal/intervention experiments.

| File | Role |
| --- | --- |
| `code/create_sft_intervention_jsonl.py` | Creates full SFT JSONL variants: baseline, remove detected political-position rows, downsample left, and upsample right. |

## Stage 16: Train OLMo-2 LoRA Variants

These scripts fine-tune OLMo-2 on baseline/intervention data.

| File | Role |
| --- | --- |
| `code/train_olmo2_lora_sft_smoke.py` | Runs a small LoRA SFT smoke test on a 5k text-only JSONL file. |
| `code/train_olmo2_lora_sft_variant01_100k.py` | Runs the larger 100k-example LoRA SFT training variant. |

## Stage 17: Inspect and Flatten OLMo-3 DOLCI SFT

This stage extends the project from Tulu/OLMo-2 to the newer OLMo-3 DOLCI SFT
datasets. There are two source datasets, handled as separate splits:

| DOLCI dataset | Local split name | Meaning |
| --- | --- | --- |
| `allenai/Dolci-Instruct-SFT-7B` | `instruct` | Standard instruction-following SFT data. |
| `allenai/Dolci-Think-SFT-7B` | `think` | Reasoning/think-style SFT data, generally much longer per row. |

| File | Role |
| --- | --- |
| `code/download_inspect_olmo3_dolci_sft.py` | Loads both DOLCI datasets, prints row/column structure, and writes small preview JSONL files under `data/raw/olmo3_dolci_sft/`. |
| `code/flatten_olmo3_dolci_sft.py` | Converts DOLCI message lists into flat row-level JSONL files under `data/processed/dolci_olmo3_sft/`, preserving `row_id`, dataset name, source metadata, messages, and combined text. |

The main flattened outputs are:

```text
data/processed/dolci_olmo3_sft/dolci_instruct_sft_7b_flattened.jsonl
data/processed/dolci_olmo3_sft/dolci_think_sft_7b_flattened.jsonl
```

## Stage 18: Apply the Combined ModernBERT Gate to DOLCI

This stage applies the trained/combined target-domain ModernBERT binary gate to
the DOLCI `instruct` and `think` flattened files. It writes a metadata-only
prediction JSONL and a full-row political candidate JSONL.

| File | Role |
| --- | --- |
| `code/train_encoder_target_domain_binary_combined.py` | Trains the later combined ModernBERT binary gate and saves `outputs/target_domain_binary_modernbert_combined/best_model`. |
| `code/evaluate_combined_modernbert_on_corrected_ood.py` | Evaluates the combined ModernBERT gate on corrected OOD labels. |
| `code/apply_modernbert_to_dolci_olmo3_sft.py` | Applies the combined ModernBERT model to a flattened DOLCI JSONL, using a configurable probability threshold, and writes candidate rows where `prob_political >= threshold`. |

The important point is that this is a model gate, not a keyword-only filter. It
is used to avoid sending every DOLCI row to the much more expensive 4-way LLM
classifier.

## Stage 19: Prepare and Classify DOLCI Candidates with the 4-Way LLM Prompt

Rows that pass the DOLCI ModernBERT gate are converted into CSV inputs for the
same 4-way SFT political classifier used earlier.

| File | Role |
| --- | --- |
| `code/prepare_dolci_candidates_for_4way.py` | Converts DOLCI candidate JSONL files for both `instruct` and `think` into CSV files with `full_text`, ModernBERT probability, IDs, and metadata expected by the 4-way classifier. |
| `code/run_sft_political_4way_classifier.py` | Runs the 4-way prompt on DOLCI candidate CSVs, producing raw outputs and parsed labels. |
| `code/reparse_sft_4way_outputs.py` | Can be reused if DOLCI 4-way outputs need numeric/textual reparsing. |

The combined DOLCI classification file used by later stages is:

```text
outputs/dolci_olmo3_sft_modernbert_combined/4way_outputs/combined/dolci_olmo3_sft_binary_candidates_4way_combined.csv
```

Its central label column is:

```text
final_political_4way_status
```

with values like `non_political`, `center_neutral`, `left`, `right`, or missing
values treated as `unparsed`.

## Stage 20: Create DOLCI Intervention Row Lists

This stage turns the combined DOLCI 4-way classification file into row-removal
lists for intervention experiments.

| File | Role |
| --- | --- |
| `code/create_dolci_olmo3_intervention_row_lists.py` | Reads the combined DOLCI 4-way classification file, normalizes labels, and writes row-ID lists for `no_pol`, `no_ideology`, `no_left`, `no_right`, and `lr_balanced` interventions. |

The intervention meanings are:

| Variant | Removes |
| --- | --- |
| `no_pol` | `center_neutral`, `left`, and `right` rows. |
| `no_ideology` | Only `left` and `right` rows. |
| `no_left` | Only `left` rows. |
| `no_right` | Only `right` rows. |
| `lr_balanced` | Extra `left` rows so the remaining left count matches the right count. |

Row lists are written under:

```text
data/interventions/dolci_olmo3_sft/
```

## Stage 21: DOLCI Token Accounting

This stage measures how much of the DOLCI corpus is affected by each label and
each intervention. This matters because the `think` split has much longer rows
on average than the `instruct` split, so row percentages and token percentages
can tell different stories.

| File | Role |
| --- | --- |
| `code/compute_dolci_olmo3_token_stats_by_label.py` | Counts Llama-tokenized tokens for every DOLCI row and aggregates by split and 4-way label, treating rows absent from the candidate/classification file as `not_candidate`. |
| `code/compute_dolci_intervention_token_removal_stats.py` | Uses the intervention row lists to estimate removed rows and removed tokens for each intervention variant across `instruct` and `think`. |

The main outputs are:

```text
data/interventions/dolci_olmo3_sft/token_stats_by_label.json
data/interventions/dolci_olmo3_sft/intervention_token_removal_stats.json
```

## Stage 22: Build OLMo-3 DOLCI Intervention JSONLs

This stage streams through the DOLCI flattened JSONL files and writes actual
trainable intervention datasets while preserving separate `instruct` and `think`
outputs.

| File | Role |
| --- | --- |
| `code/build_dolci_olmo3_intervention_jsonl.py` | Given `--variant`, removes row IDs from the appropriate DOLCI split and writes filtered JSONL files plus a summary JSON. |

Supported variants:

```text
no_pol
no_ideology
no_left
no_right
lr_balanced
```

Outputs are written under:

```text
data/interventions/dolci_olmo3_sft/jsonl/<variant>/
```

## How the Pieces Fit Together

At the highest level, the full project flow is:

```text
1. Build political benchmarks
   -> Manifesto chunks and benchmark v1/v2
   -> external v3 binary benchmarks

2. Test political classifiers
   -> prompted LLM classifiers
   -> encoder classifiers
   -> benchmark summaries and error analyses

3. Move into SFT data
   -> inspect Tulu/OLMo mixture
   -> flatten conversations
   -> keyword-filter candidate political examples

4. Produce manual truth
   -> 4-way classifier sample
   -> Streamlit/manual labels
   -> evaluate classifier errors

5. Train target-domain gate
   -> manual labels become binary training data
   -> ModernBERT gate predicts political/non-political

6. Test two-stage behavior
   -> ModernBERT gate first
   -> LLM 4-way classifier second
   -> evaluate on in-domain and OOD manual samples

7. Intervene on Tulu/OLMo-2 SFT data
   -> remove or rebalance detected political-position examples
   -> train OLMo-2 LoRA variants

8. Extend to OLMo-3 DOLCI SFT
   -> inspect and flatten Dolci-Instruct-SFT-7B and Dolci-Think-SFT-7B
   -> apply combined ModernBERT gate to both splits
   -> classify candidates with the 4-way LLM prompt
   -> compute label/token statistics by instruct vs think split
   -> build DOLCI intervention row lists and JSONL variants
```

## Recommended Reading Order

If you want to understand the codebase from scratch, read in this order:

1. `code/benchmark_utils.py`
2. `code/benchmark_prompts.py`
3. `code/build_benchmark_v1.py`
4. `code/run_benchmark_v1.py`
5. `code/evaluate_benchmark_v1.py`
6. `code/flatten_tulu_olmo2_sft.py`
7. `code/prefilter_tulu_olmo2_political_candidates.py`
8. `code/run_sft_political_4way_classifier.py`
9. `apps/manual_sft_labeler.py`
10. `code/create_target_domain_binary_train.py`
11. `code/train_encoder_target_domain_binary.py`
12. `code/apply_target_domain_binary_modernbert.py`
13. `code/evaluate_modernbert_on_ood_sample.py`
14. `code/evaluate_two_stage_ood.py`
15. `code/create_sft_intervention_jsonl.py`
16. `code/train_olmo2_lora_sft_variant01_100k.py`
17. `code/download_inspect_olmo3_dolci_sft.py`
18. `code/flatten_olmo3_dolci_sft.py`
19. `code/apply_modernbert_to_dolci_olmo3_sft.py`
20. `code/prepare_dolci_candidates_for_4way.py`
21. `code/create_dolci_olmo3_intervention_row_lists.py`
22. `code/compute_dolci_olmo3_token_stats_by_label.py`
23. `code/compute_dolci_intervention_token_removal_stats.py`
24. `code/build_dolci_olmo3_intervention_jsonl.py`

That reading path follows the intellectual arc of the project: define the
classification problem, build measurement, inspect SFT data, label examples,
train the gate, evaluate OOD behavior, then create intervention data. The final
DOLCI scripts show how the same idea is scaled to OLMo-3 `instruct` and `think`
SFT data with split-aware token accounting.

## Current Research Interpretation

The project is not merely classifying politics for its own sake. It is trying to
answer a causal data question:

> What happens if political-position examples are removed or rebalanced in SFT
> training data?

To answer that, the pipeline needs reliable detection. The current two-stage
classifier is a practical compromise:

- ModernBERT is cheap and fast, but may miss neutral political examples.
- The LLM 4-way classifier is more expressive, but expensive to run everywhere.
- Manual labels provide the audit trail for both.

The OOD work is especially important because it tests whether the gate learned a
real concept of political content or just memorized patterns from the first
manual sample.

The OLMo-3 DOLCI branch is the current scale-up of that idea. It keeps the
`instruct` and `think` SFT splits separate because their row lengths and training
roles differ. The `think` split contains much longer reasoning-style examples,
so token-level accounting is necessary: row counts alone can understate or
overstate the amount of training signal removed by an intervention.
