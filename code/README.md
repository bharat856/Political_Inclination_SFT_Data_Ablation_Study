# Code Folder Guide

This folder contains the scripts for a political-content classification research
pipeline. The work has three main threads:

1. Build benchmark datasets from political manifestos and public political-bias
   datasets.
2. Run prompted LLM classifiers and encoder classifiers on those benchmarks.
3. Identify, label, and intervene on political examples inside the Tulu/OLMo SFT
   mixture, then train LoRA SFT variants.

Most scripts are meant to be run from the project root:

```bash
cd /fs/scratch/users/bharatpi/causal_political_content
python code/<script_name>.py
```

Some scripts are lightweight pandas utilities. Others load large models with
vLLM, Transformers, PEFT, TRL, or Hugging Face datasets and should be run only
in an environment with the right GPU, cache, and package setup.

## Repository Data Flow

The rough pipeline looks like this:

```text
Manifesto raw metadata/text
  -> select_manifestos_for_large_download.py
  -> create_manifesto_large_chunks.py / chunk_manifestos*.py
  -> build_benchmark_v1.py or create_benchmark_v2_from_large_chunks.py
  -> run_benchmark_v1.py / run_original_classifier_on_benchmark.py
  -> evaluate_* and analyze_* scripts

External binary benchmarks
  -> create_benchmark_v3a_political_ideologies.py
  -> create_benchmark_v3b_senator_tweets_binary.py
  -> create_benchmark_v3b_allsides.py
  -> run_binary_political_classifier*.py or train_encoder_benchmark_v3a_binary.py
  -> create_*_summary.py

Tulu/OLMo SFT mixture
  -> inspect_tulu_olmo2_sft*.py
  -> flatten_tulu_olmo2_sft.py
  -> prefilter_tulu_olmo2_political_candidates.py
  -> run_sft_political_4way_classifier.py
  -> manual review scripts
  -> train_encoder_target_domain_binary.py
  -> apply_target_domain_binary_modernbert.py
  -> create_sft_intervention_jsonl.py
  -> train_olmo2_lora_sft*.py
```

## Important Conventions

- Three-way ideology labels usually mean `left`, `center`, and `right`.
- Numeric three-way labels usually mean `1 = left`, `2 = center/neutral`,
  `3 = right`.
- Four-way SFT labels usually mean `0 = non_political`, `1 = left`,
  `2 = center/neutral`, `3 = right`.
- Binary labels usually mean `left` and `right`; in encoder scripts these are
  commonly mapped to `0 = left`, `1 = right`.
- Generated artifacts normally go under `data/processed/` or `outputs/`.
- Several scripts contain project-local hard-coded paths. If moving the repo,
  check `benchmark_utils.py`, `run_llama_manifesto_classifier.py`, and the
  Manifesto chunking scripts first.

## Core Shared Files

| File | Purpose |
| --- | --- |
| `benchmark_prompts.py` | Stores prompt templates for Manifesto benchmark v1. `build_prompt()` returns the text prompt for a selected prompt version. |
| `benchmark_utils.py` | Shared helpers for project path resolution, label normalization, safe model names, timestamps, JSON writing, and parsing model predictions. |

## Manifesto Benchmark Creation

These scripts turn Manifesto Project metadata and corpus text into chunk-level
classification benchmarks.

| File | What it does |
| --- | --- |
| `select_manifestos_for_large_download.py` | Samples Manifesto metadata into a balanced large-download list, using the Manifesto `rile` score to assign left/center/right labels. |
| `chunk_manifestos.py` | Splits a small sample of downloaded manifesto text into sentence-aware chunks. |
| `chunk_manifestos_strict.py` | Splits the same kind of manifesto text into stricter fixed-size word chunks. |
| `create_manifesto_large_chunks.py` | Builds overlapping chunks from the larger 300-document Manifesto corpus and writes `large_300_manifesto_chunks.csv` plus metadata. |
| `analyze_chunk_similarity.py` | Computes TF-IDF cosine similarity between chunks in each manifesto document to diagnose repeated or overly similar chunks. |
| `build_benchmark_v1.py` | Creates `data/processed/benchmark_v1/benchmark_v1.csv` from prepared chunks, normalizes labels, optionally balances classes, and can create train/dev/test splits. |
| `create_benchmark_v2_from_large_chunks.py` | Creates a larger balanced benchmark from `large_300_manifesto_chunks.csv`, selecting diverse chunks per document with a TF-IDF similarity filter. |

Useful examples:

```bash
python code/build_benchmark_v1.py \
  --input data/processed/manifesto/chunks/sample_30_manifesto_chunks_strict_balanced.csv \
  --output data/processed/benchmark_v1/benchmark_v1.csv \
  --prompt-version v2

python code/create_benchmark_v2_from_large_chunks.py
```

## Prompted LLM Benchmarking

These scripts run generative LLMs as political classifiers. Most use vLLM and
expect local or Hugging Face model identifiers.

| File | What it does |
| --- | --- |
| `run_benchmark_v1.py` | General benchmark v1 runner. Supports multiple models, vLLM or Transformers backends, resume mode, and error JSONs. |
| `run_llama_manifesto_classifier.py` | Earlier row-by-row Manifesto classifier with dry-run and single-prompt testing helpers. |
| `run_original_classifier_on_benchmark.py` | Runs the original three-way prompt on a given benchmark and saves parsed predictions. |
| `run_binary_political_classifier.py` | Runs a numeric left/right prompt on binary benchmarks. |
| `run_binary_political_classifier_v2.py` | Runs a stricter left/right label prompt and parser for binary benchmarks. |
| `run_sft_political_4way_classifier.py` | Classifies SFT conversations into non-political, left, neutral, or right. |
| `compare_original_vs_our_prompt.py` | Runs the original three-way prompt and the newer four-way prompt on the same labeled file, then prints metrics and saves both outputs. |
| `compare_original_prompt_three_models.py` | Runs the original three-way prompt across three configured models and saves per-model outputs. |
| `test_model_load.py` | Simple smoke test for loading a causal LM with Transformers and generating a short response. |

Useful examples:

```bash
python code/run_benchmark_v1.py \
  --benchmark data/processed/benchmark_v1/benchmark_v1.csv \
  --models cyankiwi/gemma-4-31B-it-AWQ-8bit \
  --backend vllm \
  --limit 20

python code/compare_original_vs_our_prompt.py \
  --model-path /path/to/model \
  --input outputs/tulu_olmo2_sft_political_classification_sample/manual_review_sample_200.csv \
  --sample-n 50
```

## Evaluation, Error Analysis, and Summaries

These scripts read prediction CSVs and produce metrics, markdown summaries,
comparison tables, or error samples.

| File | What it does |
| --- | --- |
| `evaluate_benchmark_v1.py` | Evaluates benchmark v1 prediction CSVs and writes summary/per-class metrics. |
| `evaluate_benchmark_v2_document_majority.py` | Aggregates chunk predictions to document-level majority votes for benchmark v2. |
| `analyze_benchmark_v2_errors.py` | Joins Llama/Gemma/Qwen benchmark v2 predictions and writes detailed agreement/error subsets. |
| `analyze_benchmark_v2_errors_light.py` | Lighter version of the benchmark v2 error analysis with summary tables and samples. |
| `create_final_benchmark_comparison.py` | Combines v1 and v2 original-classifier results into CSV and markdown comparison outputs. |
| `create_all_classifier_comparison.py` | Manually assembles a high-level comparison across prompted LLMs and encoder models. |
| `create_classifier_benchmark_summary_v2_v3.py` | Summarizes v2/v3 classifier results into CSV and markdown files. |
| `create_benchmark_v3a_summary.py` | Summarizes prompted binary-classifier outputs for benchmark v3a. |
| `create_benchmark_v3a_final_summary.py` | Combines prompted LLM and encoder results for benchmark v3a. |
| `reparse_binary_outputs.py` | Re-parses messy binary LLM outputs and writes `__reparsed.csv` files. |
| `reparse_sft_4way_outputs.py` | Re-parses four-way SFT classifier outputs and writes a `__reparsed.csv` file. |

## Binary Benchmark v3 Builders

These create additional binary benchmarks from public Hugging Face datasets.

| File | What it does |
| --- | --- |
| `create_benchmark_v3a_political_ideologies.py` | Builds a left/right benchmark from `JyotiNayak/political_ideologies`. |
| `create_benchmark_v3b_senator_tweets_binary.py` | Builds a Democrat/Republican senator-tweets benchmark mapped to left/right. |
| `create_benchmark_v3b_allsides.py` | Builds an AllSides-style left/center/right news benchmark from `valurank/PoliticalBias_AllSides_Txt`. |
| `try_lcr_dataset_candidates.py` | Inspects candidate Hugging Face datasets for possible left/center/right benchmark use. |
| `inspect_allsides_streaming.py` | Streams and prints sample rows from the AllSides dataset. |
| `inspect_newsmediabias.py` | Inspects a media-bias dataset candidate. |
| `inspect_news_political_bias.py` | Inspects a news-political-bias dataset candidate. |
| `inspect_political_datasets.py` | Tries a list of political datasets and prints basic structure/examples. |

## Encoder Classifiers

These train sequence-classification models such as ModernBERT, DeBERTa, or
other Hugging Face encoder models.

| File | What it does |
| --- | --- |
| `train_encoder_manifesto_classifier.py` | Trains a three-way left/center/right encoder on benchmark v2 with document-level train/validation/test splitting. Saves chunk and document-majority predictions. |
| `train_encoder_benchmark_v3a_binary.py` | Trains a binary left/right encoder on a provided v3-style benchmark split. |
| `train_encoder_target_domain_binary.py` | Trains `answerdotai/ModernBERT-base` on manually labeled Tulu/SFT political examples. |
| `check_modernbert_train_val_split.py` | Audits the manual binary training file and training script assumptions before ModernBERT training. |
| `apply_target_domain_binary_modernbert.py` | Applies the trained ModernBERT binary classifier to Tulu/OLMo political candidates. |

Example:

```bash
python code/train_encoder_manifesto_classifier.py \
  --benchmark data/processed/benchmark_v2/benchmark_v2.csv \
  --model answerdotai/ModernBERT-base \
  --output-dir outputs/encoder_manifesto_benchmark_v2/modernbert_base
```

## Tulu/OLMo SFT Filtering and Manual Review

These scripts inspect the Tulu/OLMo SFT mixture, find likely political examples,
create review files, and convert manual labels into training/evaluation data.

| File | What it does |
| --- | --- |
| `inspect_tulu_olmo2_sft.py` | Streams examples from `allenai/tulu-3-sft-olmo-2-mixture` for quick inspection. |
| `inspect_tulu_olmo2_sft_full.py` | Loads the full SFT mixture and writes source, dataset, role-pattern, and turn-count distributions. |
| `flatten_tulu_olmo2_sft.py` | Flattens SFT conversations into JSONL rows with combined text and writes a sample. |
| `prefilter_tulu_olmo2_political_candidates.py` | Uses keyword filtering to select likely political SFT examples and write candidate JSONL plus a sample CSV. |
| `create_high_confidence_political_candidates.py` | Applies stronger keyword/category rules to candidate rows and writes a high-confidence subset. |
| `create_tulu_sft_highconf_classification_sample.py` | Samples high-confidence candidates into CSV and JSONL files for classification/review. |
| `create_sft_4way_manual_review_sample.py` | Samples four-way classifier outputs into a 200-row manual-review CSV. |
| `export_manual_review_to_txt.py` | Exports a slice of the manual-review CSV to a readable text batch. |
| `import_manual_labels_from_tsv.py` | Merges external manual labels from TSV back into a CSV. |
| `evaluate_manual_review_sample.py` | Evaluates four-way classifier predictions against manual labels and writes error cases. |
| `create_binary_manual_eval.py` | Converts four-way manual labels/predictions into a binary left/right evaluation file. |
| `evaluate_binary_manual_review.py` | Evaluates the binary manual-review file. |
| `create_target_domain_binary_train.py` | Converts manually labeled review data into `target_domain_binary_manual_train.csv`. |

## SFT Intervention and LoRA Training

These scripts create SFT intervention datasets and train OLMo-2 LoRA variants.

| File | What it does |
| --- | --- |
| `create_sft_intervention_jsonl.py` | Builds full SFT JSONL variants: baseline, remove all political-position examples, balanced downsample-left, and balanced upsample-right. |
| `train_olmo2_lora_sft_smoke.py` | Small LoRA SFT smoke run on OLMo-2 using a 5k text-only JSONL file. |
| `train_olmo2_lora_sft_variant01_100k.py` | Larger 100k-example LoRA SFT run for variant 01. |

## Running Safely

- Start with `--limit`, `--dry-run`, or `--single-prompt-test` when a script
  supports it.
- Check input paths before launching a long GPU run; many scripts assume files
  already exist under `data/processed/` or `outputs/`.
- vLLM scripts may need careful `CUDA_VISIBLE_DEVICES`, tensor parallel, and GPU
  memory settings.
- Hugging Face dataset scripts require network access unless datasets are
  already cached.
- The files under `__pycache__/` are generated Python bytecode and are not part
  of the source workflow.

## Quick Orientation by Task

| If you want to... | Start with... |
| --- | --- |
| Build the first Manifesto benchmark | `build_benchmark_v1.py` |
| Run LLMs on benchmark v1 | `run_benchmark_v1.py` |
| Evaluate benchmark v1 predictions | `evaluate_benchmark_v1.py` |
| Build the larger Manifesto benchmark | `create_manifesto_large_chunks.py`, then `create_benchmark_v2_from_large_chunks.py` |
| Train a Manifesto encoder | `train_encoder_manifesto_classifier.py` |
| Create binary public benchmarks | `create_benchmark_v3a_political_ideologies.py`, `create_benchmark_v3b_senator_tweets_binary.py`, or `create_benchmark_v3b_allsides.py` |
| Classify SFT examples politically | `run_sft_political_4way_classifier.py` |
| Prepare manual labels | `export_manual_review_to_txt.py`, then `import_manual_labels_from_tsv.py` |
| Train the target-domain binary ModernBERT | `create_target_domain_binary_train.py`, then `train_encoder_target_domain_binary.py` |
| Build SFT interventions | `create_sft_intervention_jsonl.py` |
| Train OLMo-2 LoRA variants | `train_olmo2_lora_sft_smoke.py` or `train_olmo2_lora_sft_variant01_100k.py` |
