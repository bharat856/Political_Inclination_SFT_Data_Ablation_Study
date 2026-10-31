# Political Inclination: SFT Data Ablation Study

Research code for studying how the political composition of supervised fine-tuning (SFT) data relates to language-model political responses. The project combines political-content detection, ideological slant classification, controlled data ablations, OLMo fine-tuning, and model-behavior evaluation.

## Research workflow

```text
SFT conversations
    → political-candidate detection (ModernBERT ∪ MPNet)
    → left / neutral / right slant labels (Qwen + Gemma)
    → controlled removal lists
    → OLMo LoRA fine-tuning
    → ProbVAA political-response evaluation and IFEval tooling
```

Earlier experiments also develop classifiers using manifesto, political-ideology, news, and senator-tweet benchmarks. The final thesis training conditions use the **Instruct split only** of `allenai/Dolci-Instruct-SFT-7B`.

## Controlled training conditions

The existing project audit documents these dataset sizes before batching omissions:

| Condition | Intervention | Training examples |
| --- | --- | ---: |
| Full | Original Instruct training corpus | 2,152,112 |
| NoPol | Remove all political candidates | 1,743,530 |
| NoIdeology | Remove left- and right-labeled candidates | 2,134,313 |
| NoLeft | Remove left-labeled candidates | 2,137,253 |
| NoRight | Remove right-labeled candidates | 2,149,172 |
| LRBalanced | Downsample left examples to match right examples | 2,140,193 |

See [the verified intervention audit](CONTROLLED_SFT_DATA_INTERVENTIONS_README.md) for counting units, exact filters, provenance, and limitations. Earlier combined Instruct+Think counts describe a different pipeline stage.

## Repository guide

| Location | Contents |
| --- | --- |
| [`code/`](code/) | Dataset preparation, classifiers, intervention construction, and training scripts |
| [`code/evaluation/`](code/evaluation/) | ProbVAA and IFEval runners, analysis, audits, and tests |
| [`code/graphs/`](code/graphs/) | Plotting and figure scripts |
| [`scripts/`](scripts/) | Environment helpers, benchmark launchers, corpus downloads, presentation generation |
| [`apps/`](apps/) | Supporting project applications |
| [`presentations/`](presentations/) | Presentation materials |
| Root reports and prompts | Research history, method documentation, study guides, and prompt definitions |

## Start here

1. Read the [script study guide](THESIS_SCRIPT_STUDY_GUIDE.md) to trace the final thesis pipeline.
2. Consult the [code guide](code/README.md) for individual script purposes and benchmark examples.
3. Read the [fine-tuning audit](MODEL_FINE_TUNING_AUDIT_README.md) and [controlled intervention statistics](CONTROLLED_SFT_DATA_INTERVENTIONS_README.md) before interpreting training conditions.
4. Review the [ProbVAA documentation](code/evaluation/README_probvaa.md) and adjacent evaluation reports for prompt provenance and reconstruction constraints.
5. Consult [benchmark results](BENCHMARK_RESULTS_README.md) for earlier classifier experiments and [project context](PROJECT_CONTEXT.md) for detailed historical notes.

## Running the project

Run scripts from the repository root. This is a research workspace with several environment-specific entry points, rather than an installable Python package.

Requirements vary by stage: dataset and analysis utilities use packages such as pandas and Hugging Face Datasets; training and inference additionally use PyTorch, Transformers, PEFT, TRL, Accelerate, and/or vLLM. Corpus download scripts may require R. GPU workloads require compatible CUDA software and sufficient GPU memory.

Existing environment helpers in `scripts/` describe the original server setup. **Inspect paths and GPU assignments before using them.** Several scripts reference `/fs/scratch/users/bharatpi/causal_political_content` and expect prepared data or model artifacts. A fresh clone alone is insufficient to reproduce GPU experiments.

Example benchmark invocation, after preparing its inputs and environment:

```bash
python code/build_benchmark_v1.py \
  --input data/processed/manifesto/chunks/sample_30_manifesto_chunks_strict_balanced.csv \
  --output data/processed/benchmark_v1/benchmark_v1.csv \
  --prompt-version v2
```

## Data, artifacts, and interpretation

The original workspace contains more than 600 GB of datasets, caches, environments, model artifacts, and outputs. This source copy contains code, root documentation, prompts, figures, applications, scripts, and presentations. Datasets, environments, caches, checkpoints, generated outputs, logs, validation artifacts, and upstream reference checkouts are excluded. [COPY_MANIFEST.json](COPY_MANIFEST.json) records copied original files with SHA-256 hashes and the sizes and file counts of excluded directories. The original workspace remains intact. A complete data-and-checkpoint backup requires separate storage.

Original source files are copied without code changes. Historical documentation can contain server-specific links and describe superseded experiments; use the final-run audits to distinguish those stages. Political labels are operational classifier outputs, and condition comparisons should be interpreted alongside the documented training-exposure and seed limitations. Dataset sizes alone do not establish a model-behavior result.

## Upstream references

The project documentation identifies these research sources:

- [Political content in training data](https://github.com/tceron/political_content_training_data)
- [Political worldviews evaluation](https://github.com/tceron/eval_political_worldviews)

Refer to the evaluation documentation for inspected commits and asset provenance. Upstream code and datasets retain their respective licensing and attribution requirements.
