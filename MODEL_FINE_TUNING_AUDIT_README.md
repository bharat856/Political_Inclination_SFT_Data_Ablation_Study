# Final OLMo 3 Fine-Tuning Configuration: Read-Only Audit

Audit date: 2026-10-19  
Repository root: `/fs/scratch/users/bharatpi/causal_political_content`

This document records an evidence-based, read-only audit of the six final OLMo 3 supervised fine-tuning runs used for the experimental conditions Full, NoPol, NoIdeology, NoLeft, NoRight, and LRBalanced. No training or evaluation was launched, no classification was rerun, no model weights were loaded, and nothing was downloaded. Configuration files, logs, trainer states, adapter metadata, safetensors headers, and checksum manifests were inspected. Only this audit document was added after explicit authorization.

The dataset-construction audit and exact intervention counts are documented separately in [`CONTROLLED_SFT_DATA_INTERVENTIONS_README.md`](CONTROLLED_SFT_DATA_INTERVENTIONS_README.md).

## Scope: final runs only

| Condition | Final run directory |
|---|---|
| Full | `outputs/sft_runs/olmo3_instruct_full_4gpu_v4` |
| NoPol | `outputs/sft_runs/olmo3_instruct_nopol_4gpu_v5` |
| NoIdeology | `outputs/sft_runs/olmo3_instruct_noideology_4gpu_v5` |
| NoLeft | `outputs/sft_runs/olmo3_instruct_noleft_4gpu_v4` |
| NoRight | `outputs/sft_runs/olmo3_instruct_noright_4gpu_v5` |
| LRBalanced | `outputs/sft_runs/olmo3_instruct_lrbalanced_4gpu_v5` |

Similarly named earlier runs, smoke tests, post-reboot tests, and the NoRight `v4` run were not used as evidence for the final configurations.

## A. Base Model

- **Verified local checkpoint:** `/fs/scratch/users/bharatpi/causal_political_content/hf_cache/allenai_Olmo-3-1025-7B`.
- **Corresponding model name:** `allenai/Olmo-3-1025-7B`.
- **Consistency across conditions:** every final run's `run_config_v3.json` and `adapter_config.json` points to the same local base-model path.
- **Architecture:** `Olmo3ForCausalLM`, a decoder-only causal language model.
- **Exact parameter count:** 7,298,011,136 parameters, from the safetensors index metadata.
- **Checkpoint type:** the cached model README identifies this as the OLMo 3 7B Base checkpoint. It is a pretrained base model produced through staged pretraining, mid-training, and long-context training, not an instruction-tuned checkpoint.
- **Base-weight training status:** the base weights remained frozen. The scripts call `prepare_model_for_kbit_training`, then attach PEFT LoRA adapters; the saved adapters contain only LoRA tensors. The final `v5` logs independently report only the LoRA parameters as trainable.
- **Tokenizer:** loaded from the same base-model directory. The cached tokenizer identifies `GPT2Tokenizer`, with vocabulary size 100,278, maximum model length 65,536, EOS token `<|endoftext|>` (ID 100257), padding token `<|pad|>` (ID 100277), and no BOS token. Saved tokenizer configurations are identical across the six final run roots.
- **Model configuration relevant to training:** 32 transformer layers; hidden size 4,096; intermediate size 11,008; 32 attention heads; 32 key/value heads; SiLU activation; RMSNorm epsilon `1e-6`; zero attention dropout; untied input/output embeddings; RoPE theta 500,000; YaRN factor 8 with original maximum position count 8,192; configured maximum position count 65,536; sliding-window size 4,096; repeated pattern of three sliding-attention layers followed by one full-attention layer.
- **Training cache behavior:** the original model configuration has `use_cache=true`; the training scripts set `model.config.use_cache = False`, as required for checkpointed training.

The exact upstream Hugging Face repository commit or revision cannot be recovered from the final artifacts. Every saved `adapter_config.json` has `revision: null`; therefore, the local path and model identity are verified, but a remote commit hash is not.

Evidence:

- `hf_cache/allenai_Olmo-3-1025-7B/config.json`
- `hf_cache/allenai_Olmo-3-1025-7B/model.safetensors.index.json`
- `hf_cache/allenai_Olmo-3-1025-7B/tokenizer_config.json`
- `hf_cache/allenai_Olmo-3-1025-7B/README.md`
- each final run's `run_config_v3.json` and `adapter_config.json`
- `code/train_olmo3_lora_dolci_instruct_v4.py`
- `code/train_olmo3_lora_dolci_instruct_v5.py`

## B. Quantization

QLoRA was used in all six final runs.

| Quantization property | Verified value |
|---|---|
| Integration/library | Transformers `BitsAndBytesConfig` using bitsandbytes |
| Weight-loading precision | 4-bit (`load_in_4bit=True`) |
| Quantization type | NF4 |
| Double quantization | Enabled |
| Quantized computation dtype | BF16 |
| Model-loading dtype | BF16 (`dtype=torch.bfloat16`) |
| Device placement | One model replica on each process's `local_rank` CUDA device |
| Gradient checkpointing | Enabled |
| Gradient-checkpointing kwargs | `use_reentrant=False` |

The current environment contains bitsandbytes 0.49.2, but the exact bitsandbytes version present at training time was not saved in the final run metadata and is therefore unverified. The run-generated READMEs record PEFT 0.19.1, TRL 1.6.0, Transformers 5.10.2, PyTorch 2.11.0, Datasets 5.0.0, and Tokenizers 0.22.2. The current environment's Accelerate version is 1.13.0, but the historical training-time Accelerate version is not embedded in the artifacts.

Evidence:

- `code/train_olmo3_lora_dolci_instruct_v4.py`
- `code/train_olmo3_lora_dolci_instruct_v5.py`
- each final run's `run_config_v3.json`
- each final run's generated `README.md`
- the six final training logs listed in the Evidence Index

## C. LoRA Configuration

| LoRA property | Verified value |
|---|---|
| Rank `r` | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Bias | `none` |
| Task type | `CAUSAL_LM` |
| RSLoRA | Disabled |
| DoRA | Disabled |
| QALoRA | Disabled |
| aLoRA | Disabled |
| LoRA-GA | Not used |
| EVA | Not used |
| CorDA | Not used |
| Layer replication | Not used |
| Modules saved outside LoRA | None (`modules_to_save: null`) |

The safetensors headers of all six root adapters contain 448 tensors and exactly 39,976,960 BF16 LoRA parameters. This is approximately 0.547779% of the 7,298,011,136-parameter base model. No model tensors were loaded to obtain these counts; only safetensors metadata headers were inspected.

The `v5` logs explicitly report 39,976,960 BF16 trainable parameters on every rank. The two `v4` final runs do not contain the later hard dtype-check log message, but their adapter headers verify the same LoRA parameter count and BF16 saved dtype. In every run, the base-model weights were frozen and only adapter parameters were trainable.

`inference_mode: true` in the saved adapter configurations describes the adapter after it was saved for inference; it does not mean that the adapter was non-trainable during fine-tuning.

Evidence:

- each final run's `adapter_config.json`
- each final run's `adapter_model.safetensors` header
- `code/train_olmo3_lora_dolci_instruct_v4.py`
- `code/train_olmo3_lora_dolci_instruct_v5.py`
- final `v5` training logs

## D. Target Modules

The exact adapter target-module set is:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

The attention projections are `q_proj`, `k_proj`, `v_proj`, and `o_proj`. The MLP projections are `gate_proj`, `up_proj`, and `down_proj`. No LoRA adapters were attached to token embeddings, normalization layers, or the language-model head.

Per adapter, the safetensors-header parameter totals are:

| Module family | LoRA parameters |
|---|---:|
| `q_proj` | 4,194,304 |
| `k_proj` | 4,194,304 |
| `v_proj` | 4,194,304 |
| `o_proj` | 4,194,304 |
| `gate_proj` | 7,733,248 |
| `up_proj` | 7,733,248 |
| `down_proj` | 7,733,248 |
| **Total** | **39,976,960** |

Evidence: each final run's `adapter_config.json` and `adapter_model.safetensors` header.

## E. Sequence Processing

### Verified processing path

The intervention loader yields three fields per example: `row_id`, `text`, and `split`. Each example is therefore one flattened row and one actual SFT training example. It is not a batch of original conversation objects at trainer input time.

The flattening script converts each original `messages` list into a plain-text string. Every message is rendered as an uppercase role label, a colon, a newline, and its content; messages are separated by a blank line. For example:

```text
SYSTEM:
...

USER:
...

ASSISTANT:
...
```

The trainer consumes the `text` column through `dataset_text_field="text"`. It does not receive structured chat messages and does not apply a model chat template.

| Sequence property | Verified value |
|---|---|
| Input at trainer boundary | Plain-text iterable examples with `row_id`, `text`, `split` |
| Text construction | Flattened uppercase role markers plus message contents |
| Chat template | None applied by the training scripts |
| Maximum sequence length | 2,048 tokens |
| Truncation | Enabled; serialized mode `keep_start` |
| Padding side | Right |
| Batch padding | Dynamic to the longest sequence in the batch |
| Padding token | `<|pad|>`, ID 100277 |
| Packing | Disabled |
| Padding-free mode | Disabled |
| Completion-only loss | Not enabled (`completion_only_loss=None`, resolving false for this plain-text LM dataset) |
| Assistant-only loss | Disabled |
| Label smoothing | 0 |

Consequently, all non-padding causal-language-model tokens in the flattened string contribute to the loss, including system text, user text, assistant text, and role-label text. Padding positions are ignored with label value `-100`. This is full-sequence causal LM loss, not response-only or assistant-only loss.

The installed TRL 1.6.0 preprocessing path appends an EOS token to non-conversational text when one is absent before tokenization. That behavior is supported by the framework source used in the environment, but the final tokenized examples were not persisted. It is therefore not possible to verify the EOS result for every individual training example. In particular, `keep_start` truncation can remove a newly appended EOS token from examples longer than 2,048 tokens. No BOS token is configured. The serialized `packing_strategy='bfd'` value has no effect because packing is disabled.

Evidence:

- `code/flatten_olmo3_dolci_sft.py`
- `code/dolci_intervention_loader_instruct_v1.py`
- `code/train_olmo3_lora_dolci_instruct_v4.py`
- `code/train_olmo3_lora_dolci_instruct_v5.py`
- each final run's `run_config_v3.json`
- `hf_cache/allenai_Olmo-3-1025-7B/tokenizer_config.json`

## F. Batch Size and Gradient Accumulation

All six conditions used the same settings:

- Per-device training batch size: 1 example.
- Gradient accumulation: 16 microbatches.
- Distributed world size: 4 processes/GPUs.
- Effective global batch size: **64 examples per optimizer step**.

The calculation is:

```text
1 example/device/microbatch × 16 microbatches/optimizer step × 4 devices
= 64 examples/optimizer step
```

The final runs did not differ in per-device batch size, accumulation, world size, or effective global batch size.

Evidence: each final run's `run_config_v3.json`, final checkpoint `trainer_state.json`, and training log.

## G. Optimizer and Learning Rate

The common settings are shown in Table 1. The configured warmup was 3% of each run's maximum optimizer-step count, converted by the script to an integer number of steps. Because dataset sizes and maximum step counts differed, warmup steps also differed:

| Condition | Warmup steps |
|---|---:|
| Full | 1,008 |
| NoPol | 817 |
| NoIdeology | 1,000 |
| NoLeft | 1,001 |
| NoRight | 1,007 |
| LRBalanced | 1,003 |

The `v5` script adds a hard BF16 trainable-parameter check and BF16 restoration when resuming. Neither change creates a hyperparameter difference in the completed final runs: all final runs started without a resume checkpoint, and the adapter metadata shows a consistent BF16 LoRA dtype.

### TABLE 1: Common fine-tuning hyperparameters

| Parameter | Verified value |
|---|---|
| Base model | Local `allenai/Olmo-3-1025-7B` base checkpoint |
| Training method | 4-bit QLoRA |
| Quantization | NF4, double quantization enabled |
| Quantization compute dtype | BF16 |
| Model-loading dtype | BF16 |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| LoRA bias / task | `none` / `CAUSAL_LM` |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Trainable parameters | 39,976,960 LoRA parameters |
| Maximum sequence length | 2,048 tokens |
| Truncation / padding | `keep_start`; right-side dynamic padding |
| Packing | Disabled |
| Loss scope | Full non-padding token sequence; not assistant-only or completion-only |
| Per-device batch size | 1 |
| Gradient accumulation | 16 |
| Number of processes/GPUs | 4 |
| Effective global batch size | 64 examples per optimizer step |
| Optimizer | bitsandbytes `adamw_8bit` |
| Learning rate | `2e-5` |
| Scheduler | Linear |
| Warmup policy | `int(max_steps × 0.03)`; exact count differs by condition |
| Weight decay | 0.0 |
| Gradient clipping | Maximum norm 1.0 |
| Training precision | BF16 enabled; FP16 disabled |
| Gradient checkpointing | Enabled, `use_reentrant=False` |
| Random seed / data seed | 42 / 42 |
| Streaming shuffle buffer | 100,000 examples |
| Drop incomplete dataloader batch | Enabled |
| Logging interval | 10 optimizer steps |
| Checkpoint interval | 1,000 optimizer steps |
| Checkpoint retention | 2 checkpoints |
| Evaluation during training | None |
| Reporting integration | None |
| `ddp_find_unused_parameters` | False |

## H. Distributed Training

- **Verified process count:** four ranks, numbered 0 through 3, with world size 4 and local CUDA devices 0 through 3.
- **Launch mechanism:** logs begin in `torch/distributed/run.py`, demonstrating a PyTorch distributed/`torchrun` launch. The exact shell command was not retained.
- **Training strategy:** the evidence supports standard Transformers/Accelerate DistributedDataParallel operation. Each process loads a model replica on its local device. No final configuration enables FSDP or DeepSpeed.
- **Dataset sharding:** the trainer/Accelerate stack shards the iterable dataset once. The scripts prevent an additional manual split from being applied, despite a potentially confusing loader message indicating that distributed values are configured.
- **Gradient synchronization:** accumulation and synchronization are managed by Trainer/Accelerate; gradients synchronize at optimizer-step boundaries under DDP. This synchronization detail is framework behavior consistent with the verified configuration, not a separately persisted per-step trace.
- **GPU model:** not recorded in the retained logs or final configuration and therefore unverified. It is not inferred from the `4gpu` directory names.

Evidence:

- six final training logs listed in the Evidence Index
- each final run's `run_config_v3.json`
- `code/train_olmo3_lora_dolci_instruct_v4.py`
- `code/train_olmo3_lora_dolci_instruct_v5.py`

## I. Final Training Run Comparison

All six conditions were configured for approximately one pass through their respective iterable training datasets, using the same 64-example effective global batch. Since `dataloader_drop_last=true` and the dataset sizes were not exact multiples of 64, a small remainder was omitted from each run.

“Final loss” below is the aggregate `train_loss` in the terminal training summary, not the last ten-step logging-window loss. “Epochs” is the logged terminal epoch metric. The trainer configuration contains a large sentinel value for `num_train_epochs` because an iterable dataset and explicit `max_steps` were used; `global_step`, `max_steps`, and terminal `epoch=1.0` are the reliable completion measures.

### TABLE 2: Final experimental training runs

| Condition | Dataset examples | Optimizer steps | Epochs | Training duration | Final loss | Status |
|---|---:|---:|---:|---:|---:|---|
| Full | 2,152,112 | 33,624 | 1.0 | 135:21:44 | 0.8465 | Completed; final checkpoint present and verified |
| NoPol | 1,743,530 | 27,240 | 1.0 | 110:37:09 | 0.8164 | Completed; final checkpoint present and verified |
| NoIdeology | 2,134,313 | 33,346 | 1.0 | 134:07:12 | 0.8413 | Completed; final checkpoint present and verified |
| NoLeft | 2,137,253 | 33,392 | 1.0 | 133:18:12 | 0.8426 | Completed; final checkpoint present and verified |
| NoRight | 2,149,172 | 33,578 | 1.0 | 135:00:08 | 0.8453 | Completed; final checkpoint present and verified |
| LRBalanced | 2,140,193 | 33,438 | 1.0 | 134:20:19 | 0.8433 | Completed; final checkpoint present and verified |

### Exposure and final checkpoint details

| Condition | Configured maximum steps | Examples consumed (`steps × 64`) | Remainder omitted | Approx. logged tokens | Final checkpoint |
|---|---:|---:|---:|---:|---|
| Full | 33,624 | 2,151,936 | 176 | `3.124e8` | `outputs/sft_runs/olmo3_instruct_full_4gpu_v4/checkpoint-33624` |
| NoPol | 27,240 | 1,743,360 | 170 | 259,961,474 | `outputs/sft_runs/olmo3_instruct_nopol_4gpu_v5/checkpoint-27240` |
| NoIdeology | 33,346 | 2,134,144 | 169 | `3.097e8` | `outputs/sft_runs/olmo3_instruct_noideology_4gpu_v5/checkpoint-33346` |
| NoLeft | 33,392 | 2,137,088 | 165 | `3.102e8` | `outputs/sft_runs/olmo3_instruct_noleft_4gpu_v4/checkpoint-33392` |
| NoRight | 33,578 | 2,148,992 | 180 | `3.113e8` | `outputs/sft_runs/olmo3_instruct_noright_4gpu_v5/checkpoint-33578` |
| LRBalanced | 33,438 | 2,140,032 | 161 | `3.105e8` | `outputs/sft_runs/olmo3_instruct_lrbalanced_4gpu_v5/checkpoint-33438` |

The duration values correspond to the final progress elapsed times; the logs also record approximately 487,300, 398,200, 482,800, 479,900, 486,000, and 483,600 seconds, respectively.

### Were training exposures equivalent?

They were equivalent in training policy, not in absolute exposure. Each condition used the same batch configuration and approximately one pass through its own dataset. The interventions produced datasets of different sizes, so the runs necessarily completed different numbers of optimizer steps, processed different numbers of examples and tokens, and ran for different durations. NoPol had the largest reduction and therefore substantially less absolute training exposure. There is no evidence that smaller datasets were upsampled to match the Full condition's step count.

The approximate token counts above are log-derived. Except for NoPol's exactly recoverable logged value, they are retained in scientific notation at the precision recorded by the logs and should not be presented as exact token counts.

Evidence: each final run's `run_config_v3.json`, final checkpoint `trainer_state.json`, and corresponding final training log.

## J. Training Verification

### Artifact and completion checks

For every final condition, the root run directory contains:

- `adapter_model.safetensors` (approximately 80,014,016 bytes),
- `adapter_config.json`,
- tokenizer files,
- `training_args.bin`,
- `run_config_v3.json`, and
- a generated `README.md`.

Each final numbered checkpoint contains adapter files, optimizer state (approximately 81.7 MB), scheduler state, four per-rank RNG-state files, `trainer_state.json`, training arguments, and tokenizer files. A retained preceding checkpoint is also present for each run: step 33,000 for Full, NoIdeology, NoLeft, NoRight, and LRBalanced, and step 27,000 for NoPol.

The final trainer states satisfy `global_step == max_steps`, record terminal `epoch=1.0`, and agree with the final checkpoint directory names. All six logs show all four ranks returning from `trainer.train()`, followed by the root save and 100% progress. These facts provide positive evidence that the runs completed rather than merely having output directories.

The numeric contents of all six final trainer states were checked for non-finite values; none were found. Logged loss fields were also checked for `NaN` or `Inf`; none were found.

Each run's `checksums/SHA256SUMS.txt` was checked against all present listed files with `sha256sum -c --ignore-missing`. All six checks completed successfully with no failures. The manifests contain 33 or 34 entries, depending on the run. This establishes consistency with the saved local manifests, but it is not an upstream authenticity check for the base model.

No run records a best metric or best checkpoint because evaluation during training was disabled. This absence is expected and is not evidence of failure.

### Cross-run consistency

Verified consistent across all six final conditions:

- local base-model path,
- tokenizer artifacts,
- QLoRA quantization settings,
- LoRA rank, alpha, dropout, bias, task type, target modules, and trainable-parameter count,
- maximum sequence length and loss scope,
- per-device batch size, gradient accumulation, and distributed world size,
- optimizer, learning rate, scheduler family, precision, clipping, weight decay, seeds, and checkpoint policy,
- complete final adapter and trainer-state artifacts.

Expected run-specific differences are dataset identity and size, maximum/warmup steps, consumed examples and tokens, duration, and aggregate final training loss.

## K. Unverified Details and Reproducibility Limitations

The following claims cannot be made from the retained evidence:

1. **Exact upstream base-model revision:** the final adapters have `revision: null`, and no remote commit hash is saved.
2. **Historical bitsandbytes and Accelerate versions:** current environment versions do not prove the exact versions used during training.
3. **Exact launch command:** `torchrun`/PyTorch distributed launch is visible in the logs, but the initiating shell command is absent.
4. **GPU hardware model:** not recorded in the retained evidence.
5. **Per-example post-tokenization sequences:** tokenized training records were not persisted, so EOS placement and truncation outcomes cannot be audited example by example.
6. **Exact total token exposure for five runs:** logs provide rounded scientific-notation totals for all except NoPol.
7. **Independent upstream weight authenticity:** run checksums validate retained run files against local manifests, not the cached base weights against a remote repository revision.
8. **Evaluation-based model selection:** no evaluation was run during training, and no best checkpoint or best metric exists.

These limitations do not negate the verified configuration or completion evidence, but they should be disclosed in a reproducibility statement.

## Suggested Thesis Chapter Consolidation

To avoid unnecessary fragmentation, the proposed fine-tuning chapter can be organized as follows:

1. Keep **Base Model** as a short standalone section.
2. Merge **Quantization**, **LoRA Configuration**, and **Target Modules** into **Parameter-Efficient Fine-Tuning Configuration**.
3. Keep **Sequence Processing and Loss Construction** standalone because it determines the learning objective and clarifies that the loss was not assistant-only.
4. Merge **Batch Size and Gradient Accumulation**, **Optimizer and Learning Rate**, and **Distributed Training** into **Optimization and Distributed Training**.
5. Merge **Training Each Experimental Condition** and **Training Verification** into **Experimental Runs and Completion Verification**, centered on Table 2.
6. Place checksums, exact paths, software-version qualifications, and other unverified details in **Implementation and Reproducibility** or an appendix.

## Evidence Index

### Base model and tokenizer

- `hf_cache/allenai_Olmo-3-1025-7B/config.json`
- `hf_cache/allenai_Olmo-3-1025-7B/model.safetensors.index.json`
- `hf_cache/allenai_Olmo-3-1025-7B/tokenizer_config.json`
- `hf_cache/allenai_Olmo-3-1025-7B/README.md`

### Data and training code

- `code/flatten_olmo3_dolci_sft.py`
- `code/dolci_intervention_loader_instruct_v1.py`
- `code/train_olmo3_lora_dolci_instruct_v4.py`
- `code/train_olmo3_lora_dolci_instruct_v5.py`

### Final training logs

- `logs/olmo3_instruct_full_4gpu_v4.log`
- `logs/olmo3_instruct_nopol_4gpu_v5.log`
- `logs/olmo3_instruct_noideology_4gpu_v5.log`
- `logs/olmo3_instruct_noleft_4gpu_v4.log`
- `logs/olmo3_instruct_noright_4gpu_v5.log`
- `logs/olmo3_instruct_lrbalanced_4gpu_v5.log`

### Per-run evidence

Within each final run directory listed at the start of this report, the principal supporting files are:

- `run_config_v3.json`
- `adapter_config.json`
- `adapter_model.safetensors` (header inspected only)
- `README.md`
- `checksums/SHA256SUMS.txt`
- the final checkpoint's `trainer_state.json`
- the final checkpoint's adapter, optimizer, scheduler, RNG, tokenizer, and training-argument artifacts

## Concise Thesis-Ready Summary

All six final models were obtained by 4-bit NF4 QLoRA fine-tuning of the same local `allenai/Olmo-3-1025-7B` base checkpoint. The frozen 7,298,011,136-parameter base model was augmented with 39,976,960 trainable BF16 LoRA parameters (`r=16`, `alpha=32`, dropout 0.05) applied to all attention projection and MLP projection modules. Training used plain-text flattened conversations, a 2,048-token maximum length, full-sequence causal-language-model loss, per-device batch size 1, 16 gradient-accumulation steps, and four distributed processes, giving an effective global batch size of 64. Each condition received approximately one pass over its own intervention dataset. Thus, optimization policy was held constant, but absolute training exposure varied with dataset size. The final checkpoint artifacts, trainer states, logs, and local checksum manifests support successful completion of all six runs, with no recorded non-finite losses.
