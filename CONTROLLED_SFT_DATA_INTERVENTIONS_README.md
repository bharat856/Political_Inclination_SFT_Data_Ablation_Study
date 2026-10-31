# Controlled SFT Data Interventions: Verified Dataset Statistics

Audit date: 2026-10-19

## Scope and authoritative dataset

The six completed thesis conditions (`Full`, `NoPol`, `NoIdeology`, `NoLeft`,
`NoRight`, and `LRBalanced`) were trained from the **Instruct split only** of
`allenai/Dolci-Instruct-SFT-7B`. The authoritative Full dataset contains
**2,152,112 examples**.

This distinction is important. Several older artifacts in
`data/interventions/dolci_olmo3_sft/` and `logs/verify_all_intervention_conditions.log`
describe earlier interventions over the combined Instruct+Think corpus
(4,420,290 rows). Those counts are not the dataset sizes used by the final
`olmo3_instruct_*` thesis runs. The final training loader instead reads the
MPNet-TANISE row lists under
`data/interventions/dolci_olmo3_sft_union_mpnet_t04/row_lists/`, and the
completed run configurations independently record the Instruct-only sizes.

## Thesis-ready statistics

All counts below are verified dataset-example counts before the small
end-of-epoch omission required by global-batch divisibility. “Other political”
means a political-candidate example assigned the mutually exclusive final
label `neutral`. No Instruct candidate was unresolved.

| Condition | Total training examples | Political | Non-political | Left | Right | Other political (`neutral`) | Removed vs. Full | Removed vs. Full (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | 2,152,112 | 408,582 | 1,743,530 | 14,859 | 2,940 | 390,783 | 0 | 0.000000% |
| NoPol | 1,743,530 | 0 | 1,743,530 | 0 | 0 | 0 | 408,582 | 18.985164% |
| NoIdeology | 2,134,313 | 390,783 | 1,743,530 | 0 | 0 | 390,783 | 17,799 | 0.827048% |
| NoLeft | 2,137,253 | 393,723 | 1,743,530 | 0 | 2,940 | 390,783 | 14,859 | 0.690438% |
| NoRight | 2,149,172 | 405,642 | 1,743,530 | 14,859 | 0 | 390,783 | 2,940 | 0.136610% |
| LRBalanced | 2,140,193 | 396,663 | 1,743,530 | 2,940 | 2,940 | 390,783 | 11,919 | 0.553828% |

Percentages use

\[
100 \times \frac{N_{\mathrm{Full}} - N_{\mathrm{condition}}}
{N_{\mathrm{Full}}}, \qquad N_{\mathrm{Full}}=2{,}152{,}112.
\]

The component counts reconcile exactly in every row:
`political = left + right + other political`, and
`total = political + non-political`.

## Exact condition definitions

The political gate is the union of:

1. the earlier ModernBERT binary political candidate flag; and
2. the MPNet embedding candidate flag at threshold 0.4.

For Instruct, this union contains 408,582 rows: 78,539 selected by both gates,
276,836 by MPNet only, and 53,207 by ModernBERT only. The other 1,743,530 rows
are treated as non-political by the intervention pipeline.

The 408,582 candidates were assigned one final TANISE slant label from
`left`, `neutral`, or `right` by combining the Qwen and Gemma predictions:
agreement is retained; a slanted prediction wins over a neutral prediction;
and Gemma breaks a direct left-versus-right disagreement. Missing predictions
fall back to the available valid prediction. There are no unresolved Instruct
rows.

The exact filters used for training were:

| Condition | Exact rule applied to Full |
|---|---|
| Full | Remove nothing. |
| NoPol | Remove every row selected by the ModernBERT-or-MPNet political union (408,582 rows). |
| NoIdeology | Remove every final `left` or `right` row (14,859 + 2,940 = 17,799 rows); retain `neutral` political rows. |
| NoLeft | Remove every final `left` row (14,859 rows). |
| NoRight | Remove every final `right` row (2,940 rows). |
| LRBalanced | Downsample the majority `left` class by removing 11,919 designated left rows, so 2,940 left and 2,940 right rows remain; retain all neutral and non-political rows. |

At load time, these rules are implemented literally as membership exclusion:
a row is yielded if and only if its `row_id` is absent from the condition's
removal set.

## Counting unit

The source cache metadata records 2,152,112 original training rows. The
flattening script iterates those rows once, concatenates all messages in each
row into one `text` field, and writes one JSONL object per non-empty source
row. The resulting JSONL also has exactly 2,152,112 lines, with row IDs from
0 through 2,152,111. Therefore, for this dataset:

- one original dataset row/conversation = one flattened JSONL row;
- one flattened row = one SFT dataset example supplied to the trainer;
- messages or turns were **not** exploded into separate examples; and
- training used `dataset_text_field="text"` with `packing=False`.

Thus, the table reports actual SFT **dataset examples**, not message/turn
counts. It does not claim that every listed example produced an optimizer
update: the full-pass schedule and `dataloader_drop_last=True` omitted a small
tail to satisfy the global effective batch of 64. The completed run records
show the following scheduled/consumed counts: Full 2,151,936 (176 omitted),
NoIdeology 2,134,144 (169 omitted), NoLeft 2,137,088 (165 omitted), NoRight
2,148,992 (180 omitted), LRBalanced 2,140,032 (161 omitted), and NoPol
1,743,360 (170 omitted, computed exactly as 27,240 steps x 64 and corroborated
by the completed training log). These batching omissions are not part of the
intervention and therefore are not subtracted in the main dataset table.

## Duplication, upsampling, common source, and label overlap

- **No intervention duplication or upsampling:** verified. The loaders stream
  the same base JSONL once and only exclude IDs. They never append or repeat
  examples. All final intervention CSVs have unique `(split, row_id)` keys,
  and the final classification table has 693,639 unique keys in 693,639 rows.
- **LRBalanced construction:** verified as downsampling the majority left
  class. Its Instruct removal file contains 11,919 unique IDs, all of which are
  left-labeled, and it is a subset of the NoLeft removal set. No right,
  neutral, or non-political rows are added or removed for balancing.
- **Same original dataset:** verified. Every final condition reads the same
  `dolci_instruct_sft_7b_flattened.jsonl`; only the removal-ID set changes.
- **Label overlap:** the final labels `left`, `neutral`, and `right` are
  mutually exclusive within the political candidate set. “Political” is a
  superclass equal to their union, so it intentionally overlaps each of those
  three reporting categories. “Non-political” is the complement of that union
  and does not overlap it. The condition removal sets are nested where
  expected: NoLeft and NoRight are disjoint subsets of NoIdeology, which is a
  subset of NoPol; LRBalanced is a subset of NoLeft.
- **Duplicates already present upstream:** unavailable from the intervention
  metadata. The audit verifies that construction introduced no duplicate row
  keys and performed no upsampling, but it does not establish that the
  upstream DOLCI collection contains no semantically or textually duplicate
  conversations.
- **Sampling provenance for the Instruct-specific LRBalanced ID choice:** the
  retained artifact proves the exact selected IDs and the resulting balance,
  but no script or metadata file in the repository records the random seed or
  sampling implementation used to create
  `remove_lr_balanced_instruct_row_ids.csv`. Those details are therefore
  unavailable and should not be inferred from the older combined-corpus
  builder's seed.

## Supporting files

Authoritative evidence for the final thesis conditions:

- Base flattened training data:
  `data/processed/dolci_olmo3_sft/dolci_instruct_sft_7b_flattened.jsonl`
- Source dataset metadata and original-row count:
  `cache/huggingface/datasets/allenai___dolci-instruct-sft-7_b/default/0.0.0/bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221/dataset_info.json`
- Flattening procedure:
  `code/flatten_olmo3_dolci_sft.py`
- Political-union summary:
  `outputs/dolci_olmo3_sft_union_mpnet_t04/instruct/dolci_instruct_sft_7b_embedding_union_gate_summary.json`
- Final Qwen/Gemma/TANISE labels:
  `data/interventions/dolci_olmo3_sft_union_mpnet_t04/final_tanise_ensemble_labels.csv`
- Final-label rule and combined-corpus summary:
  `code/build_final_mpnet_tanise_interventions.py` and
  `data/interventions/dolci_olmo3_sft_union_mpnet_t04/final_tanise_ensemble_summary.json`
- Exact removal lists:
  `data/interventions/dolci_olmo3_sft_union_mpnet_t04/row_lists/remove_no_pol_row_ids.csv`,
  `remove_no_ideology_row_ids.csv`, `remove_no_left_row_ids.csv`,
  `remove_no_right_row_ids.csv`, and `remove_lr_balanced_instruct_row_ids.csv`
- Final Instruct-only loader and filtering implementation:
  `code/dolci_intervention_loader_instruct_v1.py`
- Trainer's one-row/one-example configuration:
  `code/train_olmo3_lora_dolci_instruct_v4.py` and
  `code/train_olmo3_lora_dolci_instruct_v5.py`
- Completed run configurations:
  `outputs/sft_runs/olmo3_instruct_full_4gpu_v4/run_config_v3.json`,
  `outputs/sft_runs/olmo3_instruct_noleft_4gpu_v4/run_config_v3.json`,
  `outputs/sft_runs/olmo3_instruct_noright_4gpu_v5/run_config_v3.json`,
  `outputs/sft_runs/olmo3_instruct_lrbalanced_4gpu_v5/run_config_v3.json`,
  `outputs/sft_runs/olmo3_instruct_noideology_4gpu_v5/run_config_v3.json`, and
  `outputs/sft_runs/olmo3_instruct_nopol_4gpu_v5/run_config_v3.json`
- Completed training logs, which independently report the raw, removed, and
  yielded row counts:
  `logs/olmo3_instruct_full_4gpu_v4.log`,
  `logs/olmo3_instruct_noleft_4gpu_v4.log`,
  `logs/olmo3_instruct_noright_4gpu_v5.log`,
  `logs/olmo3_instruct_lrbalanced_4gpu_v5.log`,
  `logs/olmo3_instruct_noideology_4gpu_v5.log`, and
  `logs/olmo3_instruct_nopol_4gpu_v5.log`

## Reproducibility note about superseded counts

Do not cite the 4,420,290-row combined Instruct+Think statistics from
`data/interventions/dolci_olmo3_sft/final_intervention_report.json`, nor the
intermediate union counts in `logs/verify_all_intervention_conditions.log`, as
the sizes of the final six thesis training datasets. They document prior
pipeline stages and are useful provenance, but they do not match the
Instruct-only loader and completed `olmo3_instruct_*` runs audited here.
