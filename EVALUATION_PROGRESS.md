# Evaluation progress and recovery handoff

IFEval recovery and official scoring are COMPLETE for all four adapters (2026-10-11). Final validation verified all 550 original responses unchanged, exactly 1,614 newly generated responses, and 541 unique benchmark prompt IDs per model (2,164 total). Adapter weights/configuration hashes and matched prompt/generation configurations passed verification.


| Model | Strict prompt | Loose prompt | Strict instruction | Loose instruction |
|---|---:|---:|---:|---:|
| Full | 56.56% | 60.63% | 66.79% | 70.02% |
| NoLeft | 57.49% | 62.11% | 66.67% | 70.50% |
| NoRight | 58.04% | 62.85% | 67.15% | 71.58% |
| LRBalanced | 54.71% | 58.41% | 64.39% | 67.99% |

All models use the same pinned official benchmark, prompt format, generation settings and seed. Exact counts, versions and file hashes are in comparison.json. These scores do not by themselves establish causal significance or broad capability.

Interrupted original segments lack trustworthy final runtime and CUDA peak measurements. Full-run totals are unavailable; resumed segment measurements are preserved separately. Political pilot scores remain diagnostic and are not final thesis results.

Official scorer caveat: keys 1122 and 1129 request punctuation frequencies ("#" and "!"), but the pinned letter-frequency checker accepts only ASCII letters and substitutes seeded random letters. Strict and loose passes can choose different fallback letters, so an official strict pass can become a loose failure. These are 2 of 834 instructions. Scores here preserve the official outputs without correction. Per-model anomalies are retained in summary.json and per_example.jsonl.

## Artifacts and recovery commands

- Thesis comparison: `outputs/evaluation_recovery_20261011/final_comparison/report.md`, `comparison.csv`, `comparison.json`, and `provenance.json`.
- Final integrity audit: `outputs/evaluation_recovery_20261011/final_output_validation.json`; SHA256 manifest: `artifact_manifest.json` in that directory.
- Exact executed tmux launch and CPU recovery commands: `outputs/evaluation_recovery_20261011/executed_commands.md` and `executed_commands.json`. All GPU recovery workers finished naturally; no generation worker was restarted or duplicated.
- Original responses: `outputs/evaluation_stage_20261011/{Model}/ifeval/` remain unchanged.
- New suffixes and per-prompt checkpoints: `outputs/evaluation_recovery_20261011/{Model}/resumed/`.
- Complete merged raw generations, scorer inputs and metadata: `{Model}/complete/`; official outputs: `complete/scoring/`; all per-example results and scores: `{Model}/analysis/`. Separate logs and historical recovery events remain per model.

Official source commit: `e6890f85757dd84e27ca6df2dd30651dafad28e0`. Dataset SHA256: `67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49`. Configuration preserved from the smoke test: USER/ASSISTANT training wrapper, greedy generation, max_new_tokens=1280, seed 42, BF16 base with PEFT runtime FP32 adapter tensors. Exact effective configuration/package versions are retained in metadata. Official source, data and scorer were unchanged. Analysis-only strict/loose monotonicity assumptions were removed to preserve official punctuation-fallback outcomes; historical failed analyses and CPU recovery commands are preserved.

## ProbVAA and CPU preparation

All four candidate political pilots are complete: 288 generations/model, all 12 intended template variants included, matched prompts and distinct reproducible sample seeds. Parsing success: Full 91.67%, NoLeft 88.19%, NoRight 90.97%, LRBalanced 92.01%. Pilot scores are diagnostic, not thesis political results. Report: `outputs/evaluation_stage_20261011/pilot_report.md`.

CPU-only preparation is complete under `outputs/probvaa_preparation_20261011/`: 17,208 unique prompts (239 statements × 6 formulations × 12 variants), per-model pilot scoring audits, metadata and README. Twenty-seven CPU tests passed. Recovered templates/wrapper and unique variant/seed choices are reconstructed protocol choices; reference parsing and scoring formulas remain unchanged. The old reference runner does not implement the candidate configuration. Full-run repeated sample count remains undecided. Remaining methodological issues include first-label ambiguity, residual truncation/NA handling, reference-template reconstruction and sample-count selection.

No full ProbVAA evaluation was launched. No trained adapters were modified. No further GPU evaluation is required for the completed IFEval benchmark. If revisiting historical running states, inspect actual processes rather than trusting saved status JSON.

## 2026-10-12 final CPU ProbVAA preparation

Original27tests and5newtests pass. Corrected pilots revalidated: all12variants,64-token configuration,paired distinct seeds,reference NA denominator. Resumable candidate runner and unexecuted tmux commands prepared; noGPUjob launched. Full launch awaits repeat count, reconstructed-protocol acceptance and residual truncation/parser decision. Read outputs/probvaa_final_preparation_20261012/report.md and candidate_s2_commands.sh. Paired IFEval bootstrap results: outputs/probvaa_final_preparation_20261012/ifeval_paired/report.md. Existing scientific outputs and adapters unchanged.

## 2026-10-12 original-paper audit — NO-GO supersedes earlier readiness

Latest instruction requires paper/appendix comparability and approval before any additional GPU inference. Full evaluation remains unlaunched. Read `outputs/probvaa_paper_audit_20261012/REPORT.md`, `protocol_comparison.csv`, `coverage_summary.json`, and `SFT_F_baseline_validation_plan.json` before proceeding.

Verified original2024 paper and official repository/HF/OSF archive; also version-pinned later training-data paper v1/v3. All239 statements/1434 English formulations match release, but current SFT candidates select base/generation templates, not Table7 instruction templates. Thirty repetitions are specified by both papers; two is an adapted stage. Original2024 drops invalid responses before reliability bootstrap; later notebook/current primary retains NA in denominator; later-v1 paper states parsed-only and magnitude-weighted policies. Do not conflate these scoring targets. Original release confirms a duplicated reversed instruction template (variants2/14), giving11 distinct prompt strings despite12 nominal slots. This needs an explicit paper-intent versus released-error decision.

Machine-readable full statement/prompt/appendix coverage and separate generated/parsed/NA/ambiguous/missing accounting saved. Paper-target SFT-F responses0/516240; candidate corrected pilot288responses with264parsed/24NA. Original parser matches1161test responses. All35 CPUtests and complete prompt tokenization passed.

Earlier authorized smoke is CHECKPOINTED at3/8responses in `outputs/probvaa_method_validation_20261012/Full_smoke`; it exited naturally and has no active worker. Do not resume without new GPU approval. Resume/end-to-end smoke remains untested. Supplementary reporting implemented in `code/evaluation/probvaa_supplementary.py`; outputs stay separate from primary. GPU4–7idle and396GiBfree at audit. Trained adapters and prior evaluation outputs unchanged.

Next: freeze original2024 vs later-paper primary target; declare repetition/prompt/order/decoding/format/scoring corrections; CPU-validate named protocol; then request approval for the prepared representative SFT-F GPU test. Do NOT execute prior candidate full-launch commands. Approx30-repeat runtime with unchanged64tokenbehavior is434–517GPUh/model;20token original-code scenario≈180–186h/model is uncertain extrapolation requiring measured smoke throughput.
