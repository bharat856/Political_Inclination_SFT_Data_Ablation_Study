from pathlib import Path
import argparse
import json
import os
import warnings

import torch
import torch.distributed as dist
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

# Reuse the v2 loader's filtering/shuffle logic, but disable its
# manual split_dataset_by_node. Accelerate must be the only sharder.
import dolci_intervention_loader_instruct_v1 as dolci_loader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "hf_cache/allenai_Olmo-3-1025-7B"

VALID_CONDITIONS = dolci_loader.VALID_CONDITIONS
expected_row_count = dolci_loader.expected_row_count


warnings.filterwarnings(
    "ignore",
    message=r".*_check_is_size will be removed.*",
    category=FutureWarning,
)


class LocalMetricsSFTTrainer(SFTTrainer):
    """
    SFTTrainer with:
      1. local-only auxiliary metric gathering;
      2. resume-safe BF16 LoRA restoration;
      3. hard dtype validation before optimization.
    """

    _dtype_checked = False

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        original = self.accelerator.gather_for_metrics
        self.accelerator.gather_for_metrics = (
            lambda value, *args, **kwargs: value
        )

        try:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        finally:
            self.accelerator.gather_for_metrics = original

    def _load_from_checkpoint(
        self,
        resume_from_checkpoint,
        model=None,
    ):
        # Let Transformers / PEFT restore the checkpoint normally.
        super()._load_from_checkpoint(
            resume_from_checkpoint,
            model=model,
        )

        target_model = model if model is not None else self.model

        # PEFT checkpoint restoration in the current stack promotes
        # the trainable LoRA parameters to FP32. Restore the intended
        # BF16 adapter dtype before resumed optimization.
        converted = 0

        for name, param in target_model.named_parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.bfloat16)
                converted += param.numel()

        print(
            f"[RESUME DTYPE FIX] "
            f"rank={self.accelerator.process_index} "
            f"converted_trainable_params={converted} "
            f"target_dtype=torch.bfloat16",
            flush=True,
        )

    def training_step(self, model, inputs, *args, **kwargs):
        if not self._dtype_checked:
            from collections import Counter

            trainable = Counter()
            lora = Counter()

            for name, param in model.named_parameters():
                if param.requires_grad:
                    trainable[str(param.dtype)] += param.numel()

                    if "lora_" in name.lower():
                        lora[str(param.dtype)] += param.numel()

            rank = self.accelerator.process_index

            print(
                f"[DTYPE CHECK] "
                f"rank={rank} "
                f"global_step={self.state.global_step} "
                f"trainable={dict(trainable)} "
                f"lora={dict(lora)}",
                flush=True,
            )

            expected = 39_976_960

            actual_bf16 = trainable.get(
                "torch.bfloat16",
                0,
            )

            total_trainable = sum(
                trainable.values()
            )

            if (
                total_trainable != expected
                or actual_bf16 != expected
            ):
                raise RuntimeError(
                    "TRAINABLE DTYPE SAFETY CHECK FAILED: "
                    f"expected {expected} BF16 trainable parameters, "
                    f"found total={total_trainable}, "
                    f"BF16={actual_bf16}, "
                    f"dtypes={dict(trainable)}"
                )

            self._dtype_checked = True

        return super().training_step(
            model,
            inputs,
            *args,
            **kwargs,
        )

def log_rank(message: str, rank: int) -> None:
    print(f"[rank={rank}] {message}", flush=True)


def install_single_shard_patch(global_rank: int) -> None:
    """
    The v2 loader manually shards with split_dataset_by_node().
    SFTTrainer/Accelerate then shards IterableDataset again.

    That results in double sharding.

    Keep the v2 loader's filtering, JSONL streaming and shuffle logic,
    but replace its manual distributed split with an identity operation.
    """

    if not hasattr(dolci_loader, "split_dataset_by_node"):
        raise RuntimeError(
            "Could not find module-level split_dataset_by_node in "
            "dolci_intervention_loader_v2. Refusing to train because "
            "the anti-double-sharding fix cannot be guaranteed."
        )

    def identity_split(dataset, rank, world_size):
        if rank == 0:
            print(
                "[v3] loader split_dataset_by_node intercepted -> identity. "
                "Accelerate will perform the ONLY distributed dataset shard.",
                flush=True,
            )
        return dataset

    dolci_loader.split_dataset_by_node = identity_split

    if global_rank == 0:
        print(
            "[v3] Single-shard patch installed.",
            flush=True,
        )


def barrier(label: str, rank: int, world_size: int) -> None:
    if world_size <= 1:
        return

    if not dist.is_available() or not dist.is_initialized():
        return

    log_rank(
        f"Entering distributed barrier: {label}",
        rank,
    )

    if torch.cuda.is_available():
        dist.barrier(
            device_ids=[torch.cuda.current_device()]
        )
    else:
        dist.barrier()

    log_rank(
        f"Passed distributed barrier: {label}",
        rank,
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--condition",
        required=True,
        choices=VALID_CONDITIONS,
    )

    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    budget = parser.add_mutually_exclusive_group(
        required=True
    )

    budget.add_argument(
        "--max-steps",
        type=int,
        default=None,
    )

    budget.add_argument(
        "--full-dataset-pass",
        action="store_true",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
    )

    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--save-steps",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help=(
            "Checkpoint directory, or 'latest'. "
            "Only resume checkpoints produced by this v3 script. "
            "DO NOT resume the old v2 checkpoint-17000."
        ),
    )

    return parser.parse_args()


def distributed_info():
    local_rank = int(
        os.environ.get("LOCAL_RANK", "0")
    )

    global_rank = int(
        os.environ.get("RANK", "0")
    )

    world_size = int(
        os.environ.get("WORLD_SIZE", "1")
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return local_rank, global_rank, world_size


def find_latest_checkpoint(output_dir: Path):
    checkpoints = []

    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue

        try:
            step = int(
                path.name.split("-")[-1]
            )
            checkpoints.append(
                (step, path)
            )
        except ValueError:
            pass

    if not checkpoints:
        return None

    return str(
        max(
            checkpoints,
            key=lambda x: x[0],
        )[1]
    )


def checkpoint_step(checkpoint):
    if not checkpoint:
        return 0

    try:
        return int(
            Path(checkpoint).name.split("-")[-1]
        )
    except ValueError as exc:
        raise ValueError(
            f"Could not determine checkpoint step "
            f"from {checkpoint}"
        ) from exc


def main():
    args = parse_args()

    (
        local_rank,
        global_rank,
        world_size,
    ) = distributed_info()

    is_main_process = global_rank == 0

    # ==========================================================
    # CRITICAL FIX:
    # Disable manual loader sharding.
    # SFTTrainer/Accelerate performs the only distributed shard.
    # ==========================================================
    install_single_shard_patch(
        global_rank
    )

    cuda_device = (
        torch.cuda.current_device()
        if torch.cuda.is_available()
        else "CPU"
    )
    log_rank(
        f"Process started | local_rank={local_rank} "
        f"world_size={world_size} cuda_device={cuda_device}",
        global_rank,
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    set_seed(args.seed)

    dataset_rows = expected_row_count(
        args.condition
    )

    global_effective_batch = (
        world_size
        * args.batch_size
        * args.gradient_accumulation_steps
    )

    if args.full_dataset_pass:
        max_steps = (
            dataset_rows
            // global_effective_batch
        ) - 2 

        consumed_rows = (
            max_steps
            * global_effective_batch
        )

        omitted_rows = (
            dataset_rows
            - consumed_rows
        )

    else:
        max_steps = args.max_steps
        consumed_rows = None
        omitted_rows = None

    resume = (
        args.resume_from_checkpoint
    )

    if resume == "latest":
        resume = find_latest_checkpoint(
            output_dir
        )

        if resume is None:
            raise RuntimeError(
                f"No checkpoint-* found in "
                f"{output_dir}"
            )

    resume_step = checkpoint_step(
        resume
    )
    if resume:
        resume_dir = Path(resume).parent
        if not (resume_dir / "run_config_v3.json").exists():
            raise RuntimeError(
                f"Refusing to resume {resume}: no run_config_v3.json "
                "found next to it. This looks like a v2 (double-sharded) "
                "checkpoint. Start a fresh v3 run in a new output dir."
            )
    if is_main_process:
        print(
            "=" * 100,
            flush=True,
        )

        print(
            "OLMo-3 DOLCI distributed QLoRA V3",
            flush=True,
        )

        print(
            "CORRECTED SINGLE-SHARD PIPELINE",
            flush=True,
        )

        print(
            "=" * 100,
            flush=True,
        )

        print(
            "Condition:",
            args.condition,
            flush=True,
        )

        print(
            "Dataset rows:",
            f"{dataset_rows:,}",
            flush=True,
        )

        print(
            "World size:",
            world_size,
            flush=True,
        )

        print(
            "Per-GPU batch size:",
            args.batch_size,
            flush=True,
        )

        print(
            "Gradient accumulation:",
            args.gradient_accumulation_steps,
            flush=True,
        )

        print(
            "Global effective batch:",
            global_effective_batch,
            flush=True,
        )

        print(
            "Optimizer steps:",
            f"{max_steps:,}",
            flush=True,
        )

        if consumed_rows is not None:
            print(
                "Rows consumed:",
                f"{consumed_rows:,}",
                flush=True,
            )

            print(
                "Rows omitted:",
                omitted_rows,
                flush=True,
            )

        print(
            "Output:",
            output_dir,
            flush=True,
        )

        print(
            "Resume checkpoint:",
            resume,
            flush=True,
        )

        print(
            "Resume step:",
            f"{resume_step:,}",
            flush=True,
        )

        print(
            "Dataset sharding: "
            "SFTTrainer/Accelerate ONLY",
            flush=True,
        )

        print(
            "Optimizer: adamw_8bit",
            flush=True,
        )

    log_rank(
        "Constructing GLOBAL streaming dataset...",
        global_rank,
    )

    # resume_skip_samples deliberately set to zero.
    # Trainer performs checkpoint skipping AFTER the correct
    # distributed shard.
    train_dataset = (
        dolci_loader.load_dolci_intervention(
            args.condition,
            shuffle_buffer_size=(
                args.shuffle_buffer_size
            ),
            seed=args.seed,
            rank=global_rank,
            world_size=world_size,
            resume_skip_samples=0,
        )
    )

    log_rank(
        (
            "Global streaming dataset ready; "
            "manual rank sharding disabled."
        ),
        global_rank,
    )

    log_rank(
        "Loading tokenizer...",
        global_rank,
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=True,
        )
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=(
            torch.bfloat16
        ),
        bnb_4bit_use_double_quant=True,
    )

    log_rank(
        (
            f"Loading model on "
            f"local GPU {local_rank}..."
        ),
        global_rank,
    )

    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=(
                bnb_config
            ),
            device_map={
                "": local_rank
            },
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
    )

    model.config.use_cache = False

    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )

    model.config.eos_token_id = (
        tokenizer.eos_token_id
    )

    if (
        tokenizer.bos_token_id
        is not None
    ):
        model.config.bos_token_id = (
            tokenizer.bos_token_id
        )

    if (
        getattr(
            model,
            "generation_config",
            None,
        )
        is not None
    ):
        model.generation_config.pad_token_id = (
            tokenizer.pad_token_id
        )

        model.generation_config.eos_token_id = (
            tokenizer.eos_token_id
        )

        if (
            tokenizer.bos_token_id
            is not None
        ):
            model.generation_config.bos_token_id = (
                tokenizer.bos_token_id
            )

    model = (
        prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    training_args = SFTConfig(
        output_dir=str(
            output_dir
        ),

        per_device_train_batch_size=(
            args.batch_size
        ),

        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
        ),

        max_steps=max_steps,

        dataloader_drop_last=True,

        learning_rate=(
            args.learning_rate
        ),

        lr_scheduler_type="linear",

        warmup_steps=int(
            max_steps * 0.03
        ),

        weight_decay=0.0,

        bf16=True,
        fp16=False,

        gradient_checkpointing=True,

        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },

        # Changed from paged_adamw_8bit.
        optim="adamw_8bit",

        max_length=(
            args.max_seq_length
        ),

        loss_type="nll",

        dataset_text_field="text",

        packing=False,

        logging_steps=(
            args.logging_steps
        ),

        save_steps=(
            args.save_steps
        ),

        save_total_limit=(
            args.save_total_limit
        ),

        report_to="none",

        seed=args.seed,

        data_seed=args.seed,

        remove_unused_columns=True,

        ddp_find_unused_parameters=False,

        accelerator_config={
            "dispatch_batches": False,
            "split_batches": False,
        },
    )

    if is_main_process:

        run_config = {
            "script_version": "v3",

            "pipeline_fix": (
                "single shard in "
                "SFTTrainer/Accelerate only"
            ),

            "condition": (
                args.condition
            ),

            "model": (
                args.model
            ),

            "dataset_rows": (
                dataset_rows
            ),

            "world_size": (
                world_size
            ),

            "per_device_train_batch_size": (
                args.batch_size
            ),

            "gradient_accumulation_steps": (
                args.gradient_accumulation_steps
            ),

            "global_effective_batch": (
                global_effective_batch
            ),

            "max_steps": (
                max_steps
            ),

            "rows_consumed": (
                consumed_rows
            ),

            "rows_omitted_by_batch_divisibility": (
                omitted_rows
            ),

            "learning_rate": (
                args.learning_rate
            ),

            "max_seq_length": (
                args.max_seq_length
            ),

            "shuffle_buffer_size": (
                args.shuffle_buffer_size
            ),

            "seed": (
                args.seed
            ),

            "optimizer": (
                "adamw_8bit"
            ),

            "manual_split_dataset_by_node": (
                False
            ),

            "resume_checkpoint": (
                resume
            ),

            "resume_step": (
                resume_step
            ),

            "lora": {
                "r": args.lora_r,
                "alpha": (
                    args.lora_alpha
                ),
                "dropout": (
                    args.lora_dropout
                ),
            },
        }

        (
            output_dir
            / "run_config_v3.json"
        ).write_text(
            json.dumps(
                run_config,
                indent=2,
            )
        )

    barrier(
        "before SFTTrainer construction",
        global_rank,
        world_size,
    )

    # Back to standard SFTTrainer.
    # No LocalMetricsSFTTrainer monkeypatch.
    trainer = LocalMetricsSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    barrier(
        "before trainer.train",
        global_rank,
        world_size,
    )

    log_rank(
        (
            f"Calling trainer.train() "
            f"| resume={resume}"
        ),
        global_rank,
    )

    trainer.train(
        resume_from_checkpoint=(
            resume
            if resume
            else None
        )
    )

    log_rank(
        "trainer.train() completed.",
        global_rank,
    )

    barrier(
        "after trainer.train",
        global_rank,
        world_size,
    )

    trainer.save_model(
        str(output_dir)
    )

    if is_main_process:
        tokenizer.save_pretrained(
            str(output_dir)
        )

        print(
            "Saved:",
            output_dir,
            flush=True,
        )

    if (
        dist.is_available()
        and dist.is_initialized()
    ):
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
