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

from dolci_intervention_loader_v2 import (
    VALID_CONDITIONS,
    expected_row_count,
    load_dolci_intervention,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "hf_cache/allenai_Olmo-3-1025-7B"


warnings.filterwarnings(
    "ignore",
    message=r".*_check_is_size will be removed.*",
    category=FutureWarning,
)


def log_rank(message: str, rank: int) -> None:
    print(f"[rank={rank}] {message}", flush=True)


def barrier(label: str, rank: int, world_size: int) -> None:
    """
    Diagnostic synchronization point.

    torchrun provides rank/world-size environment variables immediately,
    but the default torch.distributed process group is initialized later
    by Accelerate/SFTTrainer. Therefore barriers before that point must
    not call dist.barrier().
    """
    if world_size <= 1:
        log_rank(f"Barrier skipped (single process): {label}", rank)
        return

    if not dist.is_available():
        log_rank(
            f"Barrier skipped (torch.distributed unavailable): {label}",
            rank,
        )
        return

    if not dist.is_initialized():
        log_rank(
            f"Barrier deferred (process group not initialized yet): {label}",
            rank,
        )
        return

    log_rank(f"Entering distributed barrier: {label}", rank)

    if torch.cuda.is_available():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()

    log_rank(f"Passed distributed barrier: {label}", rank)


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

    budget = parser.add_mutually_exclusive_group(required=True)

    budget.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Use a fixed optimizer-step budget.",
    )

    budget.add_argument(
        "--full-dataset-pass",
        action="store_true",
        help="Calculate steps for one pass through the intervention dataset.",
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

    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--save-total-limit", type=int, default=2)

    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=100_000,
    )

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Checkpoint directory, or 'latest'.",
    )

    return parser.parse_args()


def distributed_info():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    global_rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return local_rank, global_rank, world_size


def find_latest_checkpoint(output_dir: Path):
    checkpoints = []

    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue

        try:
            step = int(path.name.split("-")[-1])
            checkpoints.append((step, path))
        except ValueError:
            pass

    if not checkpoints:
        return None

    return str(max(checkpoints, key=lambda item: item[0])[1])


def checkpoint_step(checkpoint):
    if not checkpoint:
        return 0

    try:
        return int(Path(checkpoint).name.split("-")[-1])
    except ValueError as exc:
        raise ValueError(
            f"Could not determine checkpoint step from: {checkpoint}"
        ) from exc


def main():
    args = parse_args()

    local_rank, global_rank, world_size = distributed_info()
    is_main_process = global_rank == 0

    log_rank(
        (
            f"Process started | "
            f"local_rank={local_rank} "
            f"world_size={world_size} "
            f"cuda_device="
            f"{torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}"
        ),
        global_rank,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    dataset_rows = expected_row_count(args.condition)

    global_effective_batch = (
        world_size
        * args.batch_size
        * args.gradient_accumulation_steps
    )

    if args.full_dataset_pass:
        max_steps = dataset_rows // global_effective_batch
    else:
        max_steps = args.max_steps

    consumed_rows = max_steps * global_effective_batch
    omitted_rows = dataset_rows - consumed_rows

    resume = args.resume_from_checkpoint

    if resume == "latest":
        resume = find_latest_checkpoint(output_dir)

    resume_step = checkpoint_step(resume)

    resume_skip_samples_per_rank = (
        resume_step
        * args.gradient_accumulation_steps
        * args.batch_size
    )

    if is_main_process:
        print("=" * 100, flush=True)
        print("OLMo-3 DOLCI distributed QLoRA V2", flush=True)
        print("=" * 100, flush=True)
        print("Condition:", args.condition, flush=True)
        print("Dataset rows:", f"{dataset_rows:,}", flush=True)
        print("World size:", world_size, flush=True)
        print("Per-GPU batch size:", args.batch_size, flush=True)
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
        print("Optimizer steps:", f"{max_steps:,}", flush=True)
        print("Rows consumed:", f"{consumed_rows:,}", flush=True)
        print(
            "Rows omitted by batch divisibility:",
            omitted_rows,
            flush=True,
        )
        print("Output:", output_dir, flush=True)
        print("Resume checkpoint:", resume, flush=True)
        print(
            "Resume optimizer step:",
            f"{resume_step:,}",
            flush=True,
        )
        print(
            "Expected replay samples per rank:",
            f"{resume_skip_samples_per_rank:,}",
            flush=True,
        )
        print(
            "Expected replay samples globally:",
            f"{resume_skip_samples_per_rank * world_size:,}",
            flush=True,
        )

    barrier(
        "before dataset construction",
        global_rank,
        world_size,
    )

    log_rank("Constructing streaming dataset...", global_rank)

    train_dataset = load_dolci_intervention(
        args.condition,
        shuffle_buffer_size=args.shuffle_buffer_size,
        seed=args.seed,
        rank=global_rank,
        world_size=world_size,
        resume_skip_samples=resume_skip_samples_per_rank,
    )

    log_rank("Streaming dataset constructed.", global_rank)

    barrier(
        "after dataset construction",
        global_rank,
        world_size,
    )

    log_rank("Loading tokenizer...", global_rank)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    log_rank(
        (
            f"Tokenizer ready | "
            f"pad={tokenizer.pad_token_id} "
            f"bos={tokenizer.bos_token_id} "
            f"eos={tokenizer.eos_token_id}"
        ),
        global_rank,
    )

    barrier(
        "after tokenizer",
        global_rank,
        world_size,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    log_rank(
        f"Loading model on local GPU {local_rank}...",
        global_rank,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map={"": local_rank},
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )

    log_rank("Base model loaded.", global_rank)

    model.config.use_cache = False

    # Explicitly align special token IDs before Trainer construction.
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    if tokenizer.bos_token_id is not None:
        model.config.bos_token_id = tokenizer.bos_token_id

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

        if tokenizer.bos_token_id is not None:
            model.generation_config.bos_token_id = tokenizer.bos_token_id

    log_rank(
        "Model/tokenizer special-token IDs aligned.",
        global_rank,
    )

    barrier(
        "after model load",
        global_rank,
        world_size,
    )

    log_rank(
        "Preparing model for k-bit training...",
        global_rank,
    )

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    log_rank(
        "K-bit preparation complete.",
        global_rank,
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

    log_rank("Building SFTConfig...", global_rank)

    training_args = SFTConfig(
        output_dir=str(output_dir),

        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
        ),
        max_steps=max_steps,

        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        warmup_steps=int(max_steps * 0.03),
        weight_decay=0.0,

        bf16=True,
        fp16=False,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },

        optim="adamw_8bit",

        max_length=args.max_seq_length,
        loss_type="nll",
        dataset_text_field="text",
        packing=False,

        # Avoid Trainer's cross-rank token-count gather.
        # This is the collective visible in the previous NCCL timeout.
        average_tokens_across_devices=True,

        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

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

    log_rank(
        (
            f"SFTConfig ready | "
            f"max_length={training_args.max_length} "
            f"average_tokens_across_devices="
            f"{training_args.average_tokens_across_devices}"
        ),
        global_rank,
    )

    if is_main_process:
        run_config = {
            "script_version": "v2",
            "training_type": "full-dataset distributed QLoRA",
            "condition": args.condition,
            "model": args.model,
            "dataset_rows": dataset_rows,
            "world_size": world_size,
            "per_device_train_batch_size": args.batch_size,
            "gradient_accumulation_steps": (
                args.gradient_accumulation_steps
            ),
            "global_effective_batch": global_effective_batch,
            "max_steps": max_steps,
            "rows_consumed": consumed_rows,
            "rows_omitted_by_batch_divisibility": omitted_rows,
            "learning_rate": args.learning_rate,
            "max_seq_length": args.max_seq_length,
            "shuffle_buffer_size": args.shuffle_buffer_size,
            "seed": args.seed,
            "average_tokens_across_devices": training_args.average_tokens_across_devices,
            "resume_checkpoint": resume,
            "resume_step": resume_step,
            "resume_skip_samples_per_rank": (
                resume_skip_samples_per_rank
            ),
            "lora": {
                "r": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
            },
        }

        (output_dir / "run_config_v2.json").write_text(
            json.dumps(run_config, indent=2)
        )

    barrier(
        "before SFTTrainer construction",
        global_rank,
        world_size,
    )

    log_rank("Constructing SFTTrainer...", global_rank)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    log_rank("SFTTrainer constructed.", global_rank)

    barrier(
        "before trainer.train",
        global_rank,
        world_size,
    )

    log_rank(
        f"Calling trainer.train() | resume={resume}",
        global_rank,
    )

    if is_main_process and resume_step > 0:
        print(
            "\n"
            "Checkpoint resume is starting.\n"
            "Trainer may keep its normal optimizer-step progress bar "
            "unchanged while reconstructing the iterable dataset.\n"
            "The 'Resume replay rank0' bar tracks examples flowing "
            "through rank 0 during this phase.\n",
            flush=True,
        )

    trainer.train(
        resume_from_checkpoint=resume if resume else None
    )

    log_rank("trainer.train() completed.", global_rank)

    barrier(
        "after trainer.train",
        global_rank,
        world_size,
    )

    log_rank("Saving final adapter...", global_rank)

    trainer.save_model(str(output_dir))

    if is_main_process:
        tokenizer.save_pretrained(str(output_dir))
        print("Saved:", output_dir, flush=True)

    if dist.is_available() and dist.is_initialized():
        log_rank(
            "Destroying distributed process group cleanly.",
            global_rank,
        )
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
