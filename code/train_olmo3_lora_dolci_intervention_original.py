from pathlib import Path
import argparse
import json
import math
import os

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

from dolci_intervention_loader import (
    VALID_CONDITIONS,
    expected_row_count,
    load_dolci_intervention,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "hf_cache/allenai_Olmo-3-1025-7B"


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
        help="Micro-batch size per GPU.",
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


def main():
    args = parse_args()
    local_rank, global_rank, world_size = distributed_info()
    is_main_process = global_rank == 0

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
        # Use only complete global optimizer batches.
        max_steps = dataset_rows // global_effective_batch
    else:
        max_steps = args.max_steps

    consumed_rows = max_steps * global_effective_batch
    omitted_rows = dataset_rows - consumed_rows

    if is_main_process:
        print("=" * 100)
        print("OLMo-3 DOLCI distributed QLoRA")
        print("=" * 100)
        print("Condition:", args.condition)
        print("Dataset rows:", f"{dataset_rows:,}")
        print("World size:", world_size)
        print("Per-GPU batch size:", args.batch_size)
        print(
            "Gradient accumulation:",
            args.gradient_accumulation_steps,
        )
        print(
            "Global effective batch:",
            global_effective_batch,
        )
        print("Optimizer steps:", f"{max_steps:,}")
        print("Rows consumed:", f"{consumed_rows:,}")
        print("Rows omitted by batch divisibility:", omitted_rows)
        print("Output:", output_dir)

    train_dataset = load_dolci_intervention(
        args.condition,
        shuffle_buffer_size=args.shuffle_buffer_size,
        seed=args.seed,
        rank=global_rank,
        world_size=world_size,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map={"": local_rank},
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )

    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
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
        optim="paged_adamw_8bit",

        max_length=args.max_seq_length,
        loss_type="nll",
        dataset_text_field="text",
        packing=False,

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

    if is_main_process:
        run_config = {
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
            "lora": {
                "r": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
            },
        }

        (output_dir / "run_config.json").write_text(
            json.dumps(run_config, indent=2)
        )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    resume = args.resume_from_checkpoint

    if resume == "latest":
        resume = find_latest_checkpoint(output_dir)
        if is_main_process:
            print("Latest checkpoint:", resume)

    trainer.train(
        resume_from_checkpoint=resume if resume else None
    )

    trainer.save_model(str(output_dir))

    if is_main_process:
        tokenizer.save_pretrained(str(output_dir))
        print("Saved:", output_dir)


if __name__ == "__main__":
    main()
