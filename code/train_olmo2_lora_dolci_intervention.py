from pathlib import Path
import argparse
import json
import os

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig
from trl import SFTTrainer

from dolci_intervention_loader import (
    VALID_CONDITIONS,
    expected_row_count,
    load_dolci_intervention,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "hf_cache/allenai_OLMo-2-1124-7B"


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

    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)

    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)

    parser.add_argument("--shuffle-buffer-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Checkpoint directory, or 'latest' to resume automatically.",
    )

    return parser.parse_args()


def find_latest_checkpoint(output_dir: Path):
    checkpoints = []

    for p in output_dir.glob("checkpoint-*"):
        if not p.is_dir():
            continue

        try:
            step = int(p.name.split("-")[-1])
            checkpoints.append((step, p))
        except ValueError:
            continue

    if not checkpoints:
        return None

    return str(max(checkpoints, key=lambda x: x[0])[1])


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    print("=" * 100)
    print("OLMo-2 DOLCI intervention QLoRA")
    print("=" * 100)
    print("Condition:", args.condition)
    print("Expected dataset rows:", f"{expected_row_count(args.condition):,}")
    print("Maximum optimizer steps:", args.max_steps)
    print(
        "Effective batch size:",
        args.batch_size * args.gradient_accumulation_steps,
    )
    print("Output:", output_dir)

    train_dataset = load_dolci_intervention(
        args.condition,
        shuffle_buffer_size=args.shuffle_buffer_size,
        seed=args.seed,
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
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    model.config.use_cache = False

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

    training_args = TrainingArguments(
        output_dir=str(output_dir),

        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,

        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        warmup_steps=0,
        weight_decay=0.0,

        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",

        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        report_to="none",
        seed=args.seed,
        data_seed=args.seed,

        remove_unused_columns=False,
    )

    run_config = {
        "condition": args.condition,
        "model": args.model,
        "expected_dataset_rows": expected_row_count(args.condition),
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": (
            args.batch_size * args.gradient_accumulation_steps
        ),
        "learning_rate": args.learning_rate,
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
        print("Latest checkpoint:", resume)

    trainer.train(
        resume_from_checkpoint=resume if resume else None
    )

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print("Saved:", output_dir)


if __name__ == "__main__":
    main()
