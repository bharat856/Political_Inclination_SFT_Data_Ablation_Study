from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer

MODEL_PATH = "hf_cache/allenai_OLMo-2-1124-7B"
DATA_PATH = "data/processed/tulu_olmo2_sft/final_sft_variants/variant_01_full_original_sft_100k_textonly.jsonl"
OUT_DIR = "outputs/sft_runs/variant_01_full_original_sft_100k_lora"

dataset = load_dataset("json", data_files=DATA_PATH, split="train")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model.config.use_cache = False

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    num_train_epochs=1,
    bf16=True,
    logging_steps=10,
    save_steps=1000,
    save_total_limit=2,
    report_to="none",
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model(OUT_DIR)

print("Saved:", OUT_DIR)
