#!/usr/bin/env python3

import argparse
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Classify this sentence as 1 left, 2 center, or 3 right: Workers should receive higher wages. Answer:")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    args = parser.parse_args()

    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
    print("HF_HOME:", os.environ.get("HF_HOME"), flush=True)
    print("Model:", args.model, flush=True)
    print("torch.cuda.device_count():", torch.cuda.device_count(), flush=True)

    print("Loading tokenizer...", flush=True)
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"Tokenizer loaded in {time.perf_counter() - t0:.2f}s", flush=True)

    print("Loading model with Transformers...", flush=True)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print(f"Model loaded in {time.perf_counter() - t0:.2f}s", flush=True)

    print("Running generation...", flush=True)
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    output = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print(f"Generation finished in {time.perf_counter() - t0:.2f}s", flush=True)
    print("Raw output:", repr(output), flush=True)

if __name__ == "__main__":
    main()
