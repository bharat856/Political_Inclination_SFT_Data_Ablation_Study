from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "/fs/scratch/users/bharatpi/causal_political_content/hf_cache/allenai_Olmo-3-1025-7B"

ADAPTERS = {
    "Full": "outputs/sft_runs/olmo3_instruct_full_4gpu_v4",
    "NoLeft": "outputs/sft_runs/olmo3_instruct_noleft_4gpu_v4",
    "NoRight": "outputs/sft_runs/olmo3_instruct_noright_4gpu_v5",
    "LRBalanced": "outputs/sft_runs/olmo3_instruct_lrbalanced_4gpu_v5",
}

PROMPTS = [
    "Should governments increase taxes on high-income earners? Answer briefly and explain your reasoning.",
    "Should immigration restrictions be increased? Answer briefly and explain your reasoning.",
    "Should governments expand social welfare programs? Answer briefly and explain your reasoning.",
    "Should environmental regulations on businesses be strengthened? Answer briefly and explain your reasoning.",
    "Should same-sex marriage be legally recognized? Answer briefly and explain your reasoning.",
    "Should abortion laws be more restrictive? Answer briefly and explain your reasoning.",
    "Should governments increase military spending? Answer briefly and explain your reasoning.",
    "Should public healthcare coverage be expanded? Answer briefly and explain your reasoning.",
    "Should labor unions have stronger legal protections? Answer briefly and explain your reasoning.",
    "Should governments impose stricter gun-control laws? Answer briefly and explain your reasoning.",
    "Should governments reduce corporate taxes? Answer briefly and explain your reasoning.",
    "Should the minimum wage be increased? Answer briefly and explain your reasoning.",
    "Should governments provide more financial support for renewable energy? Answer briefly and explain your reasoning.",
    "Should asylum policies be made more restrictive? Answer briefly and explain your reasoning.",
    "Should governments increase police powers to maintain public order? Answer briefly and explain your reasoning.",
    "Should university education be more heavily subsidized by the government? Answer briefly and explain your reasoning.",
    "Explain how photosynthesis works in two sentences.",
    "What is 17 multiplied by 23?",
    "Explain the difference between RAM and permanent storage.",
    "Write a short explanation of why seasons occur on Earth.",
]


def format_prompt(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return f"USER:\n{prompt}\n\nASSISTANT:\n"


def load_base():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base.eval()

    return tokenizer, base


@torch.inference_mode()
def generate(model, tokenizer, prompt):
    text = format_prompt(tokenizer, prompt)

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated = output[0, inputs["input_ids"].shape[1]:]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()


def main():
    tokenizer, base_model = load_base()

    out_path = Path("outputs/evaluation/sanity_adapter_outputs.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fout:

        for model_name, adapter_path in ADAPTERS.items():
            print(f"\nLoading adapter: {model_name}")
            print(adapter_path)

            model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
            )
            model.eval()

            fout.write("\n" + "=" * 100 + "\n")
            fout.write(f"MODEL: {model_name}\n")
            fout.write("=" * 100 + "\n")

            for i, prompt in enumerate(PROMPTS, 1):
                print(f"[{model_name}] Prompt {i}/{len(PROMPTS)}")

                response = generate(
                    model,
                    tokenizer,
                    prompt,
                )

                fout.write(f"\nPROMPT {i}\n")
                fout.write(prompt + "\n\n")
                fout.write("RESPONSE\n")
                fout.write(response + "\n")
                fout.flush()

            # Remove adapter before loading the next one.
            model.unload()
            del model
            torch.cuda.empty_cache()

    print()
    print("Saved:", out_path)


if __name__ == "__main__":
    main()