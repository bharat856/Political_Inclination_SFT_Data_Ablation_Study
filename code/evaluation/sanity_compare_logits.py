from pathlib import Path
import csv
import math

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


BASE_MODEL = (
    "/fs/scratch/users/bharatpi/causal_political_content/"
    "hf_cache/allenai_Olmo-3-1025-7B"
)

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
    messages = [
        {"role": "user", "content": prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return f"USER:\n{prompt}\n\nASSISTANT:\n"


def clean_token(tokenizer, token_id):
    text = tokenizer.decode([token_id])

    # Make whitespace visible in terminal/CSV.
    return repr(text)


@torch.inference_mode()
def get_next_token_logits(model, tokenizer, prompt):
    formatted = format_prompt(tokenizer, prompt)

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    outputs = model(**inputs)

    # Distribution for the first token the assistant would generate.
    logits = outputs.logits[0, -1, :].float().cpu()

    return logits


def js_divergence(logits_a, logits_b):
    """
    Jensen-Shannon divergence between two next-token distributions.
    Stable and symmetric.
    """

    log_p = F.log_softmax(logits_a, dim=-1)
    log_q = F.log_softmax(logits_b, dim=-1)

    p = log_p.exp()
    q = log_q.exp()

    m = 0.5 * (p + q)
    log_m = torch.log(m.clamp_min(1e-12))

    kl_pm = torch.sum(p * (log_p - log_m))
    kl_qm = torch.sum(q * (log_q - log_m))

    return float(0.5 * (kl_pm + kl_qm))


def compare_logits(full_logits, other_logits):
    diff = full_logits - other_logits

    cosine = F.cosine_similarity(
        full_logits.unsqueeze(0),
        other_logits.unsqueeze(0),
    ).item()

    mean_abs = diff.abs().mean().item()
    max_abs = diff.abs().max().item()
    l2 = torch.linalg.vector_norm(diff).item()

    js = js_divergence(full_logits, other_logits)

    full_top5 = set(torch.topk(full_logits, 5).indices.tolist())
    other_top5 = set(torch.topk(other_logits, 5).indices.tolist())

    top5_overlap = len(full_top5 & other_top5)

    return {
        "cosine_similarity": cosine,
        "mean_abs_logit_diff": mean_abs,
        "max_abs_logit_diff": max_abs,
        "l2_logit_diff": l2,
        "js_divergence": js,
        "top5_overlap": top5_overlap,
    }


def main():
    out_dir = Path("outputs/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("Loading base model...")

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    base_model.eval()

    all_logits = {}

    for model_name, adapter_path in ADAPTERS.items():
        print()
        print("=" * 80)
        print(f"Loading adapter: {model_name}")
        print(adapter_path)
        print("=" * 80)

        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
        )
        model.eval()

        model_logits = []

        for i, prompt in enumerate(PROMPTS, 1):
            print(f"[{model_name}] Prompt {i}/{len(PROMPTS)}")

            logits = get_next_token_logits(
                model,
                tokenizer,
                prompt,
            )

            model_logits.append(logits)

        all_logits[model_name] = model_logits

        # Restore clean base before loading another adapter.
        base_model = model.unload()

        del model
        torch.cuda.empty_cache()

    print()
    print("Calculating comparisons against Full...")

    csv_path = out_dir / "sanity_logit_comparison.csv"

    rows = []

    for model_name in ["NoLeft", "NoRight", "LRBalanced"]:

        for i, prompt in enumerate(PROMPTS):
            full_logits = all_logits["Full"][i]
            other_logits = all_logits[model_name][i]

            metrics = compare_logits(
                full_logits,
                other_logits,
            )

            full_top = torch.argmax(full_logits).item()
            other_top = torch.argmax(other_logits).item()

            rows.append({
                "prompt_id": i + 1,
                "prompt": prompt,
                "comparison": f"Full_vs_{model_name}",

                "full_top1_id": full_top,
                "full_top1_token": clean_token(
                    tokenizer,
                    full_top,
                ),

                "other_top1_id": other_top,
                "other_top1_token": clean_token(
                    tokenizer,
                    other_top,
                ),

                "top1_same": full_top == other_top,

                **metrics,
            })

    fieldnames = list(rows[0].keys())

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("SUMMARY")
    print("=" * 80)

    for model_name in ["NoLeft", "NoRight", "LRBalanced"]:

        subset = [
            r for r in rows
            if r["comparison"] == f"Full_vs_{model_name}"
        ]

        n = len(subset)

        mean_cosine = sum(
            r["cosine_similarity"] for r in subset
        ) / n

        mean_abs = sum(
            r["mean_abs_logit_diff"] for r in subset
        ) / n

        mean_js = sum(
            r["js_divergence"] for r in subset
        ) / n

        same_top1 = sum(
            r["top1_same"] for r in subset
        )

        mean_top5_overlap = sum(
            r["top5_overlap"] for r in subset
        ) / n

        print()
        print(f"Full vs {model_name}")
        print("-" * 50)
        print(f"Mean cosine similarity:    {mean_cosine:.8f}")
        print(f"Mean absolute logit diff:  {mean_abs:.8f}")
        print(f"Mean JS divergence:        {mean_js:.8f}")
        print(f"Same top-1 token:          {same_top1}/{n}")
        print(
            f"Mean top-5 overlap:        "
            f"{mean_top5_overlap:.2f}/5"
        )

    print()
    print("Per-prompt results saved to:")
    print(csv_path)


if __name__ == "__main__":
    main()