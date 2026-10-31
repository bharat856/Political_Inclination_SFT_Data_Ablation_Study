#!/bin/bash

set -e

# Example:
#   bash scripts/run_benchmark_v1.sh
#   bash scripts/run_benchmark_v1.sh cyankiwi/gemma-4-31B-it-AWQ-8bit meta-llama/Llama-3.1-8B-Instruct

PROJECT_ROOT="/fs/scratch/users/bharatpi/causal_political_content"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

# shellcheck source=/dev/null
source "$PROJECT_ROOT/scripts/setup_benchmark_env.sh"

if [ "$#" -gt 0 ]; then
  MODELS=("$@")
else
  MODELS=(
    "cyankiwi/gemma-4-31B-it-AWQ-8bit"
    "cyankiwi/Qwen3.5-27B-AWQ-BF16-INT8"
    "meta-llama/Llama-3.1-8B-Instruct"
  )
fi

python code/build_benchmark_v1.py \
  --input data/processed/manifesto/chunks/sample_30_manifesto_chunks_strict_balanced.csv \
  --output data/processed/benchmark_v1/benchmark_v1.csv \
  --prompt-version v2

python code/run_benchmark_v1.py \
  --benchmark data/processed/benchmark_v1/benchmark_v1.csv \
  --models "${MODELS[@]}" \
  --backend vllm \
  --prompt-version v2 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --output-dir outputs/benchmark_v1 \
  --fallback-transformers false

python code/evaluate_benchmark_v1.py \
  --predictions-dir outputs/benchmark_v1/predictions \
  --output-dir outputs/benchmark_v1/metrics
