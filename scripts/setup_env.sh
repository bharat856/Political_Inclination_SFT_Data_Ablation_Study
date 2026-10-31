
#!/bin/bash

export PROJECT_ROOT=/fs/scratch/users/bharatpi/causal_political_content

export HF_HOME=$PROJECT_ROOT/cache/huggingface

export HF_HUB_CACHE=$PROJECT_ROOT/cache/huggingface/hub

export TRANSFORMERS_CACHE=$PROJECT_ROOT/cache/huggingface/transformers

export HF_DATASETS_CACHE=$PROJECT_ROOT/cache/huggingface/datasets

export VLLM_CACHE_ROOT=$PROJECT_ROOT/cache/vllm
export VLLM_CACHE_DIR=$PROJECT_ROOT/cache/vllm

export UV_CACHE_DIR=$PROJECT_ROOT/cache/uv

# Use max 2 GPUs. Change numbers based on free GPUs from nvidia-smi.

export CUDA_VISIBLE_DEVICES=3,4

source $PROJECT_ROOT/envs/classifier/bin/activate

echo "Environment ready"

echo "PROJECT_ROOT=$PROJECT_ROOT"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo "HF_HOME=$HF_HOME"

echo "VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT"
echo "VLLM_CACHE_DIR=$VLLM_CACHE_DIR"

echo "Python=$(which python)"

