#!/bin/bash

set -e

PROJECT_ROOT="/fs/scratch/users/bharatpi/causal_political_content"
cd "$PROJECT_ROOT"

REQUESTED_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

if [ -f "$PROJECT_ROOT/scripts/setup_env.sh" ]; then
  # shellcheck source=/dev/null
  source "$PROJECT_ROOT/scripts/setup_env.sh"
fi

export PROJECT_ROOT="$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="$REQUESTED_CUDA_VISIBLE_DEVICES"
export CUDA_HOME="${CUDA_HOME:-$PROJECT_ROOT/envs/cuda12}"
export CUDA_HOME_TARGET="$CUDA_HOME/targets/x86_64-linux"

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:$CUDA_HOME_TARGET/lib:/usr/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:$CUDA_HOME_TARGET/lib:/usr/lib64:${LIBRARY_PATH:-}"
export CPATH="$CUDA_HOME/include:$CUDA_HOME_TARGET/include:${CPATH:-}"

export CC="$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++"

export HF_HOME="$PROJECT_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export VLLM_CACHE_ROOT="$PROJECT_ROOT/cache/vllm"
export VLLM_CACHE_DIR="$PROJECT_ROOT/cache/vllm"
export TORCH_HOME="$PROJECT_ROOT/cache/torch"
export CONDA_PKGS_DIRS="$PROJECT_ROOT/cache/conda_pkgs"
export UV_CACHE_DIR="$PROJECT_ROOT/cache/uv"

mkdir -p "$CUDA_HOME/lib64" "$HF_HOME" "$VLLM_CACHE_ROOT" "$TORCH_HOME" "$CONDA_PKGS_DIRS"

if [ ! -e "$CUDA_HOME/lib64/libcudart.so" ]; then
  CUDART_SOURCE="$(find "$CUDA_HOME" -type f -name 'libcudart.so*' | sort | head -n 1 || true)"
  if [ -n "$CUDART_SOURCE" ]; then
    cp -L "$CUDART_SOURCE" "$CUDA_HOME/lib64/libcudart.so"
  else
    echo "Warning: no real libcudart.so* found under $CUDA_HOME"
  fi
fi

if [ -e /usr/lib64/libcuda.so ] && [ ! -e "$CUDA_HOME/lib64/libcuda.so" ]; then
  ln -s /usr/lib64/libcuda.so "$CUDA_HOME/lib64/libcuda.so"
fi

echo "Benchmark environment ready"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_HOME=$CUDA_HOME"
echo "CC=$CC"
echo "CXX=$CXX"
echo "HF_HOME=$HF_HOME"
echo "VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT"
echo "TORCH_HOME=$TORCH_HOME"
echo "Python=$(which python)"
