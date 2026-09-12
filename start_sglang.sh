#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/.venv}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
MODEL="${MODEL:-${SCRIPT_DIR}/Qwen2.5-0.5B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
GPU="${GPU:-7}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.2}"
HF_HOME="${HF_HOME:-${SCRIPT_DIR}/.cache/huggingface}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "错误：找不到虚拟环境 ${VENV_DIR}" >&2
  exit 1
fi

if [[ ! -d "${CUDA_HOME}" ]]; then
  echo "错误：找不到 CUDA 目录 ${CUDA_HOME}" >&2
  exit 1
fi

export CUDA_HOME HF_HOME
export CUDA_VISIBLE_DEVICES="${GPU}"
export PATH="${VENV_DIR}/bin:${CUDA_HOME}/bin:${PATH}"

echo "启动 SGLang：model=${MODEL}, served-model=${SERVED_MODEL_NAME}, http://${HOST}:${PORT}, GPU=${GPU}"

exec "${VENV_DIR}/bin/python" -m sglang.launch_server \
  --model-path "${MODEL}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --attention-backend triton \
  --sampling-backend pytorch \
  --log-level warning \
  "$@"
# python -m sglang.launch_server \
#   --model-path ./Qwen2.5-0.5B-Instruct --port 43440
# # python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --port 43440 --device cpu
