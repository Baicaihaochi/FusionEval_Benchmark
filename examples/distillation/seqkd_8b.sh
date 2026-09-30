#!/usr/bin/env bash
set -euo pipefail
# Use your prepared Slime/Megatron environment and an existing Ray cluster.
# Fill paths and endpoints here, or export them before launching.
# Empty SLIME_ROOT uses the bundled slime/ directory.
export SLIME_ROOT="${SLIME_ROOT:-}"
export MEGATRON_ROOT="${MEGATRON_ROOT:-}"
export STUDENT_HF="${STUDENT_HF:-}"
export STUDENT_CHECKPOINT="${STUDENT_CHECKPOINT:-}"
export STUDENT_STEP="${STUDENT_STEP:-}"
export PROMPT_DATA="${PROMPT_DATA:-}"
export OUTPUT_DIR="${OUTPUT_DIR:-}"
export RAY_JOB_ADDRESS="${RAY_JOB_ADDRESS:-}"

ARGS=(
  --swiglu
  --num-layers 36
  --hidden-size 4096
  --ffn-hidden-size 12288
  --num-attention-heads 32
  --group-query-attention
  --num-query-groups 8
  --use-rotary-position-embeddings
  --disable-bias-linear
  --normalization RMSNorm
  --norm-epsilon 1e-6
  --rotary-base 2000000
  --vocab-size 151936
  --kv-channels 128
  --qk-layernorm
  --untie-embeddings-and-output-weights
  --seq-length 40960
  --max-position-embeddings 40960
  --hf-checkpoint "${STUDENT_HF}"
  --ref-load "${STUDENT_CHECKPOINT}"
  --ref-ckpt-step "${STUDENT_STEP}"
  --no-load-optim
  --no-load-rng
  --finetune
  --save "${OUTPUT_DIR}/checkpoints"
  --save-interval 5
  --start-rollout-id 0
  --prompt-data "${PROMPT_DATA}"
  --data-source-path seqkd_data.SeqKDDataSource
  --rollout-function-path seqkd_data.generate_rollout
  --input-key messages
  --metadata-key metadata
  --tool-key tools
  --num-rollout 50
  --rollout-batch-size 512
  --global-batch-size 512
  --n-samples-per-prompt 1
  --rollout-seed 42
  --balance-data
  --loss-type sft_loss
  --loss-mask-type qwen3_5
  --disable-compute-advantages-and-returns
  --debug-train-only
  --optimizer adam
  --lr 3e-6
  --lr-decay-style constant
  --lr-warmup-init 3e-6
  --lr-warmup-iters 0
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.95
  --tensor-model-parallel-size 4
  --context-parallel-size 2
  --pipeline-model-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --sequence-parallel
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu 20480
  --actor-num-nodes 1
  --actor-num-gpus-per-node 8
  --num-gpus-per-node 8
  --seed 1234
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend fused
)

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${PYTHON:-python3}" "$ROOT/scripts/distillation/run.py" "$@" -- "${ARGS[@]}"
