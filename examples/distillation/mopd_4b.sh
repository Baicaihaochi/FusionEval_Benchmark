#!/usr/bin/env bash
set -eo pipefail
# 1 node: student on GPUs 0-2, teachers on 3-7.
# 2 nodes: head trains on 0-7; worker serves student on 0-2, teachers on 3-7.
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
export TEACHER_MATH_URL="${TEACHER_MATH_URL:-}"
export TEACHER_IF_URL="${TEACHER_IF_URL:-}"
export TEACHER_CODE_URL="${TEACHER_CODE_URL:-}"
export TEACHER_SCIENCE_URL="${TEACHER_SCIENCE_URL:-}"
export TEACHER_AGENT_URL="${TEACHER_AGENT_URL:-}"

# Local deployment; set LOCAL_SERVICES=0 to use existing Ray/teacher endpoints.
export LOCAL_SERVICES="${LOCAL_SERVICES:-1}"
export NODE_COUNT="${NODE_COUNT:-1}"  # 1 or 2; run this file with --worker on node 2.
export HEAD_IP="${HEAD_IP:-}"        # Required for 2 nodes.
export WORKER_IP="${WORKER_IP:-}"    # Required for 2 nodes.
export TEACHER_MATH_MODEL="${TEACHER_MATH_MODEL:-}"
export TEACHER_IF_MODEL="${TEACHER_IF_MODEL:-}"
export TEACHER_CODE_MODEL="${TEACHER_CODE_MODEL:-}"
export TEACHER_SCIENCE_MODEL="${TEACHER_SCIENCE_MODEL:-}"
export TEACHER_AGENT_MODEL="${TEACHER_AGENT_MODEL:-}"
export TEACHER_PORT="${TEACHER_PORT:-28000}"
export RAY_PORT="${RAY_PORT:-6379}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
if [[ "$LOCAL_SERVICES" == 1 ]]; then
  teacher_host=127.0.0.1
  ray_host=127.0.0.1
  if [[ "$NODE_COUNT" == 2 ]]; then
    teacher_host="$WORKER_IP"
    ray_host="$HEAD_IP"
  fi
  domains=(MATH IF CODE SCIENCE AGENT)
  for i in "${!domains[@]}"; do
    export "TEACHER_${domains[$i]}_URL=http://${teacher_host}:$((TEACHER_PORT+i))/generate"
  done
  export RAY_JOB_ADDRESS="http://${ray_host}:${RAY_DASHBOARD_PORT}"
fi

# Student parallelism and batch size by node count.
case "$NODE_COUNT" in
  1) TP=1; CP=1; ACTOR_GPUS=3; NODE_GPUS=3; BATCH=510; PLACEMENT=(--colocate); PARALLEL=() ;;
  2) TP=2; CP=2; ACTOR_GPUS=8; NODE_GPUS=8; BATCH=512; PLACEMENT=(); PARALLEL=(--sequence-parallel) ;;
  *) echo "NODE_COUNT must be 1 or 2" >&2; exit 2 ;;
esac

# Student rollout uses 3 GPUs in both layouts.
ARGS=(
  --swiglu
  --num-layers 36
  --hidden-size 2560
  --ffn-hidden-size 9728
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
  --seq-length 40960
  --max-position-embeddings 40960
  --hf-checkpoint "${STUDENT_HF}"
  --ref-load "${STUDENT_CHECKPOINT}"
  --ref-ckpt-step "${STUDENT_STEP}"
  --start-rollout-id 0
  --save "${OUTPUT_DIR}/checkpoints"
  --save-interval 5
  --prompt-data "${PROMPT_DATA}"
  --input-key messages
  --metadata-key metadata
  --tool-key tools
  --data-source-path slime.rollout.data_source.RolloutDataSource
  --apply-chat-template
  --num-rollout 50
  --rollout-batch-size "$BATCH"
  --n-samples-per-prompt 1
  --rollout-shuffle
  --rollout-seed 42
  --rollout-max-prompt-len 8192
  --rollout-max-response-len 32768
  --rollout-max-context-len 40960
  --rollout-temperature 1.0
  --rollout-top-p 1.0
  --rollout-top-k -1
  --global-batch-size "$BATCH"
  --balance-data
  --custom-rm-path opd_adapter.reward_func
  --custom-reward-post-process-path opd_adapter.post_process_rewards
  --rm-url "${TEACHER_MATH_URL}"
  --opd-teacher-metadata-key teacher
  --opd-teacher-url "math=${TEACHER_MATH_URL}"
  --opd-teacher-url "if=${TEACHER_IF_URL}"
  --opd-teacher-url "code=${TEACHER_CODE_URL}"
  --opd-teacher-url "science=${TEACHER_SCIENCE_URL}"
  --opd-teacher-url "agent=${TEACHER_AGENT_URL}"
  --advantage-estimator grpo
  --disable-rewards-normalization
  --disable-grpo-std-normalization
  --use-opd
  --opd-type sglang
  --opd-kl-coef 1.0
  --opd-advantage-clip 5.0
  --opd-response-mask-mode repetition
  --balmopd-group-by domain
  --opd-domain-gradient-mode none
  --kl-loss-coef 0.0
  --entropy-coef 0.0
  --balmopd-beta 0.0
  --optimizer adam
  --lr 3e-6
  --lr-decay-style constant
  --lr-warmup-init 3e-6
  --lr-warmup-iters 0
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.95
  --tensor-model-parallel-size "$TP"
  --context-parallel-size "$CP"
  --pipeline-model-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  "${PARALLEL[@]}"
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu 20480
  --update-weight-buffer-size 10737418240
  --rollout-num-gpus-per-engine 1
  --sglang-mem-fraction-static 0.76
  --sglang-chunked-prefill-size 4096
  --sglang-max-prefill-tokens 4096
  --sglang-max-running-requests 32
  --actor-num-nodes 1
  --actor-num-gpus-per-node "$ACTOR_GPUS"
  "${PLACEMENT[@]}"
  --rollout-num-gpus 3
  --num-gpus-per-node "$NODE_GPUS"
  --seed 1234
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend fused
)

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "$LOCAL_SERVICES" == 1 ]]; then
  source "$ROOT/scripts/distillation/services.sh"
  start_services "$@"
  if [[ "${1:-}" == --worker ]]; then exit 0; fi
fi
"${PYTHON:-python3}" "$ROOT/scripts/distillation/run.py" "$@" -- "${ARGS[@]}"
