#!/usr/bin/env bash
# Sourced by MOPD recipes; cleans up only processes started here.
start_services() {
  local mode="${1:-}"
  local option
  for option in "$@"; do
    if [[ "$option" == --plan || "$option" == --dry-run ]]; then
      echo "# ${NODE_COUNT} node(s): student rollout GPUs 0-2; teachers GPUs 3-7."
      return
    fi
  done
  : "${OUTPUT_DIR:?Set OUTPUT_DIR in the recipe}"
  if [[ "$NODE_COUNT" == 2 ]]; then
    : "${HEAD_IP:?Set HEAD_IP for two nodes}" "${WORKER_IP:?Set WORKER_IP for two nodes}"
  elif [[ "$mode" == --worker ]]; then
    echo '--worker requires NODE_COUNT=2' >&2; return 2
  fi
  export NO_PROXY="127.0.0.1,localhost,${HEAD_IP},${WORKER_IP}${NO_PROXY:+,$NO_PROXY}"
  export no_proxy="$NO_PROXY"
  unset RAY_ADDRESS
  if [[ "$mode" != --worker ]]; then
    : "${MEGATRON_ROOT:?Set MEGATRON_ROOT}" "${STUDENT_HF:?Set STUDENT_HF}" "${PROMPT_DATA:?Set PROMPT_DATA}"
    if [[ "$mode" != --resume ]]; then
      : "${STUDENT_CHECKPOINT:?Set STUDENT_CHECKPOINT}" "${STUDENT_STEP:?Set STUDENT_STEP}"
    fi
  fi
  local ports=()
  if [[ "$mode" != --worker ]]; then ports+=("$RAY_PORT" "$RAY_DASHBOARD_PORT"); fi
  if [[ "$NODE_COUNT" == 1 || "$mode" == --worker ]]; then
    : "${TEACHER_MATH_MODEL:?Set TEACHER_MATH_MODEL}" "${TEACHER_IF_MODEL:?Set TEACHER_IF_MODEL}"
    : "${TEACHER_CODE_MODEL:?Set TEACHER_CODE_MODEL}" "${TEACHER_SCIENCE_MODEL:?Set TEACHER_SCIENCE_MODEL}" "${TEACHER_AGENT_MODEL:?Set TEACHER_AGENT_MODEL}"
    ports+=("$TEACHER_PORT" "$((TEACHER_PORT+1))" "$((TEACHER_PORT+2))" "$((TEACHER_PORT+3))" "$((TEACHER_PORT+4))")
  fi
  # Do not accidentally connect to someone else's services on these ports.
  "${PYTHON:-python3}" - "${ports[@]}" <<'PYPORT'
import socket, sys
for port in sys.argv[1:]:
    with socket.socket() as sock:
        sock.bind(('', int(port)))
PYPORT
  SERVICE_PIDS=()
  local logs="$OUTPUT_DIR/services"
  mkdir -p "$logs"
  cleanup_services() {
    trap - EXIT INT TERM
    "${PYTHON:-python3}" "$ROOT/scripts/distillation/stop_services.py" "${SERVICE_PIDS[@]}"
  }
  trap cleanup_services EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM

  wait_http() {
    local url="$1" deadline=$((SECONDS + ${SERVICE_TIMEOUT:-900})) pid
    until curl --noproxy '*' --silent --fail --max-time 2 "$url" >/dev/null; do
      for pid in "${SERVICE_PIDS[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
          echo "Service exited; inspect $logs" >&2; return 1
        fi
      done
      if (( SECONDS >= deadline )); then
        echo "Service not ready: $url; inspect $logs" >&2; return 1
      fi
      sleep 2
    done
    for pid in "${SERVICE_PIDS[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then echo "Service exited; inspect $logs" >&2; return 1; fi
    done
  }

  local host=127.0.0.1 bind=127.0.0.1 ray_gpus=0,1,2 ray_count=3
  local ray_args=(--head --port="$RAY_PORT" --include-dashboard=true --dashboard-port="$RAY_DASHBOARD_PORT")
  if [[ "$NODE_COUNT" == 2 ]]; then
    bind=0.0.0.0
    host="$HEAD_IP"
    ray_gpus=0,1,2,3,4,5,6,7
    ray_count=8
    if [[ "$mode" == --worker ]]; then
      wait_http "$RAY_JOB_ADDRESS/api/version"
      host="$WORKER_IP"
      ray_gpus=0,1,2
      ray_count=3
      ray_args=(--address="$HEAD_IP:$RAY_PORT")
    fi
  fi
  if [[ "$mode" != --worker ]]; then
    ray_args+=(--dashboard-host="$bind" --temp-dir="$(mktemp -d /tmp/fe-ray-XXXXXX)")
  fi
  CUDA_VISIBLE_DEVICES="$ray_gpus" ray start "${ray_args[@]}" --node-ip-address="$host" \
    --num-gpus="$ray_count" --disable-usage-stats --block >"$logs/ray-${mode:---head}.log" 2>&1 &
  SERVICE_PIDS+=("$!")

  if [[ "$NODE_COUNT" == 1 || "$mode" == --worker ]]; then
    local models=("$TEACHER_MATH_MODEL" "$TEACHER_IF_MODEL" "$TEACHER_CODE_MODEL" "$TEACHER_SCIENCE_MODEL" "$TEACHER_AGENT_MODEL")
    local domains=(math if code science agent) i
    for i in 0 1 2 3 4; do
      if [[ -z "${models[$i]}" ]]; then echo "Set teacher model path for ${domains[$i]}" >&2; return 2; fi
      CUDA_VISIBLE_DEVICES=$((i+3)) "${PYTHON:-python3}" -m sglang.launch_server \
        --model-path "${models[$i]}" --host "$bind" --port "$((TEACHER_PORT+i))" \
        --tensor-parallel-size 1 --context-length 40960 --dtype bfloat16 \
        --mem-fraction-static 0.82 --chunked-prefill-size 4096 --max-running-requests 8 \
        >"$logs/teacher-${domains[$i]}.log" 2>&1 &
      SERVICE_PIDS+=("$!")
    done
  fi
  if [[ "$mode" == --worker ]]; then
    echo "Worker services running; logs: $logs. Stop with Ctrl-C after training."
    wait -n "${SERVICE_PIDS[@]}"
    exit 1
  fi
  wait_http "$RAY_JOB_ADDRESS/api/version"
  local url
  for url in "$TEACHER_MATH_URL" "$TEACHER_IF_URL" "$TEACHER_CODE_URL" "$TEACHER_SCIENCE_URL" "$TEACHER_AGENT_URL"; do
    wait_http "${url%/generate}/health"
  done
}
