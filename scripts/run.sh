#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 || "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'HELP'
Usage: bash run.sh {4b|8b|continual} METHOD [OPTIONS]

Fusion:       soup ta ties dare della localize_stitch fisher adamerging regmean
Calibration:  featcal surgery
Distillation: mopd seqkd
Continual:    soup ta ties della regmean

Examples:
  bash run.sh 4b ties --plan
  bash run.sh 8b regmean
  bash run.sh continual ties --resume
  bash run.sh 4b seqkd --dry-run
HELP
  exit 0
fi
if [[ $# -lt 2 ]]; then
  echo 'Provide a model group and method; see bash run.sh --help' >&2
  exit 2
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
size="$1"
method="$2"
shift 2
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"

case "$size" in
  4b|8b) group="qwen3-$size"; command=run ;;
  continual) group=continual; command=continual ;;
  *) echo 'Choose 4b, 8b, or continual' >&2; exit 2 ;;
esac

case "$method" in
  mopd|seqkd)
    if [[ "$size" == continual ]]; then
      echo 'Distillation uses 4b or 8b' >&2
      exit 2
    fi
    exec "${PYTHON:-python3}" "$root/scripts/distillation/run.py" \
      --config "$root/configs/distillation/${method}_${size}.json" "$@"
    ;;
esac

if [[ "${1:-}" == --plan ]]; then
  shift
  if [[ "$size" == continual ]]; then
    set -- --dry-run "$@"
  else
    command=plan
  fi
fi
exec "${PYTHON:-python3}" -m fusioneval "$command" \
  --config "$root/configs/$group/$method.yaml" "$@"
