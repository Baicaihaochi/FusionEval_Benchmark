#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export CUDA_VISIBLE_DEVICES=0
# Models: examples/qwen3-{4b,8b}/ or examples/llama3.2-3b/; continual configs are in each model's continual/.
# Fusion: soup, ta, ties, dare, della, localize_stitch, fisher, adamerging, regmean, featcal, surgery.
# Continual: soup, ta, ties, della, regmean. Distillation: examples/distillation/{mopd,seqkd}_{4b,8b}.sh
CONFIG=examples/qwen3-4b/ta.yaml
bash run.sh "$CONFIG" "$@"
