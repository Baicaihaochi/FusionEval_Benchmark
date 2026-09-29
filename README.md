# FusionEval

Models: [huggingface.co/David6995](https://huggingface.co/David6995)

Model fusion code and experiment configurations for Qwen3-4B and Qwen3-8B.
All methods share one Bash entry point; checkpoints, datasets, SFT and evaluation pipelines are not included.

## 🎆 Quick start

Use Python 3.10+ and install PyTorch for your CUDA environment. Run these commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

bash run.sh --help
bash run.sh 4b ties --plan
```

`--plan` previews a fusion configuration without loading models or using a GPU. After providing the checkpoints below, run:

```bash
bash run.sh 4b ties --dry-run  # Inspect checkpoint compatibility
bash run.sh 4b ties           # Fuse the experts
```

## Qwen3-4B Task Arithmetic

```bash
# Download missing experts from David6995; use the exact common SFT base.
python scripts/run_qwen3_4b_ta.py --base-model /path/to/common-base

# Or download the common base from its published HF repository.
python scripts/run_qwen3_4b_ta.py --base-repo OWNER/COMMON-SFT-BASE

# Fully local inputs (no Hugging Face calls).
python scripts/run_qwen3_4b_ta.py --base-model /path/to/common-base \
  --math-model /path/to/math --code-model /path/to/code \
  --agent-model /path/to/agent --if-model /path/to/if \
  --science-model /path/to/science --output outputs/qwen3-4b/ta-local
```

Each omitted expert path downloads `David6995/Qwen3-4B-expert-{Math,Code,Agent,IF,Science}`.
Local and downloaded experts can be mixed. Invalid local paths fail without a download fallback.
The common SFT base must be supplied with `--base-model` or `--base-repo`;
the original `Qwen/Qwen3-4B-Base` is not assumed to be the same checkpoint.
No public repository for the common SFT base has been verified yet.
Use `--plan` to inspect sources without downloads or fusion, `--cache-dir` to choose the download cache,
and `--local-files-only` to use cached downloads offline. Downloads respect your configured proxy environment.
`--base-revision` and `--expert-revision` select HF revisions (default: `main`).

TA uses the existing engine: `base + 0.3 * sum(expert - base)`, in the configured
Math, Code, Agent, IF, Science order, with BF16 output. Change `--scale` if needed.
The default device is CUDA; `--device cpu` is also supported. No calibration data is needed.
The engine checks tensor compatibility and saves a Hugging Face checkpoint under
`outputs/qwen3-4b/ta`; existing output directories are rejected.
Allow disk space for six source checkpoints and the fused model.

## 🐣 Inputs

Place Hugging Face checkpoints in the following directories, or edit `configs/qwen3-{4b,8b}/common.yaml`. Paths in YAML files resolve relative to the configuration file.

```text
models/qwen3-4b/{base,math,code,agent,if,science}/
models/qwen3-8b/{base,math,code,agent,if,science}/
data/{math,code,agent,if,science}.jsonl
```

Data-assisted methods use 128 calibration examples per domain; weight-only methods do not require data. JSONL accepts `messages` with optional `tools`, or tokenized `input_ids` with optional `labels`:

```json
{"messages": [{"role": "user", "content": "Compute 2 + 3."}]}
```

## ✨ Model fusion

Choose `4b` or `8b` and a method from the table. Outputs are saved under `outputs/`; existing checkpoints are not overwritten.

| Method | Command name |
|---|---|
| Soup / Task Arithmetic | `soup` / `ta` |
| TIES / DARE / DELLA | `ties` / `dare` / `della` |
| Dataless Localize-and-Stitch | `localize_stitch` |
| Fisher / AdaMerging / RegMean | `fisher` / `adamerging` / `regmean` |

```bash
bash run.sh 4b soup
bash run.sh 8b regmean
```

These runs use one process on the visible CUDA device. Selected parameters and input-length details are in [Configurations](docs/CONFIGURATIONS.md).

## 🪄 Post-fusion calibration

FeatCal and Surgery initialize from TA; `calibration_common.yaml` selects that checkpoint. Surgery produces domain-specific adapters, applied through `surgery_context` in `fusioneval.methods.surgery.runtime`.

```bash
bash run.sh 4b ta
bash run.sh 4b featcal
bash run.sh 4b surgery
```

## 🌱 Continual fusion

The Qwen3-4B sequence starts from IF and adds Math, Science, Code, and Agent. Available methods are `soup`, `ta`, `ties`, `della`, and `regmean`.

```bash
bash run.sh continual soup --plan
bash run.sh continual soup
bash run.sh continual ties --stop-after 2
bash run.sh continual ties --resume
```

## 🤗 Distillation

MOPD and Seq-KD use an external Slime/Megatron environment and an existing Ray cluster. Edit the environment template for your paths, then launch through the same entry point:

```bash
source configs/distillation/env.example.sh
bash run.sh 4b seqkd --dry-run
bash run.sh 4b seqkd --ray-address "$RAY_JOB_ADDRESS"
PROMPT_DATA="$PWD/data/mopd.jsonl" OUTPUT_DIR="$PWD/outputs/mopd_4b" \
  bash run.sh 4b mopd --ray-address "$RAY_JOB_ADDRESS"
```

See [Distillation](scripts/distillation/README.md) for teacher endpoints, data format and resume. This package includes only configuration and small adapters, not Slime itself.
