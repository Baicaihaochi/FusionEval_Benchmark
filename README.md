# FusionEval

Benchmarking Model Fusion for Large Language Models

[English](README.md) | [简体中文](README.zh-CN.md) · [Models and datasets](https://huggingface.co/David6995) · [Paper](paper/main.pdf)

FusionEval compares fusion methods by capability retention, runtime, and peak GPU memory, using math, code, tool use, instruction following, and science experts trained from a shared base. The repository provides parameter and continual fusion configurations for Qwen3-4B, Qwen3-8B, and Llama 3.2 3B, plus distillation scripts for Qwen3-4B/8B.

![FusionEval overview and Qwen3-4B performance versus runtime (Figure 1)](assets/figure1.png)

## 👋 Start here

```bash
git clone https://github.com/Baicaihaochi/FusionEval_Benchmark.git
cd FusionEval_Benchmark
```

For a first run, try Qwen3-4B with TA: install the environment → prepare models → run fusion. TA needs no calibration data. If your environment and local models are ready, set their paths in the YAML and run it.

| Model | Config directory | Download option |
|---|---|---|
| Qwen3-4B | [examples/qwen3-4b](examples/qwen3-4b) | `--model qwen3-4b` |
| Qwen3-8B | [examples/qwen3-8b](examples/qwen3-8b) | `--model qwen3-8b` |
| Llama 3.2 3B | [examples/llama3.2-3b](examples/llama3.2-3b) | `--model llama3.2-3b` |

## 🛠️ Installation

Start on a compute node with an available NVIDIA GPU. From the repository root, use [uv](https://docs.astral.sh/uv/getting-started/installation/) to install the environment:

```bash
bash scripts/setup/uv.sh
source .venv/bin/activate
```

The default environment is `.venv/`, with Python 3.11, PyTorch CUDA 12.8 and **Transformers 4.57.1**. Dependencies are in `requirements.txt`. To select another PyTorch backend:

```bash
TORCH_BACKEND=cu126 bash scripts/setup/uv.sh
```

## 📦 Models and data

Download the common base and five experts for the model you want to use:

```bash
python scripts/download/models.py --model qwen3-4b
# Other choices: --model qwen3-8b | --model llama3.2-3b

python scripts/download/data.py  # Calibration: 128 examples per domain
```

Default locations:

```text
models/<model>/{base,math,code,agent,if,science}/
data/calibration/data/{math,code,agent,if,science}.jsonl
```

`<model>` is `qwen3-4b`, `qwen3-8b` or `llama3.2-3b`. Already have the files? Set their paths in the experiment YAML and skip downloading. Experts must share the same base, architecture and tokenizer.

The download scripts accept `--output-dir` and `--plan` (preview only). Use `--include base math code` to select models. Other datasets are available through `--dataset sft`, `--dataset mopd` and `--dataset student`; calibration and SFT also accept `--domains math code`. Downloads use your existing Hugging Face login and network settings.

## 🚀 Run fusion

Each experiment uses one YAML file:

```bash
CUDA_VISIBLE_DEVICES=0 bash run.sh examples/qwen3-4b/ta.yaml
CUDA_VISIBLE_DEVICES=0 bash run.sh examples/qwen3-8b/ta.yaml
CUDA_VISIBLE_DEVICES=0 bash run.sh examples/llama3.2-3b/ta.yaml
```

Choose one command. Each run uses one GPU. You can also set the GPU and config path in `launch.sh`, then run `bash launch.sh`.

Each model directory contains these methods:

| Config | Method | Additional inputs |
|---|---|---|
| `soup.yaml` | Uniform averaging | None |
| `ta.yaml` | Task Arithmetic | None |
| `ties.yaml` | TIES | None |
| `dare.yaml` | DARE | None |
| `della.yaml` | DELLA | None |
| `localize_stitch.yaml` | Localize-and-Stitch | None |
| `fisher.yaml` | Fisher | Calibration data |
| `adamerging.yaml` | AdaMerging | Calibration data |
| `regmean.yaml` | RegMean | Calibration data |
| `featcal.yaml` | FeatCal | Calibration data and an initial fused model |
| `surgery.yaml` | Surgery | Calibration data and an initial fused model |

For FeatCal/Surgery, first run TA and set `initial_model` to its output, or save that result at `models/<model>/initial/`. Surgery produces domain adapters, applied with `surgery_context` in `fusioneval.methods.surgery.runtime`.

Llama configs target the full Hugging Face checkpoints published under David6995. Their parameters are starting values adapted from Qwen; real Llama fusion has not yet been validated.

### Edit the experiment

| Setting | YAML field |
|---|---|
| Base and experts | `base`, `experts[].path` |
| Number of experts | Add or remove entries in `experts` |
| Method and parameters | `method`, `parameters` |
| Data | `experts[].data`, or `calibration_data` for shared data |
| Architecture preset | `model_profile` |
| Compute and save precision | `runtime.precision`, `runtime.save_dtype` |
| Result and log paths | `output`, `log` |

Examples use five experts. Adjust per-expert coefficients when changing their number. To switch methods, use the matching template's `method` and `parameters`.

Null model/data paths select the default folders above. Explicit relative paths resolve from the YAML file's directory; absolute paths work too.

For example, fuse only Math and Code with a scale of 0.3. Copy the TA config:

```bash
cp examples/qwen3-4b/ta.yaml examples/qwen3-4b/my_ta.yaml
```

Replace these fields in `my_ta.yaml`, keeping the rest unchanged:

```yaml
base: null
experts:
  - id: math
    path: null
  - id: code
    path: null
parameters:
  scale: 0.3
output: null
```

```bash
CUDA_VISIBLE_DEVICES=0 bash run.sh examples/qwen3-4b/my_ta.yaml
```

Replace `path: null` with your checkpoint directory to use local models. It should contain `config.json`, tokenizer files and full safetensors weights.

Compute precision is `bfloat16` or `float32`; DELLA and continual DELLA/RegMean compute in FP32. `save_dtype` controls saved weights separately. Calibration accepts JSONL conversations (`messages`) or tokenized `input_ids`; use the selected model's tokenizer and chat template.

Preview a configuration without loading weights:

```bash
bash run.sh examples/qwen3-4b/ta.yaml --plan
```

## 📁 Outputs

With `output: null`, each run creates a separate directory:

```text
outputs/ta/qwen3-4b_m5_s0.3_a7c2e9f1/
```

The name includes the model, expert count, main parameters and a random suffix. Continual results go under `outputs/continual/<method>/`.

To choose the exact destination, set `output` in the YAML:

```yaml
output: ../../outputs/my_ta_model
log: null  # Save run.log inside the result directory
```

Existing models are not overwritten. Set `log` to a `.log` file to save logs separately. The result directory also contains `run_config.yaml`, `run_status.json` and method records. Check the status after a failure: a directory containing logs alone is not a completed model.

The last log line reports runtime and peak GPU memory:

```text
Runtime summary | status=SUCCESS | algorithm=289.80s | wall=318.25s | peak_gpu_memory=9.420 GiB (PyTorch allocated)
```

`algorithm` follows the method record, including preparation, fusion, saving and validation; `wall` covers the full invocation. Peak memory is this process's PyTorch allocation on the selected GPU across the run, excluding other processes and CUDA context overhead. Resume counts algorithm time only for newly executed stages. CPU runs or unavailable memory metrics show `N/A`. Values are also saved in `run_status.json`. These summaries cover fusion and continual fusion; external Slime distillation uses its training logs.

## 🔄 Continual fusion and resume

Each model's `continual/` directory provides Soup, TA, TIES, DELLA and RegMean. The default sequence is IF → Math → Science → Code → Agent. Edit `initial` and `order` to change it.

```bash
CUDA_VISIBLE_DEVICES=0 bash run.sh examples/qwen3-4b/continual/ties.yaml
```

To resume an interrupted run, set `output` to that run's full directory path, keep its inputs and parameters unchanged, then run:

```bash
CUDA_VISIBLE_DEVICES=0 bash run.sh examples/qwen3-4b/continual/ties.yaml --resume
```

Completed stages are reused; unfinished stages are recomputed. An incomplete published stage directory raises an error. `--stop-after 2` stops after two updates.

To add experts to an existing fused model, use it as `initial`, set `initial_experts` to the number it represents, and list only new arrivals in `order`. Use a new output for this run.

Continual methods use the current model and incoming expert. TA, TIES and DELLA also use the fixed base. RegMean recomputes both models' statistics on incoming-domain data; it uses no historical statistics and is generally not equivalent to batch RegMean. DELLA uses an independent seed per update.

## 🧪 Distillation

MOPD and Seq-KD scripts: `examples/distillation/{mopd,seqkd}_{4b,8b}.sh`. In a compatible Megatron/SGLang/CUDA training environment, install the bundled [Slime](slime) from the repository root:

```bash
python -m pip install --no-deps -e ./slime
```

Training needs a separate environment; see `slime/requirements.txt` for Python dependencies. Private container dependencies are not included. Full training with this export remains untested.

Set student checkpoint paths, `STUDENT_STEP`, `PROMPT_DATA`, `OUTPUT_DIR` and `MEGATRON_ROOT` at the top of the script. Edit training parameters in `ARGS`. `SLIME_ROOT` defaults to the bundled source.

### MOPD

Set the five `TEACHER_*_MODEL` paths. Data rows must contain `messages` and `metadata.teacher`: `math`, `if`, `code`, `science` or `agent`. MOPD starts Ray and teachers on one or two allocated 8-GPU nodes:

| `NODE_COUNT` | Head node | Worker node |
|---|---|---|
| `1` (default) | GPUs 0-2: student training/rollout; GPUs 3-7: teachers | None |
| `2` | GPUs 0-7: student training | GPUs 0-2: student rollout; GPUs 3-7: teachers |

```bash
bash examples/distillation/mopd_4b.sh --plan  # Preview only
bash examples/distillation/mopd_4b.sh
```

For two nodes, set `NODE_COUNT=2`, `HEAD_IP` and `WORKER_IP`. Both nodes need the same environment and file paths, and must reach each other. Run the command above on the head; on the worker, run:

```bash
bash examples/distillation/mopd_4b.sh --worker
```

Service logs: `OUTPUT_DIR/services/`. Head services stop after training; stop the worker with Ctrl-C. Edit parallelism, batch size and ports in the script.

For existing Ray and teacher services, set `LOCAL_SERVICES=0`, `RAY_JOB_ADDRESS` and five `TEACHER_*_URL` endpoints ending in `/generate`.

### Seq-KD

Seq-KD requires an existing Ray cluster; set `RAY_JOB_ADDRESS`. The paper uses 25,600 teacher-generated responses. Tokenize each example, keep its first 4096 tokens, and record the retained response length:

```json
{"metadata":{"seqkd_token_ids":[1,2,3],"seqkd_response_length":1}}
```

```bash
bash examples/distillation/seqkd_4b.sh --plan
bash examples/distillation/seqkd_4b.sh
bash examples/distillation/seqkd_4b.sh --resume "$RESUME_CHECKPOINT"
```

Keep row order and rows with no retained response tokens. Resume needs the same data, optimizer state and data cursor. Prepare student weights and teacher responses separately; neither is included in the Student dataset.

## 💡 Common questions

| Issue | What to do |
|---|---|
| Output directory already exists | For a new experiment, use `output: null` or a new path. To resume continual fusion, use the original path with `--resume`. |
| FeatCal/Surgery cannot find the initial model | Run TA for the selected model first, then set `initial_model` to its output. |

Read the [paper PDF](paper/main.pdf).

## 📊 Results

### Qwen3-4B

| Method | AIME24 (%) | AIME25 (%) | LCBv5 (%) | LCBv6 (%) | GPQA-D (%) | IFEval (%) | IFBench (%) | BFCLv3 (%) | Raw Avg. (%) | Avg. Norm. | Min. Norm. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Experts | 73.4 | 67.4 | 44.4 | 41.6 | 51.9 | 86.0 | 38.6 | 66.0 | 58.7 | 1.000 | 1.000 |
| TIES | **69.0** | 64.4 | 41.6 | 35.9 | 36.9 | 65.3 | 36.2 | 30.0 | 47.4 | 0.819 | 0.454 |
| RegMean | 43.5 | 40.2 | 31.5 | 29.4 | 47.1 | 68.9 | 28.7 | 30.4 | 40.0 | 0.690 | 0.460 |
| DARE | 66.9 | 62.5 | 47.5 | **42.6** | 49.0 | 70.9 | 37.0 | 30.6 | 50.9 | 0.890 | 0.464 |
| Ada | 67.5 | 58.8 | 42.3 | 38.7 | 49.6 | 66.6 | 35.0 | 31.0 | 48.7 | 0.848 | 0.470 |
| DELLA | 68.5 | 64.8 | 47.0 | 40.8 | 49.5 | 64.4 | 36.0 | 31.1 | 50.3 | 0.880 | 0.471 |
| TA | 67.5 | **67.1** | 48.2 | 42.2 | **51.4** | 69.4 | 37.8 | 31.1 | 51.8 | **0.908** | 0.472 |
| L&S | 67.1 | 62.5 | **49.3** | 39.5 | 44.6 | 70.3 | **42.3** | 31.5 | 50.9 | 0.894 | 0.478 |
| Soup | 45.4 | 35.6 | 27.8 | 27.5 | 51.3 | 62.9 | 24.8 | 33.8 | 38.6 | 0.663 | 0.512 |
| Fisher | 39.8 | 38.1 | 38.2 | 31.9 | 47.1 | 63.9 | 29.6 | 45.1 | 41.7 | 0.729 | 0.542 |
| **Distillation** | | | | | | | | | | | |
| Student | 49.1 | 47.9 | 33.5 | 31.5 | 30.9 | 64.8 | 27.8 | 63.7 | 43.7 | 0.741 | 0.596 |
| Seq-KD | 47.9 | 43.8 | 36.4 | 31.9 | 34.5 | 67.7 | 37.5 | 63.7 | 45.4 | 0.784 | 0.649 |
| MOPD | 59.6 | 53.3 | 42.8 | 38.9 | 36.7 | **77.5** | 41.4 | **66.1** | **52.1** | 0.899 | **0.708** |

### Qwen3-8B

| Method | AIME24 (%) | AIME25 (%) | LCBv5 (%) | LCBv6 (%) | GPQA-D (%) | IFEval (%) | IFBench (%) | BFCLv3 (%) | Raw Avg. (%) | Avg. Norm. | Min. Norm. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Experts | 74.6 | 67.9 | 47.3 | 43.5 | 52.9 | 84.6 | 42.0 | 67.5 | 60.0 | 1.000 | 1.000 |
| Fisher | 60.4 | 41.0 | 30.6 | 28.8 | 45.2 | 64.0 | 26.8 | 29.8 | 40.8 | 0.677 | 0.442 |
| Ada | 69.0 | 65.8 | 55.9 | 48.5 | 48.2 | 73.9 | 41.4 | 31.3 | 54.3 | 0.928 | 0.463 |
| TA | 69.8 | 68.1 | 56.6 | **52.7** | 55.4 | 74.5 | 39.8 | 31.5 | 56.1 | 0.961 | 0.466 |
| DARE | 71.9 | 67.1 | 57.5 | 48.7 | 55.6 | 74.7 | 35.2 | 31.8 | 55.3 | 0.941 | 0.471 |
| RegMean | 52.5 | 42.1 | 37.1 | 35.7 | 54.0 | 72.3 | 29.0 | 31.8 | 44.3 | 0.746 | 0.471 |
| Soup | 53.1 | 39.8 | 41.2 | 36.1 | 57.2 | 75.0 | 31.8 | 34.1 | 46.0 | 0.778 | 0.505 |
| TIES | 69.0 | 63.5 | 58.4 | 44.5 | 52.3 | 73.7 | 43.2 | 34.8 | 54.9 | 0.940 | 0.515 |
| DELLA | **75.6** | 69.4 | **59.7** | 51.3 | **58.2** | 71.9 | 39.2 | 37.8 | **57.9** | **0.990** | 0.561 |
| L&S | 72.7 | **72.1** | 55.4 | 50.8 | 51.6 | 75.1 | 42.0 | 39.1 | 57.4 | 0.977 | 0.580 |
| **Distillation** | | | | | | | | | | | |
| Student | 52.8 | 51.5 | 39.2 | 36.2 | 31.2 | 67.4 | 30.8 | 63.1 | 46.5 | 0.773 | 0.590 |
| Seq-KD | 45.8 | 55.8 | 37.1 | 34.9 | 44.4 | 68.5 | 40.7 | 65.0 | 49.0 | 0.826 | 0.615 |
| MOPD | 67.9 | 59.8 | 45.9 | 42.6 | 45.5 | **80.3** | **46.1** | **66.9** | 56.9 | 0.955 | **0.859** |
