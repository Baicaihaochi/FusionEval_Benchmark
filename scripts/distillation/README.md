# 🤗 Distillation

MOPD and Seq-KD configurations target an external Slime/Megatron installation with the OPD extensions and `sft_loss` support. Use its CUDA, Ray and SGLang environment; the root `requirements.txt` covers parameter fusion only.

## 🎆 Launch

Edit [env.example.sh](../../configs/distillation/env.example.sh), then source it from the repository root. Paths, Python environments and this repository must be accessible on all Ray nodes.

```bash
source configs/distillation/env.example.sh
bash run.sh 4b seqkd --dry-run
bash run.sh 4b seqkd --ray-address "$RAY_JOB_ADDRESS"
```

For 8B, select `8b` and update the Student, input and output paths accordingly. The scripts use an existing Ray cluster and do not allocate resources or start services.

## 🌟 Settings

| Setting | 4B | 8B |
|---|---|---|
| Tensor parallelism / context parallelism | 2 / 2 | 4 / 2 |
| Batch size / learning rate | 512 / 3e-6 | 512 / 3e-6 |
| Training updates | 50 | 50 |
| Save interval | 5 | 5 |

Adam uses betas (0.9, 0.95), weight decay 0.1 and no warmup. Optimizer state is saved; Slime iteration 49 corresponds to 50 updates.

## 🍀 Seq-KD data

Supply the fixed 25,600-row sequence, with responses from the corresponding model-size domain teachers. Tokenize with the matching template, retain the first 4096 tokens, and record the remaining response length:

```json
{"metadata":{"seqkd_token_ids":[1,2,3],"seqkd_response_length":1}}
```

Only response tokens receive loss; keep rows with zero response length and preserve their order. The included adapter reads this prepared format; teacher generation and data preparation are external.

## 🌱 Resume

Use the same dataset and a checkpoint root containing saved optimizer state and the Seq-KD cursor. The configured update count remains the total target.

```bash
bash run.sh 8b seqkd --resume "$RESUME_CHECKPOINT" --ray-address "$RAY_JOB_ADDRESS"
```

`--debug-train-only` is Slime's required offline-training switch, not diagnostic logging.

## 😊 MOPD teachers

Provide five SGLang `/generate` endpoints through `TEACHER_{MATH,IF,CODE,SCIENCE,AGENT}_URL`; each prompt's `metadata.teacher` selects its domain. The topology uses 8 actor GPUs colocated with 11 rollout GPUs, plus 5 external teacher GPUs, totaling 16 GPUs.

```bash
export PROMPT_DATA="$PWD/data/mopd.jsonl"
export OUTPUT_DIR="$PWD/outputs/mopd_4b"
bash run.sh 4b mopd --ray-address "$RAY_JOB_ADDRESS"
```

MOPD uses an 8192-token prompt limit and a 32768-token response limit, separately from Seq-KD's prefix preparation.

The external Slime installation must expose `slime.rollout.multi_teacher_on_policy_distillation` and `slime.rollout.on_policy_distillation`; arbitrary upstream versions may not provide these APIs.
