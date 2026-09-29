# Edit these paths and checkpoint iteration before launching.
export SLIME_ROOT="$PWD/external/slime"
export MEGATRON_ROOT="$PWD/external/Megatron-LM"
export STUDENT_HF="$PWD/models/qwen3-4b/student"
export STUDENT_CHECKPOINT="$PWD/models/qwen3-4b/student_megatron"
export STUDENT_STEP=0
export PROMPT_DATA="$PWD/data/seqkd_4b.jsonl"
export OUTPUT_DIR="$PWD/outputs/seqkd_4b"
export RAY_JOB_ADDRESS="http://localhost:8265"

export TEACHER_MATH_URL="http://localhost:30000/generate"
export TEACHER_IF_URL="http://localhost:30001/generate"
export TEACHER_CODE_URL="http://localhost:30002/generate"
export TEACHER_SCIENCE_URL="http://localhost:30003/generate"
export TEACHER_AGENT_URL="http://localhost:30004/generate"
