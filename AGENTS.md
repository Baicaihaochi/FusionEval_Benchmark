# Runtime conventions

- Assume the user is already on allocated compute nodes with Python dependencies and GPUs available.
- Keep launch.sh short: GPU selection and a config path. Each experiment config owns its workflow, method, inputs, data, parameters and outputs; users edit one file. Do not reintroduce MODEL/METHOD routing.
- Do not add Slurm, resource allocation, reservation, site-specific node names, container startup or package installation to benchmark launchers.
- Model inputs are local, prepared by the user. Do not add automatic downloads, retry/mirror selection or proxy management. Explicitly invoked download helpers belong in scripts/download; never call them from a workload launcher.
- Leave user-specific distillation paths and endpoints empty in each distillation .sh recipe. Slime/Megatron and student checkpoints remain explicit user prerequisites; do not substitute base checkpoints for students.
- Keep checkpoint validation and error handling in Python; avoid duplicating them in Bash. Preview commands must not download or fuse weights.
- Use launch.sh → root run.sh as the single public launch chain. Do not add per-model/per-method wrapper scripts or machine preflight checks. Missing-input checks should only identify the expected path and show the manual download command. Keep numerical compatibility checks in the fusion engine and output overwrite protection. Let the underlying training stack report environment/input errors.

- Explicit environment installation belongs in scripts/setup/uv.sh. Keep direct dependencies in requirements.txt; do not install packages from launchers or probe networks/GPUs in setup. Slime training remains a separately prepared compatible runtime.

- Automatic fusion outputs use outputs/<method>/<model>_m<count>_<key-parameters>_<random-id>. Preserve explicitly configured destinations and output overwrite protection. An explicit output is a complete destination path and takes precedence over output_root. Continual resume requires an explicit existing run path; never guess the latest run.

- Log fusion runs to output/run.log by default; log may specify an exact external file. Preserve logs on failure without treating diagnostic-only output folders as completed models.

- MOPD examples may start Ray and five teachers on already allocated nodes. NODE_COUNT=1 uses student GPUs 0-2 and teacher GPUs 3-7. NODE_COUNT=2 uses 8 training GPUs on the head and 3 student rollout plus 5 teacher GPUs on the worker; users launch --worker themselves. Clean up only processes started by the invocation; never use global pkill or ray stop. LOCAL_SERVICES=0 uses pre-existing services.
