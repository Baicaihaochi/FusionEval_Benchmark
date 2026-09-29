#!/usr/bin/env python3
"""Resolve five Qwen3-4B experts and run the repository's Task Arithmetic engine."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("Math", "Code", "Agent", "IF", "Science")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    base = p.add_mutually_exclusive_group(required=True)
    base.add_argument("--base-model", type=Path, help="Local common SFT base checkpoint")
    base.add_argument("--base-repo", help="HF repo of the exact common SFT base (not an arbitrary Qwen base)")
    p.add_argument("--base-revision", default="main")
    for domain in DOMAINS:
        p.add_argument("--" + domain.lower() + "-model", type=Path,
                       help=f"Local {domain} checkpoint; otherwise download from HF")
    p.add_argument("--hf-namespace", default="David6995")
    p.add_argument("--expert-revision", default="main")
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/qwen3-4b/ta")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--scale", type=float, default=0.3)
    p.add_argument("--plan", action="store_true", help="Show sources only; no downloads, model reads or fusion")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if not math.isfinite(args.scale):
        p.error("--scale must be finite")
    sources = [{"id": "base", "path": args.base_model, "repo": args.base_repo,
                "revision": args.base_revision}]
    sources += [{"id": d.lower(), "path": getattr(args, d.lower() + "_model"),
                 "repo": f"{args.hf_namespace}/Qwen3-4B-expert-{d}",
                 "revision": args.expert_revision} for d in DOMAINS]
    output = args.output.expanduser().resolve()
    for item in sources:
        if item["path"] is not None:
            item["path"] = str(item["path"].expanduser().resolve())
            item["repo"] = None
            item["revision"] = None
    if args.plan:
        print(json.dumps({"sources": sources, "output": str(output), "scale": args.scale,
                          "device": args.device, "downloads": False, "weights_written": False}, indent=2))
        return 0
    if output.exists():
        p.error(f"output already exists: {output}")
    # Validate all explicit inputs before downloading anything. A typo never falls back to HF.
    for item in sources:
        if item["path"]:
            path = Path(item["path"])
            if not (path / "config.json").is_file() or not any(path.glob("*.safetensors")):
                p.error(f"invalid local checkpoint for {item['id']}: {path}")
            if path == output or path in output.parents or output in path.parents:
                p.error("output must not overlap an input checkpoint")
    # Dependency check precedes large downloads; the engine performs full compatibility checks.
    import yaml
    import torch  # noqa: F401
    import safetensors  # noqa: F401
    import transformers  # noqa: F401
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA is unavailable; select --device cpu or use a CUDA environment")
    for item in sources:
        if item["path"] is None:
            from huggingface_hub import snapshot_download
            item["path"] = snapshot_download(
                repo_id=item["repo"], revision=item["revision"],
                cache_dir=str(args.cache_dir.expanduser()) if args.cache_dir else None,
                local_files_only=args.local_files_only,
                allow_patterns=["*.safetensors", "*.json", "*.model", "*.tiktoken", "*.jinja", "merges.txt", "vocab.txt"],
            )
    common = {"schema_version": 1, "model_profile": str(ROOT / "configs/models/qwen3-4b.yaml"),
              "base": sources[0]["path"],
              "experts": [{"id": x["id"], "path": x["path"]} for x in sources[1:]],
              "output_root": str(output.parent),
              "runtime": {"device": args.device, "precision": "mergebench",
                          "save_dtype": "bfloat16", "seed": 42, "max_shard_size_gib": 5}}
    run = {"schema_version": 1, "common": "common.yaml", "method": "task_arithmetic",
           "parameters": {"scale": args.scale}, "output": output.name}
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    with tempfile.TemporaryDirectory(prefix="fusioneval-ta-") as tmp:
        config = Path(tmp) / "ta.yaml"
        (Path(tmp) / "common.yaml").write_text(yaml.safe_dump(common), encoding="utf-8")
        config.write_text(yaml.safe_dump(run), encoding="utf-8")
        subprocess.run([sys.executable, "-m", "fusioneval", "run", "--config", str(config)],
                       env=env, check=True)
    (output / "download_sources.json").write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
