from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .checkpoint import inspect_checkpoints
from .config import load_config
from .continual import (
    compile_continual_plan,
    load_continual_config,
    run_continual,
)
from .errors import MM8Error
from .engine import run as run_method
from .plan import compile_plan
from .registry import registry_payload

def _write(payload: Dict[str, Any], report: Optional[Path]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if report is None:
        sys.stdout.write(text)
    else:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(text, encoding="utf-8")
        print(str(report))

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fusioneval",
        description="Validate sources and run checkpoint combination methods.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("methods", help="show the method registry")
    for name, help_text in (
        ("plan", "validate config and compile a no-I/O execution plan"),
        ("inspect", "inspect checkpoint headers and metadata without writing weights"),
        ("preflight", "compile the plan and inspect every source checkpoint"),
    ):
        child = subparsers.add_parser(name, help=help_text)
        child.add_argument("--config", required=True, type=Path)
        child.add_argument("--report", type=Path)
    run = subparsers.add_parser("run", help="unified model-construction entry point")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--report", type=Path)
    run.add_argument("--dry-run", action="store_true", help="compile and inspect without writing weights")
    continual = subparsers.add_parser(
        "continual", help="run a staged sequence of continual fusion updates"
    )
    continual.add_argument("--config", required=True, type=Path)
    continual.add_argument("--report", type=Path)
    continual_mode = continual.add_mutually_exclusive_group()
    continual_mode.add_argument(
        "--dry-run", action="store_true", help="compile the full sequence without checkpoint I/O"
    )
    continual_mode.add_argument(
        "--resume", action="store_true", help="resume from the completed stage prefix"
    )
    continual.add_argument(
        "--stop-after",
        type=int,
        metavar="STAGE",
        help="stop cleanly after this stage; continue later with --resume",
    )
    return parser

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "methods":
            _write({"methods": registry_payload()}, None)
            return 0
        if args.command == "continual":
            config = load_continual_config(args.config)
            if args.dry_run and args.stop_after is not None:
                raise MM8Error("--stop-after cannot be combined with --dry-run")
            payload = (
                compile_continual_plan(config)
                if args.dry_run
                else run_continual(
                    config, resume=args.resume, stop_after=args.stop_after
                )
            )
            _write(payload, args.report)
            return 0
        config = load_config(args.config)
        if args.command == "plan":
            payload = compile_plan(config)
        elif args.command == "inspect":
            payload = inspect_checkpoints(config)
        elif args.command == "preflight" or (args.command == "run" and args.dry_run):
            payload = {
                "status": "PREFLIGHT_PASS",
                "plan": compile_plan(config),
                "inspection": inspect_checkpoints(config),
                "weights_written": False,
            }
        else:
            payload = run_method(config)
        _write(payload, args.report)
        return 0
    except MM8Error as exc:
        print("fusioneval: {}".format(exc), file=sys.stderr)
        return 2
