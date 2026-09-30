from __future__ import annotations

import argparse
import json
import sys
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .checkpoint import inspect_checkpoints
from .assets import missing_hint
from .config import load_config, _read_yaml
from .continual import (
    compile_continual_plan,
    load_continual_config,
    run_continual,
)
from .errors import MM8Error
from .engine import run as run_method
from .plan import compile_plan
from .registry import registry_payload
from .run_logging import run_log, resolved_config

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
    subparsers.add_parser("launch", help="run the workflow selected by a single config")
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

def _launch(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="Run one self-contained experiment config")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--plan", action="store_true")
    options, remaining = parser.parse_known_args(argv[1:])
    if options.config.suffix == ".sh":
        command = ["bash", str(options.config), *remaining]
        if options.plan:
            command.append("--plan")
        return subprocess.call(command)
    raw = _read_yaml(options.config, "experiment config")
    workflow = raw.get("workflow", "fusion")
    if workflow not in {"fusion", "continual"}:
        raise MM8Error("workflow must be fusion or continual; use a .sh recipe for distillation")
    command = "continual" if workflow == "continual" else ("plan" if options.plan else "run")
    if workflow == "continual" and options.plan:
        remaining.append("--dry-run")
    return main([command, "--config", str(options.config), *remaining])


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if argv and argv[0] == "launch":
            return _launch(argv)
        args = _parser().parse_args(argv)
        if args.command == "methods":
            _write({"methods": registry_payload()}, None)
            return 0
        if args.command == "continual":
            if args.resume and not _read_yaml(args.config, "continual config").get("output"):
                raise MM8Error("For --resume, set output to the existing run's complete path")
            config = load_continual_config(args.config)
            if args.dry_run and args.stop_after is not None:
                raise MM8Error("--stop-after cannot be combined with --dry-run")
            if args.dry_run:
                payload = compile_continual_plan(config)
            else:
                with run_log(config.run_root, config.log, resume=args.resume, snapshot=resolved_config(config), device=config.runtime['device']) as metrics:
                    payload = run_continual(config, resume=args.resume, stop_after=args.stop_after)
                    metrics['algorithm_runtime_seconds'] = payload['execution_runtime_seconds']
                    _write(payload, args.report)
            if args.dry_run:
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
            with run_log(config.output, config.log, snapshot=resolved_config(config), device=config.runtime['device']) as metrics:
                payload = run_method(config)
                metrics['algorithm_runtime_seconds'] = payload['timing']['algorithm_runtime_seconds']
                print("Method: {}\nParameters: {}".format(config.method.name, json.dumps(config.method_parameters)), flush=True)
                _write(payload, args.report)
            return 0
        _write(payload, args.report)
        return 0
    except FileNotFoundError as exc:
        print("fusioneval: {}. {}".format(exc, missing_hint(exc.filename)) if exc.filename else str(exc), file=sys.stderr)
        return 2
    except MM8Error as exc:
        print("fusioneval: {}".format(exc), file=sys.stderr)
        return 2
