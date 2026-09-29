from __future__ import annotations

from typing import Any, Dict

from . import __version__
from .config import RunConfig
from .methods.base import MethodPlugin

def _contract(method: MethodPlugin, parameters: Dict[str, Any]):
    return method.effective_contract(parameters)

def compile_plan(config: RunConfig) -> Dict[str, Any]:
    runtime = config.runtime
    method = config.method
    effective_precision = (
        "float32" if method.compute_precision == "float32" else runtime["precision"]
    )
    parameters = dict(config.method_parameters)
    if method.stochastic:
        parameters.setdefault("seed", runtime["seed"])
    edge_rule = (
        "one canonical storage unit with input_embedding and output_head roles when tied; "
        "two independent units when untied"
    )
    configured_scope = (
        "all_aligned_matrix_floating_with_base_for_non_matrix"
        if method.parameter_scope == "matrix_only_base_for_1d"
        else "all_aligned_trainable_floating"
    )
    contract = _contract(method, parameters)
    plan = {
        "schema_version": 1,
        "implementation_version": __version__,
        "run_id": config.run_id,
        "phase": "implemented",
        "method": {
            "local_name": method.local_name,
            "op_code": method.op_code,
            "contract_id": contract["id"],
            "contract_variant": contract["variant"],
            "contract_tags": contract["tags"],
            "status": method.status,
            "parameters": parameters,
            "aggregation": method.aggregation,
        },
        "sources": {
            "base": str(config.base),
            "experts": [
                {
                    "id": item.id,
                    "path": str(item.path),
                    **({"data": str(item.data)} if item.data is not None else {}),
                }
                for item in config.experts
            ],
            "expert_count": len(config.experts),
            **(
                {"initial_model": str(config.initial_model)}
                if config.initial_model is not None
                else {}
            ),
            **(
                {"calibration_data": str(config.calibration_data)}
                if config.calibration_data is not None
                else {}
            ),
        },
        "output": str(config.output),
        "model_profile": config.model_profile.name if config.model_profile else "auto",
        "config": str(config.path),
        "common_config": str(config.common_path),
        "parameter_contract": {
            "scope": method.parameter_scope,
            "configured_scope": configured_scope,
            "alias_policy": "shared_storage_once",
            "edge_rule": edge_rule,
            "non_floating": "require byte-identical values, then copy base",
        },
        "numeric_contract": {
            "requested_arithmetic_dtype": runtime["precision"],
            "arithmetic_dtype": effective_precision,
            "save_dtype": runtime["save_dtype"],
            "randomness": (
                "seeded torch stream in canonical key order" if method.stochastic else "deterministic"
            ),
        },
        "execution": {
            "executor": method.execution,
            "layout": method.requirements.layout,
            "prepare_scope": method.requirements.prepare_scope,
            "passes": method.requirements.passes,
            "replayable": method.requirements.replayable,
            "input_mode": method.input_mode,
            "device": runtime["device"],
            "staging_parent": str(config.output.parent),
            "output_commit": "staging_then_validate_then_atomic_rename",
        },
        "required_validation": [
            "source schema/config/tokenizer equality",
            "canonical storage-unit coverage",
            "all values finite",
            "saved schema and dtype",
            "reload",
            "tied-or-untied edge invariant",
            "fixed-prefix logits",
            "method-specific statistics",
        ],
    }
    if method.requires_data:
        plan["execution"]["data_contract"] = (
            "one domain JSONL source per expert; tokenized with the shared base template"
            if method.data_mode == "per_expert"
            else "one shared task-agnostic calibration JSONL; tokenized with the base tokenizer"
        )
    if method.requires_initial_model:
        plan["execution"]["initial_model_contract"] = (
            "explicit prebuilt checkpoint; never reconstructed by another method plugin"
        )
    if method.local_name in {"ties", "dare"} and parameters.get("exclude_edge"):
        leftover_edge = parameters.get("leftover_edge", "base")
        plan["parameter_contract"]["edge_policy"] = (
            "exclude_then_task_arithmetic_0.3" if leftover_edge == "ta" else "exclude_then_" + leftover_edge
        )
        if parameters.get("merge_func"):
            plan["parameter_contract"]["ties_merge_func"] = parameters["merge_func"]
    return plan
