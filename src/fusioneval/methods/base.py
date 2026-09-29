from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Literal, Mapping, Optional, Tuple

from ..protocol import KernelRequirements

Normalize = Callable[[Mapping[str, Any]], Dict[str, Any]]

@dataclass(frozen=True)
class MethodContext:
    expert_ids: Tuple[str, ...]

Contextualize = Callable[[Dict[str, Any], MethodContext], Dict[str, Any]]

@dataclass(frozen=True)
class ContractVariant:
    name: str = "standard"
    add_tags: Tuple[str, ...] = ()
    remove_tags: Tuple[str, ...] = ()

Contractualize = Callable[[Mapping[str, Any]], ContractVariant]

@dataclass(frozen=True)
class MethodPlugin:

    name: str
    aliases: Tuple[str, ...]
    op_code: str
    contract_version: int
    requirements: KernelRequirements
    parameter_scope: str
    aggregation: str
    contract_tags: Tuple[str, ...]
    user_parameters: Tuple[str, ...]
    runtime_parameters: Tuple[str, ...]
    normalize: Normalize
    contextualize: Optional[Contextualize] = None
    compute_precision: str = "runtime"
    execution: str = "tensor"
    requires_data: bool = False
    data_mode: Literal["none", "per_expert", "shared"] = "none"
    requires_initial_model: bool = False
    minimum_experts: int = 2
    input_mode: Literal["deltas", "experts"] = "deltas"
    status: str = "implemented"
    contractualize: Optional[Contractualize] = None

    @property
    def local_name(self) -> str:
        return self.name

    @property
    def contract_id(self) -> str:
        return "{}.v{}".format(self.op_code, self.contract_version)

    @property
    def stochastic(self) -> bool:
        return self.requirements.stochastic

    def configure(
        self, raw: Mapping[str, Any], context: Optional[MethodContext] = None
    ) -> Dict[str, Any]:
        value = self.normalize(raw)
        if context is not None and self.contextualize is not None:
            value = self.contextualize(value, context)
        return value

    def encode(self, configured: Mapping[str, Any]) -> Tuple[Any, ...]:
        if set(configured) != set(self.runtime_parameters):
            raise ValueError("configured parameters do not match the plugin contract")
        return tuple(configured[name] for name in self.runtime_parameters)

    def effective_contract(self, configured: Mapping[str, Any]) -> Dict[str, Any]:
        variant = (
            self.contractualize(configured)
            if self.contractualize is not None
            else ContractVariant()
        )
        tags = [tag for tag in self.contract_tags if tag not in variant.remove_tags]
        for tag in variant.add_tags:
            if tag not in tags:
                tags.append(tag)
        return {
            "id": self.contract_id
            if variant.name == "standard"
            else "{}.{}".format(self.contract_id, variant.name),
            "variant": variant.name,
            "tags": tags,
        }

    @property
    def kernel_module(self) -> str:
        return "fusioneval.methods.{}.kernel".format(self.name)

    @property
    def runner_module(self) -> str:
        return "fusioneval.methods.{}.runner".format(self.name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "aliases": list(self.aliases),
            "op_code": self.op_code,
            "contract_id": self.contract_id,
            "requirements": asdict(self.requirements),
            "parameter_scope": self.parameter_scope,
            "aggregation": self.aggregation,
            "contract_tags": list(self.contract_tags),
            "user_parameters": list(self.user_parameters),
            "runtime_parameter_count": len(self.runtime_parameters),
            "compute_precision": self.compute_precision,
            "execution": self.execution,
            "implementation_module": (
                self.kernel_module if self.execution == "tensor" else self.runner_module
            ),
            "requires_data": self.requires_data,
            "data_mode": self.data_mode,
            "requires_initial_model": self.requires_initial_model,
            "minimum_experts": self.minimum_experts,
            "input_mode": self.input_mode,
            "status": self.status,
        }
