import torch

from ...kernels import Kernel, coordinate_counts, integer_sum
from ...method_policies import is_edge, leftover_label, leftover_value
from ...source import FilteredSource

def _ties_settings(parameters):
    density, scale = parameters[0], parameters[1]
    merge_func = parameters[2] if len(parameters) > 2 else "mean"
    exclude_edge = parameters[3] if len(parameters) > 3 else False
    leftover_edge = parameters[4] if len(parameters) > 4 else "base"
    return density, scale, merge_func, exclude_edge, leftover_edge

class TIES(Kernel):
    name = "ties"

    def prepare(self, deltas):
        density, _, _, exclude_edge, _ = _ties_settings(self.parameters)
        selected = FilteredSource(
            deltas, lambda block: not (exclude_edge and is_edge(block))
        )
        thresholds = (
            tuple(selected.global_thresholds(density, "ties"))
            if not selected.is_empty()
            else ()
        )

        sign_balance = 0
        for block in selected.scan():
            trimmed = self._trim(block, thresholds)
            elected = torch.sign(trimmed.sum(dim=0))
            sign_balance += integer_sum(elected > 0) - integer_sum(elected < 0)
            del trimmed, elected
        majority = (sign_balance > 0) - (sign_balance < 0)
        self.stats.update(
            target_density=density,
            thresholds=list(thresholds),
            threshold_rule="official_ties_kth_inclusive",
            exclude_edge=exclude_edge,
            zero_sign_scope="global",
            global_majority_sign=majority,
        )
        return thresholds, majority

    @staticmethod
    def _trim(block, thresholds):
        values = torch.stack(block.deltas)
        flat = values.reshape(len(values), -1)
        thresholds = torch.as_tensor(thresholds, dtype=values.dtype, device=values.device).unsqueeze(1)
        return torch.where(flat.abs() >= thresholds, flat, 0)

    def combine(self, block, state):
        _, scale, merge_func, exclude_edge, leftover_edge = _ties_settings(self.parameters)
        if exclude_edge and is_edge(block):
            value = leftover_value(block, leftover_edge)
            label = leftover_label(leftover_edge, "edge")
            self.stats[label] = self.stats.get(label, 0) + value.numel()
            return self.counted(block, value)
        thresholds, majority = state
        trimmed = self._trim(block, thresholds)
        expert_count = len(block.deltas)
        elected = torch.sign(trimmed.sum(dim=0))
        zero_sum = elected == 0
        elected = torch.where(zero_sum, majority, elected)
        selected = torch.where(elected.unsqueeze(0) > 0, trimmed > 0, trimmed < 0)
        merged = torch.where(selected, trimmed, 0).sum(dim=0)
        if merge_func != "sum":
            count = coordinate_counts(selected, torch.float32)
            merged = merged / count.clamp_(min=1)

        self.stats.setdefault("expert_trimmed", [0] * expert_count)
        trimmed_counts = [integer_sum(row != 0) for row in trimmed]
        for index in range(expert_count):
            self.stats["expert_trimmed"][index] += trimmed_counts[index]
        self.stats["disjoint_selected"] = (
            self.stats.get("disjoint_selected", 0) + integer_sum(selected)
        )
        self.stats["zero_sum_coordinates"] = (
            self.stats.get("zero_sum_coordinates", 0) + integer_sum(zero_sum)
        )
        role_stats = self.stats.setdefault("expert_trimmed_by_role", {})
        for role in block.roles:
            counts = role_stats.setdefault(role, [0] * expert_count)
            for index in range(expert_count):
                counts[index] += trimmed_counts[index]
        merged = merged.reshape(block.shape)
        return self.counted(block, block.base + scale * merged)

def build(parameters, shared):
    return TIES(parameters, shared)
