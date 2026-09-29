from __future__ import annotations

def solve_weight(
    expert_weights,
    grams,
    crosses,
    merged_weight,
    base_weight,
    ridge_lambda,
    anchor_rho,
    covariance_eps,
    device,
):

    import torch

    if not expert_weights or len(expert_weights) != len(grams) or len(grams) != len(crosses):
        raise ValueError("nonempty weight/Gram/cross lists must have equal length")
    matrix = None
    rhs = None
    norms = []
    for weight, gram, cross in zip(expert_weights, grams, crosses):
        gram = gram.to(device=device, dtype=torch.float32)
        cross = cross.to(device=device, dtype=torch.float32)
        if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or cross.shape != gram.shape:
            raise ValueError("FeatCal Gram and cross matrices must be aligned square matrices")
        if weight.ndim != 2 or weight.shape != merged_weight.shape or weight.shape != base_weight.shape or weight.shape[1] != gram.shape[0]:
            raise ValueError("FeatCal weight and feature dimensions do not align")
        norm = gram.norm(p="fro").clamp_min(float(covariance_eps))
        norms.append(float(norm))
        current_matrix = gram / norm
        current_rhs = (cross / norm) @ weight.to(device=device, dtype=torch.float32).T
        matrix = current_matrix.clone() if matrix is None else matrix.add_(current_matrix)
        rhs = current_rhs.clone() if rhs is None else rhs.add_(current_rhs)
    identity = torch.eye(matrix.shape[0], device=device, dtype=torch.float32)
    ridge = float(ridge_lambda)
    matrix.add_(identity, alpha=ridge + float(covariance_eps))
    anchor = (
        float(anchor_rho) * merged_weight.to(device=device, dtype=torch.float32)
        + (1.0 - float(anchor_rho)) * base_weight.to(device=device, dtype=torch.float32)
    )
    rhs.add_(anchor.T, alpha=ridge)

    used_pinv = False
    try:
        solved = torch.linalg.solve(matrix, rhs)
    except RuntimeError:
        used_pinv = True
        solved = torch.linalg.pinv(matrix) @ rhs
    if not bool(torch.isfinite(solved).all()):
        raise RuntimeError("FeatCal solve produced non-finite weights")
    return solved.T.to(expert_weights[0].dtype), {
        "pseudoinverse": used_pinv,
        "task_gram_norms": norms,
    }
