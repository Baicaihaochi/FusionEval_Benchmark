"""Small, framework-independent helpers for multi-domain gradient surgery."""

from __future__ import annotations

import itertools

import torch


def gradient_norm_cap_scales(
    gram: torch.Tensor,
    max_median_ratio: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return per-domain scales for a robust gradient-norm upper bound.

    The cap is ``max_median_ratio`` times the conventional midpoint median of
    the domain gradient norms.  Domains below the cap are exact no-ops.  A
    non-positive ratio disables the cap, while a zero median is treated as an
    all-one no-op to avoid erasing the only active domain in a degenerate step.
    """

    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError(f"gram must be square, got shape={tuple(gram.shape)}")
    if not torch.isfinite(gram).all():
        raise ValueError("gram contains NaN or Inf")
    if max_median_ratio < 0:
        raise ValueError("max_median_ratio must be non-negative")

    n = gram.shape[0]
    if n == 0:
        return torch.empty(0, dtype=gram.dtype, device=gram.device)
    if max_median_ratio == 0:
        return torch.ones(n, dtype=gram.dtype, device=gram.device)

    norms = torch.sqrt(torch.clamp(torch.diag(gram), min=0))
    ordered = torch.sort(norms).values
    midpoint = n // 2
    if n % 2:
        median = ordered[midpoint]
    else:
        median = (ordered[midpoint - 1] + ordered[midpoint]) / 2
    limit = median * max_median_ratio
    if limit <= eps:
        return torch.ones_like(norms)

    scales = (limit / norms.clamp_min(eps)).clamp(max=1.0)
    return torch.where(norms <= eps, torch.ones_like(scales), scales)


def symmetric_pcgrad_coefficients(gram: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return coefficients that reproduce deterministic symmetric PCGrad.

    ``gram[i, j]`` is ``<g_i, g_j>``.  Instead of materializing projected
    gradients, the routine tracks each projected task gradient as a linear
    combination of the original gradients.  Averaging every ordering of the
    other tasks removes PCGrad's otherwise arbitrary task-order dependence.

    The returned coefficients are meant to multiply *per-domain contributions*
    that were each normalized by the full global batch size.  Consequently we
    sum, rather than average, the projected task gradients; all-one coefficients
    exactly recover the ordinary joint objective when no conflicts are present.
    """

    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError(f"gram must be square, got shape={tuple(gram.shape)}")
    if not torch.isfinite(gram).all():
        raise ValueError("gram contains NaN or Inf")

    n = gram.shape[0]
    if n == 0:
        return torch.empty(0, dtype=gram.dtype, device=gram.device)

    projected = []
    for task in range(n):
        others = [j for j in range(n) if j != task]
        orders = list(itertools.permutations(others)) or [()]
        mean_coeff = torch.zeros(n, dtype=gram.dtype, device=gram.device)
        for order in orders:
            coeff = torch.zeros(n, dtype=gram.dtype, device=gram.device)
            coeff[task] = 1
            for other in order:
                denom = gram[other, other]
                if denom <= eps:
                    continue
                dot = torch.dot(coeff, gram[:, other])
                if dot < 0:
                    coeff[other] -= dot / denom
            mean_coeff += coeff
        projected.append(mean_coeff / len(orders))

    return torch.stack(projected).sum(dim=0)


def gradient_cosine_matrix(gram: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Convert a gradient Gram matrix to a numerically safe cosine matrix."""

    norms = torch.sqrt(torch.clamp(torch.diag(gram), min=0))
    denom = torch.outer(norms, norms).clamp_min(eps)
    cosine = gram / denom
    zero = norms <= eps
    cosine[zero, :] = 0
    cosine[:, zero] = 0
    return cosine
