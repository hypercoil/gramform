# -*- coding: utf-8 -*-
"""
Inference: permutation (voxel / cluster-extent / cluster-mass / TFCE) with
max-statistic FWE, and parametric FDR (Benjamini-Hochberg) / Bonferroni.

The permutation scheme is Freedman-Lane: the tested effect is removed, the
reduced-model residuals are permuted, the full model is refit, and the
contrast statistic recomputed -- the standard scheme for a contrast in the
presence of nuisance regressors (FSL ``randomise`` / PALM). Spatial
enhancement (clusters / TFCE) uses a supplied vertex adjacency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Adjacency = tuple[np.ndarray, ...]


# ---------------------------------------------------------------------------
# spatial enhancement
# ---------------------------------------------------------------------------


def _connected_components(
    mask: np.ndarray,
    adjacency: Adjacency,
) -> list[np.ndarray]:
    visited = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    for start in np.flatnonzero(mask):
        if visited[start]:
            continue
        stack = [int(start)]
        visited[start] = True
        comp: list[int] = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for w in adjacency[u]:
                if mask[w] and not visited[w]:
                    visited[w] = True
                    stack.append(int(w))
        comps.append(np.array(comp))
    return comps


def _tfce(
    mag: np.ndarray,
    adjacency: Adjacency,
    *,
    E: float = 0.5,
    H: float = 2.0,
    n_steps: int = 100,
) -> np.ndarray:
    out = np.zeros_like(mag)
    peak = float(mag.max())
    if peak <= 0:
        return out
    dh = peak / n_steps
    for step in range(1, n_steps + 1):
        h = step * dh
        for comp in _connected_components(mag >= h, adjacency):
            out[comp] += (comp.size**E) * (h**H) * dh
    return out


def _enhance_onesided(
    mag: np.ndarray,
    kind: str,
    adjacency: Adjacency | None,
    threshold: float | None,
) -> np.ndarray:
    if adjacency is None:
        raise ValueError(f'enhancement {kind!r} requires a vertex adjacency')
    if kind == 'tfce':
        return _tfce(mag, adjacency)
    if threshold is None:
        raise ValueError(f'enhancement {kind!r} requires a cluster threshold')
    out = np.zeros_like(mag)
    for comp in _connected_components(mag > threshold, adjacency):
        out[comp] = comp.size if kind == 'cluster_extent' else mag[comp].sum()
    return out


def enhance(
    stat: np.ndarray,
    kind: str,
    adjacency: Adjacency | None,
    threshold: float | None,
) -> np.ndarray:
    """Map a (signed) statistic image to a non-negative enhanced image used
    for the max-statistic null. Two-sided: clusters / TFCE are computed per
    sign and combined elementwise."""
    if kind == 'voxel':
        return np.abs(stat)
    pos = _enhance_onesided(
        np.clip(stat, 0.0, None), kind, adjacency, threshold
    )
    neg = _enhance_onesided(
        np.clip(-stat, 0.0, None), kind, adjacency, threshold
    )
    return np.maximum(pos, neg)


# ---------------------------------------------------------------------------
# the GLM contrast statistic (inlined for the permutation hot loop)
# ---------------------------------------------------------------------------


def _stat_map(
    Y: np.ndarray,
    X: np.ndarray,
    xtx_inv: np.ndarray,
    c: np.ndarray,
    var_c: float,
    dof: int,
    stat_kind: str,
) -> np.ndarray:
    beta = xtx_inv @ (X.T @ Y)
    resid = Y - X @ beta
    sigma2 = np.einsum('nv,nv->v', resid, resid) / dof
    effect = c @ beta
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(sigma2 > 0, effect / np.sqrt(sigma2 * var_c), 0.0)
    return t * t if stat_kind == 'F' else t


# ---------------------------------------------------------------------------
# permutation test (Freedman-Lane, max-statistic FWE)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PermutationResult:
    stat: np.ndarray  # observed (signed for t) statistic map
    enhanced: np.ndarray  # observed enhanced statistic map
    corrected: np.ndarray  # FWE-corrected p-map
    n_perm: int


def permutation_test(
    Y: np.ndarray,
    X: np.ndarray,
    c: np.ndarray,
    *,
    stat_kind: str = 't',
    enhancement: str = 'voxel',
    n_perm: int = 1000,
    adjacency: Adjacency | None = None,
    threshold: float | None = None,
    seed: int = 0,
) -> PermutationResult:
    n, p = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    var_c = float(c @ xtx_inv @ c)
    dof = n - p

    tested = np.flatnonzero(c != 0)
    Z = np.delete(X, tested, axis=1)
    if Z.shape[1]:
        fitted_reduced = Z @ (np.linalg.pinv(Z) @ Y)
    else:  # no nuisance: permute Y about its grand structure
        fitted_reduced = np.zeros_like(Y)
    resid_reduced = Y - fitted_reduced

    obs = _stat_map(Y, X, xtx_inv, c, var_c, dof, stat_kind)
    enhanced_obs = enhance(obs, enhancement, adjacency, threshold)

    rng = np.random.default_rng(seed)
    max_null = np.empty(n_perm)
    for i in range(n_perm):
        perm = np.arange(n) if i == 0 else rng.permutation(n)
        y_star = fitted_reduced + resid_reduced[perm]
        stat = _stat_map(y_star, X, xtx_inv, c, var_c, dof, stat_kind)
        max_null[i] = enhance(stat, enhancement, adjacency, threshold).max()

    corrected = (max_null[:, None] >= enhanced_obs[None, :]).mean(axis=0)
    return PermutationResult(
        stat=obs,
        enhanced=enhanced_obs,
        corrected=corrected,
        n_perm=n_perm,
    )


# ---------------------------------------------------------------------------
# parametric multiple-comparison corrections
# ---------------------------------------------------------------------------


def fdr_bh(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * m / (np.arange(1, m + 1))
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def bonferroni(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    return np.clip(p * p.size, 0.0, 1.0)
