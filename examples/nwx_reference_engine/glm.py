# -*- coding: utf-8 -*-
"""
Mass-univariate Gaussian GLM (OLS) + t / F contrasts.

Vectorised over the mass axis: one design ``X`` (n_obs x p) is fit against all
``v`` response columns at once via the normal equations. A contrast that loads
only on the signal columns of a full design (signal + ``partial`` nuisance)
yields the Frisch-Waugh-Lovell partial coefficient -- so in-model nuisance
(``noise()``) is handled by inclusion, not a separate residualisation pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class GLMFit:
    """A fitted mass-univariate Gaussian GLM."""

    beta: np.ndarray  # (p, v)
    xtx_inv: np.ndarray  # (p, p)
    sigma2: np.ndarray  # (v,) residual variance, ddof = p
    dof: int  # residual degrees of freedom (n - rank)
    residuals: np.ndarray  # (n, v)


def glm_fit(Y: np.ndarray, X: np.ndarray) -> GLMFit:
    """Fit ``Y = X beta + e`` (Gaussian) over every column of ``Y``."""
    n, p = X.shape
    rank = int(np.linalg.matrix_rank(X))
    if rank < p:
        raise np.linalg.LinAlgError(
            f'design is rank-deficient (rank {rank} < {p} columns); '
            'check for collinear / duplicated covariates'
        )
    xtx = X.T @ X
    xtx_inv = np.linalg.inv(xtx)
    beta = xtx_inv @ (X.T @ Y)
    residuals = Y - X @ beta
    dof = n - p
    sigma2 = np.einsum('nv,nv->v', residuals, residuals) / dof
    return GLMFit(
        beta=beta,
        xtx_inv=xtx_inv,
        sigma2=sigma2,
        dof=dof,
        residuals=residuals,
    )


def t_contrast(
    fit: GLMFit,
    c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """A two-sided t-contrast. Returns ``(t_map, p_map)`` over the mass."""
    c = np.asarray(c, dtype=float)
    effect = c @ fit.beta  # (v,)
    var_c = float(c @ fit.xtx_inv @ c)
    se = np.sqrt(fit.sigma2 * var_c)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(se > 0, effect / se, 0.0)
    p = 2.0 * stats.t.sf(np.abs(t), fit.dof)
    return t, p


def f_contrast(
    fit: GLMFit,
    C: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """An F-contrast for the ``q x p`` matrix ``C``. Returns ``(F_map,
    p_map)``. With ``q = 1`` this equals the squared t-statistic."""
    C = np.atleast_2d(np.asarray(C, dtype=float))
    q = C.shape[0]
    effect = C @ fit.beta  # (q, v)
    middle = np.linalg.inv(C @ fit.xtx_inv @ C.T)  # (q, q)
    # numerator quadratic form per voxel: effect_v^T middle effect_v
    quad = np.einsum('qv,qr,rv->v', effect, middle, effect)
    with np.errstate(divide='ignore', invalid='ignore'):
        F = np.where(fit.sigma2 > 0, quad / (q * fit.sigma2), 0.0)
    p = stats.f.sf(F, q, fit.dof)
    return F, p
