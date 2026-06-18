# -*- coding: utf-8 -*-
"""
GLM / contrast numerics pinned against independent oracles: ``np.linalg.lstsq``
for coefficients, ``scipy.stats.linregress`` for the single-predictor t, and a
QR-based OLS for the multi-predictor t / FWL partial coefficient.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from gramform.grammars.nwx.transform import get_processor
from nwx_reference_engine.glm import f_contrast, glm_fit, t_contrast
from nwx_reference_engine.materialise import materialise


def _ols_t_independent(X, y, c):
    """An independent OLS t-statistic via QR (not the engine's path)."""
    q, r = np.linalg.qr(X)
    beta = np.linalg.solve(r, q.T @ y)
    resid = y - X @ beta
    n, p = X.shape
    sigma2 = (resid @ resid) / (n - p)
    rtr_inv = np.linalg.inv(r.T @ r)  # = (X'X)^-1
    se = np.sqrt(sigma2 * (c @ rtr_inv @ c))
    return float((c @ beta) / se), n - p


def test_beta_matches_lstsq():
    rng = np.random.default_rng(0)
    X = np.column_stack(
        [np.ones(40), rng.normal(size=40), rng.normal(size=40)]
    )
    Y = rng.normal(size=(40, 7))
    fit = glm_fit(Y, X)
    beta_oracle, *_ = np.linalg.lstsq(X, Y, rcond=None)
    np.testing.assert_allclose(fit.beta, beta_oracle, atol=1e-9)


def test_single_predictor_t_matches_linregress():
    rng = np.random.default_rng(1)
    x = rng.normal(size=50)
    y = 0.8 * x + rng.normal(size=50)
    X = np.column_stack([np.ones(50), x])
    fit = glm_fit(y[:, None], X)
    t, p = t_contrast(fit, np.array([0.0, 1.0]))
    lr = stats.linregress(x, y)
    assert t[0] == pytest.approx(lr.slope / lr.stderr, rel=1e-9)
    assert p[0] == pytest.approx(lr.pvalue, rel=1e-9)


def test_multipredictor_t_matches_independent_qr():
    rng = np.random.default_rng(2)
    n = 64
    X = np.column_stack(
        [
            np.ones(n),
            rng.normal(size=n),
            rng.normal(size=n),
            rng.normal(size=n),
        ]
    )
    y = X @ np.array([1.0, 2.0, -1.0, 0.5]) + rng.normal(size=n)
    c = np.array([0.0, 1.0, 0.0, 0.0])
    fit = glm_fit(y[:, None], X)
    t, _ = t_contrast(fit, c)
    t_oracle, dof = _ols_t_independent(X, y, c)
    assert t[0] == pytest.approx(t_oracle, rel=1e-9)
    assert fit.dof == dof


def test_f_equals_t_squared_for_single_row():
    rng = np.random.default_rng(3)
    X = np.column_stack([np.ones(40), rng.normal(size=40)])
    Y = rng.normal(size=(40, 5))
    fit = glm_fit(Y, X)
    c = np.array([0.0, 1.0])
    t, _ = t_contrast(fit, c)
    F, _ = f_contrast(fit, c[None, :])
    np.testing.assert_allclose(F, t**2, rtol=1e-9)


def test_partial_coefficient_is_fwl():
    # The reported `dx` coefficient with `noise(meanFD)` in the model equals
    # the Frisch-Waugh-Lovell coefficient: regress dx and Y on the nuisance
    # block, then regress the residuals.
    rng = np.random.default_rng(4)
    n = 80
    dx = rng.normal(size=n)
    fd = rng.normal(size=n)
    y = 1.3 * dx + 0.7 * fd + rng.normal(size=n)
    data = pd.DataFrame({'dx': dx, 'meanFD': fd})

    node = get_processor()('y ~ dx + noise(meanFD)').nodes[0]
    design = materialise(node, data, y[:, None])
    fit = glm_fit(design.Y, design.X)
    dx_col = design.columns.index('dx')
    beta_dx = fit.beta[dx_col, 0]

    # FWL oracle
    Z = np.column_stack([np.ones(n), fd])
    res_y = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    res_dx = dx - Z @ np.linalg.lstsq(Z, dx, rcond=None)[0]
    beta_fwl = (res_dx @ res_y) / (res_dx @ res_dx)
    assert beta_dx == pytest.approx(beta_fwl, rel=1e-9)


def test_rank_deficient_design_raises():
    rng = np.random.default_rng(5)
    x = rng.normal(size=30)
    X = np.column_stack([np.ones(30), x, x])  # duplicated column
    with pytest.raises(np.linalg.LinAlgError):
        glm_fit(rng.normal(size=(30, 3)), X)
