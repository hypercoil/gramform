# -*- coding: utf-8 -*-
"""
End-to-end engine tests: the acceptance-gate formula runs formula->corrected
map; the design matches expectation; and the engine reads exactly the IR it
supports, erroring helpfully on the rest (the contract).
"""

import numpy as np
import pandas as pd
import pytest

from gramform.grammars.nwx.transform import get_processor
from nwx_reference_engine import EngineError, run
from nwx_reference_engine.datasets import synthetic_vertexwise
from nwx_reference_engine.materialise import materialise

GATE_FORMULA = (
    'thk ~ dx + sex + noise(meanFD) '
    '{{ contrasts: dx = dx (t); inference=permutation(tfce, n=200) }}'
)


def test_acceptance_gate_formula_to_corrected_map():
    # M1: a formula string -> a corrected statistical map, end to end.
    ds = synthetic_vertexwise(seed=0)
    results = run(
        GATE_FORMULA,
        ds.data,
        ds.imaging,
        adjacency=ds.adjacency,
        seed=1,
    )
    r = results['dx']
    assert r.stat.shape == (ds.imaging.shape[1],)
    assert r.corrected.shape == r.stat.shape
    assert np.all((r.corrected >= 0) & (r.corrected <= 1))
    assert 'permutation-tfce' in r.correction
    # the true cluster is recovered; the null field is controlled
    detected = r.corrected[ds.true_mask] < 0.05
    false_pos = r.corrected[~ds.true_mask] < 0.05
    assert detected.mean() > 0.5
    assert false_pos.sum() == 0


def test_design_columns_and_coding():
    ds = synthetic_vertexwise(seed=0)
    node = get_processor()(GATE_FORMULA).nodes[0]
    design = materialise(node, ds.data, ds.imaging)
    # intercept + dx (numeric 0/1) + sex treatment-coded + meanFD (partial)
    assert design.columns == ('Intercept', 'dx', 'sex[T.M]', 'meanFD')
    assert design.signal_cols == (0, 1, 2)
    assert design.partial_cols == (3,)
    # the dx contrast loads only on the dx column
    c = design.contrast_vector(node.spec.estimands[0])
    assert list(c) == [0.0, 1.0, 0.0, 0.0]


def test_response_from_covariate_without_imaging():
    rng = np.random.default_rng(0)
    n = 50
    x = rng.normal(size=n)
    y = 2.0 * x + rng.normal(size=n)
    data = pd.DataFrame({'y': y, 'x': x})
    r = run(
        'y ~ x {{ contrasts: x = x (t); inference=parametric(fdr) }}',
        data,
        imaging=None,
    )['x']
    assert r.stat.shape == (1,)
    assert r.stat[0] > 5  # strong true effect


@pytest.mark.parametrize('seed', [1, 7, 13])
def test_permutation_is_deterministic_given_seed(seed):
    ds = synthetic_vertexwise(seed=0)
    a = run(
        GATE_FORMULA, ds.data, ds.imaging, adjacency=ds.adjacency, seed=seed
    )
    b = run(
        GATE_FORMULA, ds.data, ds.imaging, adjacency=ds.adjacency, seed=seed
    )
    np.testing.assert_array_equal(a['dx'].corrected, b['dx'].corrected)


# ---------------------------------------------------------------------------
# contract: the engine errors helpfully on IR it does not lower
# ---------------------------------------------------------------------------


def test_non_gaussian_family_rejected():
    ds = synthetic_vertexwise(seed=0)
    with pytest.raises(EngineError, match='Gaussian'):
        run(
            'thk ~ dx {{ family=binomial; contrasts: dx=dx (t) }}',
            ds.data,
            ds.imaging,
        )


def test_residualise_rejected():
    ds = synthetic_vertexwise(seed=0)
    with pytest.raises(EngineError, match='residualis'):
        run(
            'thk ~| meanFD {{ contrasts: x=x (t) }}',
            ds.data,
            ds.imaging,
        )


def test_frame_multinode_rejected():
    ds = synthetic_vertexwise(seed=0)
    with pytest.raises(EngineError, match='single-node'):
        run(
            'thk ~ dx + [meanFD ~ sex] {{ contrasts: dx=dx (t) }}',
            ds.data,
            ds.imaging,
        )


def test_unknown_contrast_coefficient_rejected():
    ds = synthetic_vertexwise(seed=0)
    with pytest.raises(EngineError, match='unknown'):
        run(
            'thk ~ dx {{ contrasts: c = nonexistent (t) }}',
            ds.data,
            ds.imaging,
        )


def test_missing_covariate_rejected():
    ds = synthetic_vertexwise(seed=0)
    with pytest.raises(EngineError, match='not found'):
        run(
            'thk ~ notacolumn {{ contrasts: c = notacolumn (t) }}',
            ds.data,
            ds.imaging,
        )


def test_no_contrasts_rejected():
    ds = synthetic_vertexwise(seed=0)
    with pytest.raises(EngineError, match='no contrasts'):
        run('thk ~ dx', ds.data, ds.imaging)
