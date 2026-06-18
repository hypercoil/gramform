# -*- coding: utf-8 -*-
"""
Inference internals: FDR/Bonferroni against hand-computed references, the
spatial-enhancement primitives, and permutation FWE properties (validity,
determinism, and that a strong localised effect survives correction).
"""

import numpy as np
import pytest

from gramform.grammars.nwx.transform import get_processor
from nwx_reference_engine.datasets import chain_adjacency, synthetic_vertexwise
from nwx_reference_engine.inference import (
    bonferroni,
    enhance,
    fdr_bh,
    permutation_test,
)
from nwx_reference_engine.materialise import materialise


def test_bonferroni():
    p = np.array([0.001, 0.02, 0.5, 1.0])
    np.testing.assert_allclose(bonferroni(p), [0.004, 0.08, 1.0, 1.0])


def test_fdr_bh_reference():
    # Benjamini-Hochberg on a known vector (hand-computed, monotone-enforced).
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042])
    # raw BH = p * 5 / rank: .005, .02, .065, .05125, .042
    # enforce monotone-from-largest -> .005, .02, .042, .042, .042
    np.testing.assert_allclose(
        fdr_bh(p), [0.005, 0.02, 0.042, 0.042, 0.042], rtol=1e-9
    )


def test_fdr_bh_all_null_is_conservative():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=500)
    q = fdr_bh(p)
    assert np.all(q >= p - 1e-12)  # adjustment never decreases a p-value
    assert np.all((q >= 0) & (q <= 1))


def test_enhance_voxel_is_absolute():
    stat = np.array([-3.0, 1.0, 2.0])
    np.testing.assert_array_equal(
        enhance(stat, 'voxel', None, None), [3, 1, 2]
    )


def test_enhance_cluster_extent_and_mass():
    adj = chain_adjacency(6)
    stat = np.array([0.0, 3.0, 3.0, 0.0, 4.0, 0.0])  # cluster {1,2}, {4}
    extent = enhance(stat, 'cluster_extent', adj, threshold=1.0)
    mass = enhance(stat, 'cluster_mass', adj, threshold=1.0)
    np.testing.assert_array_equal(extent, [0, 2, 2, 0, 1, 0])
    np.testing.assert_array_equal(mass, [0, 6, 6, 0, 4, 0])


def test_enhance_two_sided_combines_signs():
    adj = chain_adjacency(4)
    stat = np.array([-5.0, -5.0, 0.0, 3.0])  # neg cluster {0,1}, pos {3}
    mass = enhance(stat, 'cluster_mass', adj, threshold=1.0)
    np.testing.assert_array_equal(mass, [10, 10, 0, 3])


def test_enhance_cluster_requires_adjacency_and_threshold():
    with pytest.raises(ValueError, match='adjacency'):
        enhance(np.zeros(3), 'tfce', None, None)
    with pytest.raises(ValueError, match='threshold'):
        enhance(np.zeros(3), 'cluster_mass', chain_adjacency(3), None)


def _design_for(formula):
    ds = synthetic_vertexwise(seed=0)
    node = get_processor()(formula).nodes[0]
    design = materialise(node, ds.data, ds.imaging)
    c = design.contrast_vector(node.spec.estimands[0])
    return design, c, ds


def test_permutation_pvalues_valid_and_bounded_below():
    design, c, _ = _design_for(
        'thk ~ dx + sex + noise(meanFD) {{ contrasts: dx=dx (t) }}'
    )
    res = permutation_test(
        design.Y, design.X, c, enhancement='voxel', n_perm=100, seed=0
    )
    assert np.all((res.corrected >= 0) & (res.corrected <= 1))
    assert res.corrected.min() >= 1.0 / res.n_perm  # identity perm included


def test_permutation_deterministic_and_recovers_signal():
    design, c, ds = _design_for(
        'thk ~ dx + sex + noise(meanFD) {{ contrasts: dx=dx (t) }}'
    )
    kw = dict(enhancement='voxel', n_perm=200, seed=3)
    a = permutation_test(design.Y, design.X, c, **kw)
    b = permutation_test(design.Y, design.X, c, **kw)
    np.testing.assert_array_equal(a.corrected, b.corrected)
    # the true cluster is more significant than the null field
    assert a.corrected[ds.true_mask].mean() < a.corrected[~ds.true_mask].mean()
