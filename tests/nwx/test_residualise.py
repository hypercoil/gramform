# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Phase-5 residualisation modes (spec §4.6).

``~|`` is aggressive by default (project onto the orthogonal complement of the
noise set); ``{{ residualise=nonaggressive }}`` flips the mode and **requires** a
``signal()`` set. ``signal()`` / ``noise()`` are call-position role markers that
route terms structurally into ``ResidualiseSpec.signal`` / ``.noise``;
``noise()`` on a *normal* RHS instead routes to ``ModelSpec.partial`` (FWL,
Phase 2) -- the two constructs are kept distinct in the IR.
"""

import warnings

import pytest

from gramform.grammars.nwx.spec import (
    INTERCEPT,
    BackendWarning,
    FactorSpec,
    Lookup,
    Mode,
    ResidualiseSpec,
    TermSpec,
)
from gramform.grammars.nwx.transform import NwxError, get_processor


@pytest.fixture(scope='module')
def process():
    return get_processor()


def L(name: str) -> TermSpec:
    return TermSpec((FactorSpec(Lookup(name)),))


def parse(process, formula: str):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', BackendWarning)
        warnings.simplefilter('ignore', UserWarning)
        return process(formula)


def residualise_of(process, formula: str) -> ResidualiseSpec:
    return parse(process, formula).nodes[0].spec.residualise[0]


# ---------------------------------------------------------------------------
# aggressive default
# ---------------------------------------------------------------------------


def test_aggressive_is_default(process):
    # Non-shorthand noise names stay plain Lookups (confound shorthands ->
    # CovariateRef are exercised in test_covariate_wiring.py).
    r = residualise_of(process, 'bold ~| n1 + n2')
    assert r == ResidualiseSpec(
        target=(L('bold'),),
        noise=(INTERCEPT, L('n1'), L('n2')),
        signal=(),
        mode=Mode.AGGRESSIVE,
    )


def test_aggressive_keeps_fixed_empty(process):
    spec = parse(process, 'bold ~| rps + wm').nodes[0].spec
    assert spec.fixed == ()
    assert len(spec.residualise) == 1


# ---------------------------------------------------------------------------
# signal() / noise() routing + non-aggressive
# ---------------------------------------------------------------------------


def test_nonaggressive_routes_signal_and_noise(process):
    r = residualise_of(
        process,
        'bold ~| noise(aroma1 + aroma2) + signal(task + drift) '
        '{{ residualise=nonaggressive }}',
    )
    assert r.mode is Mode.NONAGGRESSIVE
    # ppr_add_intercept de-means the nuisance set: intercept leads the noise.
    assert r.noise == (INTERCEPT, L('aroma1'), L('aroma2'))
    assert r.signal == (L('task'), L('drift'))


def test_unwrapped_terms_are_noise(process):
    # Unwrapped terms on a `~|` RHS join the noise set alongside noise().
    r = residualise_of(
        process,
        'bold ~| motion + noise(aroma1) + signal(task) '
        '{{ residualise=nonaggressive }}',
    )
    assert r.noise == (INTERCEPT, L('motion'), L('aroma1'))
    assert r.signal == (L('task'),)


def test_nonaggressive_without_signal_is_a_hard_error(process):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', BackendWarning)
        with pytest.raises(NwxError, match='nonaggressive requires a signal'):
            process('bold ~| n {{ residualise=nonaggressive }}')


def test_signal_on_normal_rhs_is_an_error(process):
    with pytest.raises(NwxError, match='only valid on a residualisation'):
        process('y ~ x + signal(z)')


# ---------------------------------------------------------------------------
# FWL partial vs residualisation are kept distinct
# ---------------------------------------------------------------------------


def test_noise_on_normal_rhs_is_partial_not_residualise(process):
    spec = parse(process, 'thk ~ dx + noise(meanFD)').nodes[0].spec
    assert spec.partial == (L('meanFD'),)
    assert spec.residualise == ()


def test_noise_on_residualise_rhs_is_noise_not_partial(process):
    spec = parse(process, 'bold ~| noise(meanFD) + signal(task)').nodes[0].spec
    assert spec.partial == ()
    assert spec.residualise[0].noise == (INTERCEPT, L('meanFD'))


# ---------------------------------------------------------------------------
# warnings
# ---------------------------------------------------------------------------


def test_aggressive_with_signal_warns(process):
    with pytest.warns(UserWarning, match='aggressive residualisation ignores'):
        process('bold ~| signal(task) + n')


def test_residualise_directive_without_residual_op_warns(process):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', BackendWarning)
        with pytest.warns(UserWarning, match='no .*residualisation'):
            process('y ~ x {{ residualise=nonaggressive }}')


def test_nonaggressive_emits_backend_warning(process):
    with pytest.warns(BackendWarning, match='nitrix v3'):
        process(
            'bold ~| noise(n) + signal(s) {{ residualise=nonaggressive }}'
        )


# ---------------------------------------------------------------------------
# frames + value equality
# ---------------------------------------------------------------------------


def test_residualise_in_frame_carries_signal(process):
    # A residualise frame with a signal() set lands on the frame sub-node.
    g = parse(process, 'y ~ x + [bold ~| noise(n) + signal(s)]')
    res = g.nodes[1].spec.residualise[0]
    assert res.signal == (L('s'),)
    assert res.noise == (INTERCEPT, L('n'))


def test_residualise_ir_is_hashable_and_value_equal(process):
    g1 = parse(process, 'bold ~| rps + wm')
    g2 = parse(process, 'bold ~| rps + wm')
    assert g1 == g2
    assert hash(g1) == hash(g2)
