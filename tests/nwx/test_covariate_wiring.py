# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Phase-7 covariate -> term wiring.

A confound shorthand (``rps``, ``wm``, ...) is lowered to a ``CovariateRef``
factor pointing into the node's emit-only ``CovariateProgram`` -- but **only in
a confound context**: where terms are finalised as *nuisance* (``noise()`` ->
``ModelSpec.partial``; the residualise ``~|`` noise set). Modelled effects
(fixed design, ``signal()``, random effects, smooths) keep plain ``Lookup``s,
and a non-shorthand name (``csf``, a user covariate) is never rewritten.
"""

import warnings

import pytest

from gramform.grammars.nwx.covariate import Shorthand
from gramform.grammars.nwx.spec import (
    INTERCEPT,
    CovariateRef,
    FactorSpec,
    Lookup,
    TermSpec,
)
from gramform.grammars.nwx.transform import get_processor


@pytest.fixture(scope='module')
def process():
    return get_processor()


def L(name: str) -> TermSpec:
    return TermSpec((FactorSpec(Lookup(name)),))


def C(index: int) -> TermSpec:
    return TermSpec((FactorSpec(CovariateRef(index)),))


# ---------------------------------------------------------------------------
# shorthands expand to CovariateRefs in nuisance positions
# ---------------------------------------------------------------------------


def test_noise_partial_lowers_shorthands(process):
    spec = process('thk ~ dx + noise(rps + meanFD)').nodes[0].spec
    # rps (a shorthand) -> CovariateRef(0); meanFD (not a shorthand) -> Lookup.
    assert spec.partial == (C(0), L('meanFD'))
    assert spec.covariates == (Shorthand('rps'),)
    # the fixed design is untouched
    assert spec.fixed == (INTERCEPT, L('dx'))


def test_residualise_noise_lowers_shorthands(process):
    spec = process('bold ~| rps + wm + csf').nodes[0].spec
    res = spec.residualise[0]
    assert res.noise == (INTERCEPT, C(0), C(1), L('csf'))
    assert spec.covariates == (Shorthand('rps'), Shorthand('wm'))


def test_csf_stays_a_plain_lookup(process):
    # `csf` is a passthrough column, never a confound op.
    spec = process('bold ~| csf').nodes[0].spec
    assert spec.residualise[0].noise == (INTERCEPT, L('csf'))
    assert spec.covariates == ()


# ---------------------------------------------------------------------------
# shorthands are NOT expanded outside a confound context
# ---------------------------------------------------------------------------


def test_fixed_design_keeps_shorthand_as_lookup(process):
    # In the modelled design `rps` is a literal column, not a confound.
    spec = process('thk ~ rps + dx').nodes[0].spec
    assert spec.fixed == (INTERCEPT, L('rps'), L('dx'))
    assert spec.covariates == ()


def test_signal_is_not_a_confound_context(process):
    # signal() preserves variance; its terms are modelled, not nuisance.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        spec = process(
            'bold ~| rps + signal(wm) {{ residualise=nonaggressive }}'
        ).nodes[0].spec
    res = spec.residualise[0]
    assert res.noise == (INTERCEPT, C(0))
    assert res.signal == (L('wm'),)  # `wm` here is NOT lowered
    assert spec.covariates == (Shorthand('rps'),)


# ---------------------------------------------------------------------------
# dedup + frames
# ---------------------------------------------------------------------------


def test_repeated_shorthand_is_one_op(process):
    spec = process('bold ~| rps + wm + rps').nodes[0].spec
    assert spec.covariates == (Shorthand('rps'), Shorthand('wm'))
    # both `rps` occurrences resolve to the same op index
    assert spec.residualise[0].noise == (INTERCEPT, C(0), C(1))


def test_covariates_bind_to_their_node(process):
    # A residualise frame's confounds land on the frame node, not the root.
    g = process('y ~ x + [bold ~| rps]')
    assert g.nodes[0].spec.covariates == ()
    assert g.nodes[1].spec.covariates == (Shorthand('rps'),)


def test_wiring_ir_is_hashable_and_value_equal(process):
    g1 = process('bold ~| rps + wm + csf')
    g2 = process('bold ~| rps + wm + csf')
    assert g1 == g2
    assert hash(g1) == hash(g2)
