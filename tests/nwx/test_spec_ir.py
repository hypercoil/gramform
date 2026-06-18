# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Golden-by-value IR tests for the Phase-1 ``spec`` interpreter.

Each canonical formula is asserted to equal a hand-written frozen
``ModelGraph``. The IR is immutable and value-comparable, so the whole graph is
compared in one ``==`` (frozen dataclasses + tuples, no mappings). Intercept
handling is pinned in every IR position (fixed terms, each frame node, the
residualise noise set).
"""

import warnings

import pytest

from gramform.grammars.nwx.spec import (
    INTERCEPT,
    Combine,
    Const,
    FactorSpec,
    Family,
    FamilySpec,
    Level,
    Lookup,
    Mode,
    ModelGraph,
    ModelNode,
    ModelSpec,
    Referent,
    ResponseSpec,
    TermSpec,
)
from gramform.grammars.nwx.transform import NwxError, get_processor


@pytest.fixture(scope='module')
def process():
    return get_processor()


def L(name: str) -> TermSpec:
    """A single-factor lookup term."""
    return TermSpec((FactorSpec(Lookup(name)),))


def root(spec: ModelSpec, *deps: ModelNode) -> ModelGraph:
    """A one-(or more-)node graph with the canonical root node."""
    node = ModelNode(
        name='root',
        level=Level.DATASET,
        group_by=(),
        combine=Combine.FIXED,
        spec=spec,
    )
    return ModelGraph(nodes=(node, *deps))


# ---------------------------------------------------------------------------
# golden frozen IR
# ---------------------------------------------------------------------------


def test_simple_additive(process):
    expected = root(
        ModelSpec(
            response=ResponseSpec(terms=(L('y'),)),
            fixed=(INTERCEPT, L('x'), L('z')),
        )
    )
    assert process('y ~ x + z') == expected


def test_intercept_only(process):
    expected = root(
        ModelSpec(
            response=ResponseSpec(terms=(L('y'),)),
            fixed=(INTERCEPT,),
        )
    )
    assert process('y ~ 1') == expected


def test_single_factor(process):
    expected = root(
        ModelSpec(
            response=ResponseSpec(terms=(L('y'),)),
            fixed=(INTERCEPT, L('group')),
        )
    )
    assert process('y ~ group') == expected


def test_interaction_crossing(process):
    xz = TermSpec((FactorSpec(Lookup('x')), FactorSpec(Lookup('z'))))
    expected = root(
        ModelSpec(
            response=ResponseSpec(terms=(L('y'),)),
            fixed=(INTERCEPT, L('x'), L('z'), xz),
        )
    )
    assert process('y ~ x*z') == expected


def test_nesting(process):
    ab = TermSpec((FactorSpec(Lookup('a')), FactorSpec(Lookup('b'))))
    expected = root(
        ModelSpec(
            response=ResponseSpec(terms=(L('y'),)),
            fixed=(INTERCEPT, L('a'), ab),
        )
    )
    assert process('y ~ a/b') == expected


def test_frame_referent(process):
    # `[z ~ w]` fits a sub-model and injects its fitted `_hat` referent into
    # the parent design; the sub-model is emitted as a second graph node.
    frame = ModelNode(
        name='frame0',
        level=Level.DATASET,
        group_by=(),
        combine=Combine.FIXED,
        spec=ModelSpec(
            response=ResponseSpec(terms=(L('z'),)),
            fixed=(INTERCEPT, L('w')),
        ),
    )
    referent = TermSpec((FactorSpec(Referent(stage='frame0', kind='_hat')),))
    expected = root(
        ModelSpec(
            response=ResponseSpec(terms=(L('y'),)),
            fixed=(INTERCEPT, L('x'), referent),
        ),
        frame,
    )
    assert process('y ~ x + [z ~ w]') == expected


def test_residualise_frame_tilde(process):
    # `[bold ~| n]` residualises and injects a `_tilde` referent.
    g = process('y ~ x + [bold ~| n]')
    assert g.nodes[1].name == 'frame0'
    res = g.nodes[1].spec.residualise
    assert len(res) == 1
    assert res[0].mode is Mode.AGGRESSIVE
    assert res[0].target == (L('bold'),)
    # ppr_add_intercept de-means the nuisance set: intercept joins the noise.
    assert res[0].noise == (INTERCEPT, L('n'))
    referent = TermSpec(
        (FactorSpec(Referent(stage='frame0', kind='_tilde')),)
    )
    assert referent in g.nodes[0].spec.fixed


def test_directives_applied(process):
    g = process(
        'y ~ x '
        '{{ family=binomial; estimator=reml; '
        'contrasts: dx = x (t); inference=permutation(tfce, n=200) }}'
    )
    spec = g.nodes[0].spec
    assert spec.family == FamilySpec(family=Family.BINOMIAL)
    assert spec.estimation.estimator == 'reml'
    assert len(spec.estimands) == 1
    assert spec.estimands[0].name == 'dx'
    assert spec.estimands[0].weights == (('x', 1.0),)
    assert spec.estimands[0].test.value == 't'
    assert spec.inference is not None
    assert spec.inference.kind == 'permutation'
    assert spec.inference.enhancement == 'tfce'
    assert spec.inference.n_perm == 200


def test_top_level_residualise(process):
    g = process('bold ~| rps + wm')
    spec = g.nodes[0].spec
    assert spec.fixed == ()
    assert len(spec.residualise) == 1
    assert spec.residualise[0].target == (L('bold'),)
    assert spec.residualise[0].noise == (INTERCEPT, L('rps'), L('wm'))


# ---------------------------------------------------------------------------
# intercept pinned in every position
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'formula,present',
    [
        ('y ~ x', True),
        ('y ~ x + z', True),
        ('y ~ group', True),
        ('y ~ x*z', True),
        ('y ~ x - 1', False),
        ('y ~ 0 + x', False),
        ('y ~ x + 0', False),
    ],
)
def test_intercept_handling(process, formula, present):
    fixed = process(formula).nodes[0].spec.fixed
    assert (INTERCEPT in fixed) is present


def test_intercept_pinned_first(process):
    # When present, the intercept leads the fixed design.
    assert process('y ~ x + z').nodes[0].spec.fixed[0] == INTERCEPT


def test_intercept_in_frame(process):
    frame = process('y ~ x + [z ~ w]').nodes[1]
    assert INTERCEPT in frame.spec.fixed


# ---------------------------------------------------------------------------
# structural singularity + restricted top-level `|`
# ---------------------------------------------------------------------------


def test_structural_singularity(process):
    with pytest.raises(ValueError):
        process('y ~ 1:x + 2:x')


def test_multipart_restricted(process):
    with pytest.raises(NwxError):
        process('y ~ x | z')


def test_bare_constant_dropped(process):
    # A constant scale other than 1 is not a valid term; it is dropped.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fixed = process('y ~ x + 2.5').nodes[0].spec.fixed
    assert all(t != TermSpec((FactorSpec(Const(2.5)),)) for t in fixed)


# ---------------------------------------------------------------------------
# immutability / hashability of the emitted IR
# ---------------------------------------------------------------------------


def test_term_order_and_is_intercept():
    assert INTERCEPT.order == 0
    assert INTERCEPT.is_intercept
    assert L('x').order == 1
    assert not L('x').is_intercept
    interaction = TermSpec(
        (FactorSpec(Lookup('x')), FactorSpec(Lookup('z')))
    )
    assert interaction.order == 2
    # A scaled main effect: the constant does not count toward order.
    scaled = TermSpec((FactorSpec(Const(2.0)), FactorSpec(Lookup('x'))))
    assert scaled.order == 1
    assert not scaled.is_intercept


def test_ir_is_hashable_and_value_equal(process):
    g1 = process('y ~ x + z')
    g2 = process('y ~ x + z')
    assert g1 == g2
    assert hash(g1) == hash(g2)
    assert {g1, g2} == {g1}
