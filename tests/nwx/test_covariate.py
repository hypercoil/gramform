# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Phase-6 ``CovariateProgram`` — the full confound vocabulary, emit-only.

``lower_covariates`` parses a confound-formula string (via the ``dataframe``
grammar) into a nested closed-union ``CovariateProgram``; nwx holds no array.
Covers the 36P idiom, aCompCor selection, indicator / scatter spikes, set
reductions, and the shorthand table (``csf`` is a passthrough column).
"""

import pytest

from gramform.grammars.nwx.covariate import (
    SHORTHAND_EXPANSIONS,
    Column,
    CompCorSelect,
    Derivative,
    Indicator,
    NwxCovariateError,
    Power,
    Scatter,
    SetOp,
    Shorthand,
    is_shorthand,
    lower_covariates,
)


# ---------------------------------------------------------------------------
# shorthands
# ---------------------------------------------------------------------------


def test_rps_expands_to_six_columns():
    assert Shorthand('rps').expansion == (
        'trans_x',
        'trans_y',
        'trans_z',
        'rot_x',
        'rot_y',
        'rot_z',
    )


def test_csf_is_not_a_shorthand():
    assert not is_shorthand('csf')
    assert 'csf' not in SHORTHAND_EXPANSIONS


def test_known_shorthands():
    assert is_shorthand('wm') and is_shorthand('gsr') and is_shorthand('acc')
    assert Shorthand('gsr').expansion == ('global_signal',)


# ---------------------------------------------------------------------------
# lower_covariates: the documented vocabulary
# ---------------------------------------------------------------------------


def test_concatenation_mixes_shorthands_and_columns():
    # `csf` is a passthrough Column; the rest are Shorthands.
    assert lower_covariates('rps + wm + csf + gsr') == (
        Shorthand('rps'),
        Shorthand('wm'),
        Column('csf'),
        Shorthand('gsr'),
    )


def test_36p_idiom():
    # (dd_(...))^^2: inclusive backward difference then inclusive numeric power.
    prog = lower_covariates('(dd_[1](rps+wm+csf+gsr))^^2')
    assert prog == (
        Power(
            operands=(
                Derivative(
                    operands=(
                        Shorthand('rps'),
                        Shorthand('wm'),
                        Column('csf'),
                        Shorthand('gsr'),
                    ),
                    order=1,
                    inclusive=True,
                ),
            ),
            order=2,
            inclusive=True,
        ),
    )


def test_exclusive_vs_inclusive_power():
    # `^^` is inclusive (keeps the original); plain `^` is exclusive numeric.
    (incl,) = lower_covariates('rps^^2')
    assert isinstance(incl, Power) and incl.inclusive is True
    (excl,) = lower_covariates('rps^2')
    assert isinstance(excl, Power) and excl.inclusive is False


def test_acompcor_variance_and_count():
    assert lower_covariates('v_[5]') == (CompCorSelect('variance', 5),)
    assert lower_covariates('n_[6]') == (CompCorSelect('count', 6),)


def test_indicator_spike():
    assert lower_covariates('I_[fd>0.5]') == (
        Indicator(operand=Shorthand('fd'), comparison='gt', threshold=0.5),
    )


def test_scatter_per_timepoint():
    assert lower_covariates(':::I_[dv>1.5]') == (
        Scatter(
            operands=(
                Indicator(
                    operand=Shorthand('dv'), comparison='gt', threshold=1.5
                ),
            )
        ),
    )


def test_set_reductions():
    (or_op,) = lower_covariates('OR_(I_[fd>0.5] + I_[dv>1.5])')
    assert isinstance(or_op, SetOp) and or_op.kind == 'or'
    assert len(or_op.operands) == 2
    (and_op,) = lower_covariates('AND_(acc + wcc)')
    assert and_op == SetOp('and', (Shorthand('acc'), Shorthand('wcc')))


def test_comparison_operators():
    for surface, code in [
        ('I_[x=1]', 'eq'),
        ('I_[x<1]', 'lt'),
        ('I_[x>=1]', 'ge'),
    ]:
        (op,) = lower_covariates(surface)
        assert isinstance(op, Indicator) and op.comparison == code


# ---------------------------------------------------------------------------
# emit-only / value semantics
# ---------------------------------------------------------------------------


def test_program_is_frozen_hashable_value_equal():
    p1 = lower_covariates('(dd_[1](rps+wm))^^2')
    p2 = lower_covariates('(dd_[1](rps+wm))^^2')
    assert p1 == p2
    assert hash(p1) == hash(p2)
    assert {p1, p2} == {p1}


def test_bare_comparison_needs_an_indicator():
    # A comparison parses but is not a standalone covariate op (wrap in `I_`).
    with pytest.raises(NwxCovariateError):
        lower_covariates('fd > 0.5')
