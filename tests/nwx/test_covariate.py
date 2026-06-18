# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for the minimal Phase-1 ``CovariateProgram`` ops (the 36P subset).
"""

from gramform.grammars.nwx.covariate import (
    SHORTHAND_EXPANSIONS,
    Derivative,
    Power,
    Shorthand,
    is_shorthand,
)


def test_rps_expands_to_six_columns():
    op = Shorthand('rps')
    assert op.expansion == (
        'trans_x',
        'trans_y',
        'trans_z',
        'rot_x',
        'rot_y',
        'rot_z',
    )
    assert len(op.expansion) == 6


def test_csf_is_not_a_shorthand():
    # `csf` is a passthrough column name, NOT a shorthand (a common slip).
    assert not is_shorthand('csf')
    assert 'csf' not in SHORTHAND_EXPANSIONS


def test_known_shorthands():
    assert is_shorthand('wm')
    assert is_shorthand('gsr')
    assert is_shorthand('acc')
    assert Shorthand('wm').expansion == ('white_matter',)
    assert Shorthand('gsr').expansion == ('global_signal',)


def test_derivative_defaults_and_inclusive():
    d = Derivative(operands=('rps',))
    assert d.order == 1 and d.inclusive is False
    dd = Derivative(operands=('rps', 'wm'), order=2, inclusive=True)
    assert dd.order == 2 and dd.inclusive is True


def test_power_is_inclusive_by_default():
    # `^^` keeps the original (orders 1..n), distinct from Wilkinson `^`.
    p = Power(operands=('rps',), order=2)
    assert p.inclusive is True and p.order == 2


def test_ops_are_frozen_hashable_value_equal():
    assert Shorthand('rps') == Shorthand('rps')
    assert hash(Power(('a',), 2)) == hash(Power(('a',), 2))
    assert Derivative(('a',), 1) != Derivative(('a',), 2)
