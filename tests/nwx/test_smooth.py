# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Phase-4 GAM/GAMM smooths (``s`` / ``te`` / ``ti`` / ``t2``).

Each call surface is asserted to emit the expected frozen ``SmoothSpec``,
routed structurally into ``ModelSpec.smooth`` (kept out of ``fixed``). Covers
basis mapping (incl. cyclic and the reserved GAMM-bridge bases), default
``k``/basis, the ``bs="re"`` slope-var-> ``by`` rule, tensor flags, ``fx``, the
reserved-name disambiguation (a bare ``s`` is a lookup; ``s(x)`` is a smooth,
R6), and the backend-awareness warnings.
"""

import warnings

import pytest

from gramform.grammars.nwx.spec import (
    INTERCEPT,
    BasisKind,
    FactorSpec,
    Lookup,
    SmoothSpec,
    TermSpec,
)
from gramform.grammars.nwx.transform import NwxError, get_processor
from gramform.grammars.nwx.transform_ranef import BackendWarning


@pytest.fixture(scope='module')
def process():
    return get_processor()


def L(name: str) -> TermSpec:
    return TermSpec((FactorSpec(Lookup(name)),))


def F(name: str) -> FactorSpec:
    return FactorSpec(Lookup(name))


def parse(process, formula: str):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', BackendWarning)
        return process(formula)


def smooth_of(process, formula: str) -> tuple[SmoothSpec, ...]:
    return parse(process, formula).nodes[0].spec.smooth


# ---------------------------------------------------------------------------
# the §4.3 surface -> SmoothSpec table
# ---------------------------------------------------------------------------


def test_default_smooth(process):
    assert smooth_of(process, 'y ~ s(age)') == (
        SmoothSpec(covariates=(F('age'),), basis=BasisKind.TPRS, k=10),
    )


def test_smooth_kept_out_of_fixed(process):
    spec = parse(process, 'y ~ x + s(age)').nodes[0].spec
    assert spec.fixed == (INTERCEPT, L('x'))
    assert len(spec.smooth) == 1


def test_parameterised_smooth(process):
    assert smooth_of(process, 'y ~ x + s(age, k=6, bs="cr", by=dx)') == (
        SmoothSpec(
            covariates=(F('age'),),
            basis=BasisKind.CR,
            k=6,
            by=F('dx'),
        ),
    )


def test_penalty_order_and_fx(process):
    assert smooth_of(process, 'y ~ s(age, m=1, fx=TRUE)') == (
        SmoothSpec(
            covariates=(F('age'),),
            basis=BasisKind.TPRS,
            k=10,
            penalty_order=1,
            fx=True,
        ),
    )


@pytest.mark.parametrize('name', ['te', 'ti', 't2'])
def test_tensor_smooths(process, name):
    assert smooth_of(process, f'y ~ {name}(x, z, k=5)') == (
        SmoothSpec(
            covariates=(F('x'), F('z')),
            basis=BasisKind.TENSOR,
            k=5,
            tensor=True,
        ),
    )


def test_cyclic_basis(process):
    assert smooth_of(process, 'y ~ s(time, bs="cc", k=8)') == (
        SmoothSpec(
            covariates=(F('time'),),
            basis=BasisKind.CC,
            k=8,
            cyclic=True,
        ),
    )


def test_re_basis_random_intercept(process):
    assert smooth_of(process, 'y ~ s(g, bs="re")') == (
        SmoothSpec(covariates=(F('g'),), basis=BasisKind.RE, k=10),
    )


def test_re_basis_slope_var_maps_to_by(process):
    # For `bs="re"/"fs"` the second positional arg is the slope variable -> by.
    assert smooth_of(process, 'y ~ s(g, x, bs="re")') == (
        SmoothSpec(
            covariates=(F('g'),),
            basis=BasisKind.RE,
            k=10,
            by=F('x'),
        ),
    )


def test_by_kind_deferred(process):
    # by_kind (factor vs continuous) is data-dependent; nwx leaves it None.
    (s,) = smooth_of(process, 'y ~ s(age, by=dx) + dx')
    assert s.by == F('dx')
    assert s.by_kind is None


def test_smooth_in_frame(process):
    g = parse(process, 'y ~ x + [z ~ s(w)]')
    assert g.nodes[0].spec.smooth == ()
    assert g.nodes[1].spec.smooth == (
        SmoothSpec(covariates=(F('w'),), basis=BasisKind.TPRS, k=10),
    )


def test_multiple_smooths(process):
    specs = smooth_of(process, 'y ~ s(age) + s(thickness) + x')
    assert [s.covariates[0].source.name for s in specs] == [
        'age',
        'thickness',
    ]
    assert parse(process, 'y ~ s(age) + s(thickness) + x').nodes[
        0
    ].spec.fixed == (INTERCEPT, L('x'))


# ---------------------------------------------------------------------------
# reserved-name disambiguation (R6): `s` special only in call position
# ---------------------------------------------------------------------------


def test_bare_s_is_a_lookup(process):
    spec = parse(process, 'y ~ s').nodes[0].spec
    assert spec.fixed == (INTERCEPT, L('s'))
    assert spec.smooth == ()


def test_s_call_is_a_smooth(process):
    spec = parse(process, 'y ~ s(s)').nodes[0].spec
    # a column literally named `s`, smoothed.
    assert spec.fixed == (INTERCEPT,)
    assert spec.smooth == (
        SmoothSpec(covariates=(F('s'),), basis=BasisKind.TPRS, k=10),
    )


def test_unknown_basis_raises(process):
    with pytest.raises(NwxError):
        process('y ~ s(age, bs="nonsense")')


# ---------------------------------------------------------------------------
# backend-awareness warnings (spec §7 / nitrix v3 §2, §3.1)
# ---------------------------------------------------------------------------


# Every nwx smooth basis ships in nitrix v3 (ps/cc/tp/te v1; cr/gp/mrf §3.2;
# the GAMM-bridge re/fs §2/§3.1), so no basis warns at parse.
@pytest.mark.parametrize(
    'bs', ['tp', 'ps', 'cc', 'cr', 'gp', 'mrf', 're', 'fs']
)
def test_basis_does_not_warn(process, bs):
    with warnings.catch_warnings():
        warnings.simplefilter('error', BackendWarning)
        process(f'y ~ s(age, bs="{bs}")')


def test_tensor_does_not_warn(process):
    with warnings.catch_warnings():
        warnings.simplefilter('error', BackendWarning)
        process('y ~ te(x, z)')


# ---------------------------------------------------------------------------
# value-equality / hashability
# ---------------------------------------------------------------------------


def test_smooth_ir_is_hashable_and_value_equal(process):
    g1 = parse(process, 'y ~ x + s(age, k=6, by=dx)')
    g2 = parse(process, 'y ~ x + s(age, k=6, by=dx)')
    assert g1 == g2
    assert hash(g1) == hash(g2)
