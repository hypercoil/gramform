# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
PoC term-set parity.

The ``formulaic`` oracle is not IR-comparable, so a ``TermSpec`` ->
``formulaic.Term`` bijection helper is provided (a test helper, not assumed).
For each term-algebra expression, the ``spec`` interpreter's RHS term set is
converted to ``formulaic`` terms and compared against ``formulaic``'s own parse
(the same ground truth the Wilkinson PoC is tested against). This pins
dedup / intercept / interaction-expansion parity with the PoC.
"""

import formulaic
import pytest
from formulaic.parser.parser import DefaultFormulaParser
from formulaic.parser.types import Factor, Term

from gramform.grammars.nwx.spec import (
    Const,
    FactorSpec,
    Lookup,
    PyExpr,
    TermSpec,
)
from gramform.grammars.nwx.transform import get_processor


def termspec_to_term(term: TermSpec) -> Term:
    """Bijection helper: an ``nwx`` ``TermSpec`` -> a ``formulaic.Term``."""
    factors = []
    for factor in term.factors:
        source = factor.source
        if isinstance(source, Lookup):
            factors.append(
                Factor(source.name, eval_method=Factor.EvalMethod.LOOKUP)
            )
        elif isinstance(source, Const):
            value = source.value
            text = str(int(value)) if value.is_integer() else str(value)
            factors.append(
                Factor(text, eval_method=Factor.EvalMethod.LITERAL)
            )
        elif isinstance(source, PyExpr):
            factors.append(
                Factor(source.code, eval_method=Factor.EvalMethod.PYTHON)
            )
        else:  # pragma: no cover - parity exprs use only the three above
            raise TypeError(f'no formulaic term for {source!r}')
    return Term(factors=factors)


def _oracle(expr: str) -> formulaic.Formula:
    return formulaic.Formula(
        expr,
        _parser=DefaultFormulaParser(
            feature_flags=DefaultFormulaParser.FeatureFlags.ALL
        ),
    )


@pytest.fixture(scope='module')
def process():
    return get_processor()


def _check(
    process,
    expr: str,
    oracle_expr: str,
    comparison: str = 'exact',
) -> None:
    # `oracle_expr` is the formulaic-equivalent of `expr` (they differ only
    # where our `+`-associativity / grouping is flattened, as in the PoC's own
    # test mappings).
    graph = process(f'y ~ {expr}')
    mine = [termspec_to_term(t) for t in graph.nodes[0].spec.fixed]
    theirs = list(_oracle(oracle_expr))
    if comparison == 'set':
        assert set(mine) == set(theirs), (
            f'{expr!r}: {sorted(map(str, mine))} != '
            f'{sorted(map(str, theirs))}'
        )
    else:
        assert formulaic.Formula(mine) == formulaic.Formula(theirs), (
            f'{expr!r}: {list(map(str, mine))} != {list(map(str, theirs))}'
        )


# (our_expr, formulaic_oracle_expr) -- mirrors the validated Wilkinson PoC
# mappings (`test_wilkinson_formulaic.py`), minus the function-call /
# structured forms (Phase 4+).
BASIC = [
    ('x + y', 'x + y'),
    ('x:y', 'x:y'),
    ('x^2', 'x^2'),
    ('dog + cat', 'dog + cat'),
    ('rat*dog', 'rat*dog'),
    ('cat:dog', 'cat:dog'),
    ('3:x:2', '6:x'),
    ('x * y + 2:(x + z)', 'y + x:y + 2:(x + z)'),
    ('a/b', 'a/b'),
    ('a/b/c', 'a/b/c'),
]


@pytest.mark.parametrize('expr,oracle_expr', BASIC)
def test_basic_parity(process, expr, oracle_expr):
    _check(process, expr, oracle_expr)


COMPLEX = [
    ('(rat*dog + cat:dog)^2', '(rat*dog + cat:dog)^2', 'exact'),
    (
        'dog + cat + (rat*dog + cat:dog)^2',
        'dog + cat + (rat*dog + cat:dog)^2',
        'exact',
    ),
    ('(x + (y + z + z:w)^2)^3', '(x + (y + z + z:w)^2)^3', 'set'),
    ('(x + y + y:z)^3', '(x + y + y:z)^3', 'set'),
    ('(a + b + c) / (m + n)', '(a + b + c) / (m + n)', 'set'),
]


@pytest.mark.parametrize('expr,oracle_expr,comparison', COMPLEX)
def test_complex_parity(process, expr, oracle_expr, comparison):
    _check(process, expr, oracle_expr, comparison)


# Intercept removal / re-insertion (`+`-associative, so `(y + 0)` flattens).
INTERCEPT_HANDLING = [
    ('x + y - x - 1', 'x + y - x - 1'),
    ('x - 1 - y - 0', 'x - 1 - y - 0'),
    ('x + (y + 0)', 'x + y + 0'),
]


@pytest.mark.parametrize('expr,oracle_expr', INTERCEPT_HANDLING)
def test_intercept_parity(process, expr, oracle_expr):
    _check(process, expr, oracle_expr)


def test_bijection_round_trip():
    # The helper is a faithful map for the three Phase-1 sources.
    term = TermSpec(
        (
            FactorSpec(Lookup('x')),
            FactorSpec(Const(2.0)),
            FactorSpec(PyExpr('np.log(z)')),
        )
    )
    rendered = termspec_to_term(term)
    assert {f.expr for f in rendered.factors} == {'x', '2', 'np.log(z)'}
