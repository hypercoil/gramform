# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
``nwx`` covariate program (full vocabulary, Phase 6)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The :data:`CovariateProgram` is a closed union of :data:`CovariateOp`. ``nwx``
*emits* a program; the engine materialises columns (spec decision 4) -- this
module holds **no array** and imports neither ``jax``/``nitrix`` nor
``narwhals``/``numpy``.

The confound vocabulary is parsed by the **`minimaltest`** grammar (the ``ply``
port of the confound / dataframe surface -- the source of truth, transcribed in
``docs/nwx/covariate-vocabulary.md``, **not** the deleted ``dfops.py``).
:func:`lower_covariates` reuses that grammar but applies a *pure, emit-only*
interpreter over its AST that builds the nested :data:`CovariateOp` tree --
unlike ``minimaltest.transform``, which materialises ``narwhals`` columns.

Wilkinson precedence (spec §6): where a glyph differs between the term algebra
and the confound vocabulary, the **Wilkinson** meaning wins, so the
confound layer contributes only its non-colliding operators -- ``^^`` (numeric
power, vs Wilkinson ``^`` crossing-power), ``d_``/``dd_`` (differences),
``v_``/``n_`` (component selection), ``I_`` (indicators), ``AND_``/``OR_``/
``NOT_`` (set reductions), ``:::`` (scatter).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Union

from gramform.core import Literal as CoreLiteral
from gramform.core import Primitive
from gramform.grammars.minimaltest.grammar import MinimalGrammar

# --- shorthand preprocessor expansions (from minimaltest) ------------------
# ``csf`` is intentionally absent: it is a passthrough column name, NOT a
# shorthand (a common slip the legacy README implied otherwise). An
# unrecognised name is a plain column lookup, never an error.
SHORTHAND_EXPANSIONS: dict[str, tuple[str, ...]] = {
    'wm': ('white_matter',),
    'gsr': ('global_signal',),
    'gs': ('global_signal',),
    'rps': (
        'trans_x',
        'trans_y',
        'trans_z',
        'rot_x',
        'rot_y',
        'rot_z',
    ),
    'fd': ('framewise_displacement',),
    'dv': ('std_dvars',),
    'acc': ('a_comp_cor',),
    'wcc': ('w_comp_cor',),
    'ccc': ('c_comp_cor',),
}


def is_shorthand(name: str) -> bool:
    """Whether ``name`` is a recognised confound shorthand (``csf`` is not)."""
    return name in SHORTHAND_EXPANSIONS


class NwxCovariateError(ValueError):
    """A malformed / unsupported confound (covariate) expression."""


# ---------------------------------------------------------------------------
# the closed-union CovariateOp (nested: an op's operands are leaf columns /
# shorthands or further ops). The engine consumes these; nobody adds a variant
# without an engine that understands it.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Column:
    """A passthrough column referenced by name (e.g. ``csf``)."""

    name: str


@dataclass(frozen=True)
class Shorthand:
    """A confound shorthand (e.g. ``rps``); the engine materialises its
    :attr:`expansion` columns."""

    name: str

    @property
    def expansion(self) -> tuple[str, ...]:
        return SHORTHAND_EXPANSIONS[self.name]


@dataclass(frozen=True)
class Derivative:
    """Backward (temporal) difference of confound columns (``d_`` exclusive,
    orders ``1..order``; ``dd_`` inclusive, orders ``0..order``)."""

    operands: tuple[CovariateOp, ...]
    order: int = 1
    inclusive: bool = False


@dataclass(frozen=True)
class Power:
    """Inclusive *numeric* power (the ``^^`` surface, orders ``1..order``) --
    kept distinct from the Wilkinson crossing-power ``^``."""

    operands: tuple[CovariateOp, ...]
    order: int = 1
    inclusive: bool = True


@dataclass(frozen=True)
class CompCorSelect:
    """aCompCor component selection: by cumulative ``variance`` (``v_``) or by
    ``count`` of leading components (``n_``)."""

    criterion: Literal['variance', 'count']
    value: float
    operands: tuple[CovariateOp, ...] = ()


@dataclass(frozen=True)
class Indicator:
    """A spike / indicator regressor: ``operand comparison threshold`` (the
    ``I_[fd > 0.5]`` surface)."""

    operand: CovariateOp
    comparison: Literal['eq', 'ne', 'lt', 'le', 'gt', 'ge']
    threshold: float


@dataclass(frozen=True)
class SetOp:
    """A set reduction over its operands: ``AND_`` / ``OR_`` / ``NOT_``."""

    kind: Literal['and', 'or', 'not']
    operands: tuple[CovariateOp, ...]


@dataclass(frozen=True)
class Scatter:
    """Per-timepoint scatter: a spike regressor per flagged frame (``:::``)."""

    operands: tuple[CovariateOp, ...]


CovariateOp = Union[
    Column,
    Shorthand,
    Derivative,
    Power,
    CompCorSelect,
    Indicator,
    SetOp,
    Scatter,
]
CovariateProgram = tuple[CovariateOp, ...]


# ---------------------------------------------------------------------------
# emit-only lowering: a confound-formula string -> a CovariateProgram
# ---------------------------------------------------------------------------

_COMPARISONS: dict[str, Literal['eq', 'ne', 'lt', 'le', 'gt', 'ge']] = {
    'CONDITION_EQUAL': 'eq',
    'CONDITION_NOT_EQUAL': 'ne',
    'CONDITION_LESS': 'lt',
    'CONDITION_LESS_EQUAL': 'le',
    'CONDITION_GREATER': 'gt',
    'CONDITION_GREATER_EQUAL': 'ge',
}


@lru_cache(maxsize=1)
def _grammar() -> MinimalGrammar:
    """The confound-vocabulary grammar (built once)."""
    return MinimalGrammar()


def lower_covariates(formula: str) -> CovariateProgram:
    """Parse a confound-formula string into an emit-only ``CovariateProgram``.

    A top-level ``+`` (concatenation) yields several program ops; an operator
    (``^^`` / ``dd_`` / ``v_`` / ``I_`` / ``OR_`` / ``:::`` / ...) yields one
    nested op. Shorthands become :class:`Shorthand` ops (``csf`` is a
    :class:`Column`). Raises :class:`NwxCovariateError` on an unsupported form.
    """
    return _lower(_grammar().parse(formula))


def _lower(node: object) -> tuple[CovariateOp, ...]:
    if not isinstance(node, Primitive):
        raise NwxCovariateError(f'unexpected covariate node {node!r}')
    name = node.name
    if name == 'VARIABLE':
        col = node.get_parameters()
        return (Shorthand(col),) if is_shorthand(col) else (Column(col),)
    if name == 'CONCATENATE':
        out: list[CovariateOp] = []
        for child in node.parameters:
            out.extend(_lower(child))
        return tuple(out)
    if name == 'POWER':
        arg, order_node = node.parameters
        order, inclusive = _order(order_node)
        return (Power(_lower(arg), order=order, inclusive=inclusive),)
    if name == 'BACKDIFF':
        arg, order_node = node.parameters
        order, inclusive = _order(order_node)
        return (Derivative(_lower(arg), order=order, inclusive=inclusive),)
    if name == 'CUMUL_VAR':
        return (CompCorSelect('variance', _value(node.get_parameters())),)
    if name == 'FIRST_N':
        return (CompCorSelect('count', _value(node.get_parameters())),)
    if name == 'INDICATOR':
        return (_indicator(node.get_parameters()),)
    if name == 'UNION_REDUCE':
        return (SetOp('or', _lower(node.get_parameters())),)
    if name == 'INTERSECTION_REDUCE':
        return (SetOp('and', _lower(node.get_parameters())),)
    if name == 'NEGATION':
        return (SetOp('not', _lower(node.get_parameters())),)
    if name == 'SCATTER':
        return (Scatter(_lower(node.get_parameters())),)
    raise NwxCovariateError(f'unsupported covariate operator {name!r}')


def _order(node: object) -> tuple[int, bool]:
    """A ``^^``/``dd_`` order node -> ``(order, inclusive)``. The grammar maps
    the inclusive forms to a ``RANGE`` (``^^n`` -> ``1..n``, ``dd_[n]`` ->
    ``0..n``); a bare literal is the exclusive form."""
    if isinstance(node, Primitive) and node.name == 'RANGE':
        _lo, hi = node.parameters
        return int(_value(hi)), True
    return int(_value(node)), False


def _value(node: object) -> float:
    if isinstance(node, CoreLiteral) and isinstance(node.value, (int, float)):
        return node.value
    raise NwxCovariateError(f'expected a numeric value, got {node!r}')


def _indicator(cond: object) -> Indicator:
    if isinstance(cond, Primitive) and cond.name in _COMPARISONS:
        left, right = cond.parameters
        operand = _lower(left)
        if len(operand) != 1:
            raise NwxCovariateError(
                'an indicator compares a single column to a threshold'
            )
        return Indicator(
            operand=operand[0],
            comparison=_COMPARISONS[cond.name],
            threshold=_value(right),
        )
    raise NwxCovariateError(
        f'unsupported indicator condition {getattr(cond, "name", cond)!r}'
    )
