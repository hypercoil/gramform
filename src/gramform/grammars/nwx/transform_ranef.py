# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
``nwx`` random-effects interpreter (Phase 3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Lowers the ``RANDOM_EFFECT`` / ``GROUPING`` AST built by
:class:`~gramform.grammars.nwx.grammar.NwxGrammar` onto
:class:`~gramform.grammars.nwx.spec.RandomEffectSpec`. A disjoint file from the
core term algebra (:mod:`gramform.grammars.nwx.transform`), registered into the
same ``spec`` interpreter group; imports neither ``jax`` nor ``nitrix``.

Like ``noise()`` (Phase 2), a random effect is **not** a fixed term: the bar's
effects are routed structurally out of the fixed-term stream into a
``NwxState.random`` accumulator (returning no fixed terms), which
``LHS_RHS_STRUCTURE`` consumes into ``ModelSpec.random``.

Semantics (spec §4.2):

==========================  =========================  ====================
Surface                     Meaning                    IR
==========================  =========================  ====================
``(1|g)``                   random intercept           ``SCALAR``, terms (1,)
``(1+x|g)`` / ``(x|g)``     correlated intercept+slope  ``UNSTRUCTURED``
``(0+x|g)`` / ``(x-1|g)``   slope only                 ``SCALAR``, terms (x,)
``(1+x||g)``                uncorrelated               ``DIAGONAL``
``(1|g1/g2)``               nested                     two specs (g1, g1:g2)
``(1|g1:g2)``               interaction grouping       ``relation=INTERACTION``
==========================  =========================  ====================

The intercept is present unless ``0`` / ``-1`` sits on the bar LHS -- handled
by reusing the model's own intercept machinery
(:func:`add_intercept_to_formula`), so ``0+x`` / ``x-1`` behave identically to
the fixed-effect case. A random slope without its fixed effect is **legal**
(lme4 permits it), so no warning is raised for that. Genuine crossing
(``(1|g1)+(1|g2)``) is *multiple* specs (two bars), never one
``relation='crossed'`` -- that falls out naturally because each bar is its own
``factor``.
"""

import warnings

from gramform.core import Primitive
from gramform.grammars.nwx.spec import (
    BackendWarning,
    FactorSpec,
    GroupingSpec,
    Lookup,
    RandomEffectSpec,
    Relation,
    Structure,
    TermSpec,
)
from gramform.grammars.nwx.transform import (
    INTERPRETERS,
    NwxContext,
    NwxError,
    _coerce,
    to_terms,
)
from gramform.grammars.wilkinson.transform import add_intercept_to_formula

#: Re-exported for the smooth / directive modules and the Phase-3 tests; the
#: canonical definition lives in :mod:`gramform.grammars.nwx.spec`.
__all__ = ['BackendWarning', 'RANDOM_EFFECT_impl', 'GROUPING_impl']


# ---------------------------------------------------------------------------
# grouping: a raw term-algebra AST -> grouping-factor components
# ---------------------------------------------------------------------------


def _factor(ast: Primitive) -> FactorSpec:
    """A single grouping variable -> a :class:`FactorSpec`."""
    if getattr(ast, 'name', None) == 'VARIABLE':
        return FactorSpec(Lookup(ast.get_parameters()))
    raise NwxError(
        f'a random-effect grouping factor must be a variable, got '
        f'{getattr(ast, "name", ast)!r}'
    )


def _all_factors(ast: Primitive) -> tuple[FactorSpec, ...]:
    """Every distinct grouping variable in a grouping subtree, in order (the
    full nesting path that an interaction grouping factor spans)."""
    name = getattr(ast, 'name', None)
    if name == 'VARIABLE':
        return (_factor(ast),)
    if name in ('INTERACTION', 'NESTED'):
        out: list[FactorSpec] = []
        for child in ast.parameters:
            for f in _all_factors(child):
                if f not in out:
                    out.append(f)
        return tuple(out)
    raise NwxError(
        f'unsupported random-effect grouping operator {name!r}; use a '
        'variable, an interaction `g1:g2`, or nesting `g1/g2`'
    )


def _grouping_components(ast: Primitive) -> list[tuple[FactorSpec, ...]]:
    """A grouping AST -> the ordered list of grouping-factor sets, one per
    emitted random effect. ``g`` -> one single factor; ``g1:g2`` -> one
    interaction factor; ``g1/g2`` -> nesting expands to ``g1`` then ``g1:g2``
    (lme4 nesting)."""
    name = getattr(ast, 'name', None)
    if name == 'VARIABLE':
        return [(_factor(ast),)]
    if name == 'INTERACTION':
        return [_all_factors(ast)]
    if name == 'NESTED':
        left, right = ast.parameters
        prefix = _all_factors(left)
        components = list(_grouping_components(left))
        for rc in _grouping_components(right):
            merged = list(prefix)
            for f in rc:
                if f not in merged:
                    merged.append(f)
            components.append(tuple(merged))
        return components
    raise NwxError(
        f'unsupported random-effect grouping operator {name!r}; use a '
        'variable, an interaction `g1:g2`, or nesting `g1/g2`'
    )


def _structure(n_terms: int, correlated: bool) -> Structure:
    """The covariance structure implied by the bar form. A single term has one
    variance component (scalar); multiple terms are unstructured (``|``,
    correlated) or diagonal (``||``, uncorrelated)."""
    if n_terms <= 1:
        return Structure.SCALAR
    return Structure.UNSTRUCTURED if correlated else Structure.DIAGONAL


# ---------------------------------------------------------------------------
# the RANDOM_EFFECT operation
# ---------------------------------------------------------------------------


def RANDOM_EFFECT_impl(node: Primitive, context: NwxContext) -> NwxContext:
    expr_ast, grouping_node, correlated = node.parameters

    # lme4's implicit random intercept: prepend it unless `0`/`-1` suppresses
    # it. Reusing the model's intercept machinery makes `0+x` / `x-1` behave
    # exactly as in the fixed-effect case.
    expr_ast = add_intercept_to_formula(expr_ast, context)[0]
    context = expr_ast(context)
    terms: tuple[TermSpec, ...] = to_terms(_coerce(context.get_result()))

    structure = _structure(len(terms), bool(correlated))
    components = _grouping_components(grouping_node.get_parameters())

    specs = tuple(
        RandomEffectSpec(
            group=GroupingSpec(
                factors=factors,
                relation=(
                    Relation.INTERACTION
                    if len(factors) > 1
                    else Relation.SINGLE
                ),
            ),
            terms=terms,
            structure=structure,
        )
        for factors in components
    )

    # Backend awareness (spec §7 / nitrix v3 §1.1): only a single scalar effect
    # on a single (or interaction) grouping factor lowers onto the shipped
    # `reml_fit` (R1). Non-scalar structures (R2) and nested groupings (R3)
    # need the v3 `lme_fit` ladder.
    if structure is not Structure.SCALAR:
        warnings.warn(
            f'random-effect covariance structure {structure.value!r} lowers '
            'onto the nitrix v3 lme_fit ladder (FR §1.1 R2), not the shipped '
            'scalar reml_fit',
            BackendWarning,
            stacklevel=2,
        )
    if len(components) > 1:
        warnings.warn(
            'nested random effects lower onto the nitrix v3 lme_fit ladder '
            '(FR §1.1 R3), not the shipped scalar reml_fit',
            BackendWarning,
            stacklevel=2,
        )

    context = context.update_state(random=context.state.random + specs)
    # A random effect contributes no fixed terms.
    return context.with_result(())


def GROUPING_impl(node: Primitive, context: NwxContext) -> NwxContext:
    # The grouping AST is consumed structurally by RANDOM_EFFECT_impl; it is
    # never dispatched on its own. Guard against an accidental evaluation.
    raise NwxError(
        'a grouping factor was evaluated outside a random-effect context'
    )


INTERPRETERS.register_operation('build', 'RANDOM_EFFECT', RANDOM_EFFECT_impl)
INTERPRETERS.register_operation('build', 'GROUPING', GROUPING_impl)
