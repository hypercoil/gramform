# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Phase-3 random effects (lme4 bar-in-parens).

Each surface is asserted to emit the expected frozen ``RandomEffectSpec``(s),
routed structurally into ``ModelSpec.random`` (kept out of ``fixed``). Covers
the §4.2 table (scalar / unstructured / diagonal; slope-only; nesting;
interaction grouping; crossing), the implicit-intercept rule, the ``||``-vs-``|``
longest-match lexer rule, reserved-name disambiguation (a bare ``g`` is a
lookup; ``g`` is a grouping factor only after a bar), and the backend-awareness
warnings. The parser-conflict gate over ``NwxGrammar`` lives in
``test_parser_conflicts.py``.
"""

import warnings

import pytest

from gramform.grammars.nwx.grammar import NwxGrammar
from gramform.grammars.nwx.spec import (
    INTERCEPT,
    FactorSpec,
    GroupingSpec,
    Lookup,
    RandomEffectSpec,
    Relation,
    Structure,
    TermSpec,
)
from gramform.grammars.nwx.transform import NwxError, get_processor
from gramform.grammars.nwx.transform_ranef import BackendWarning


@pytest.fixture(scope='module')
def process():
    return get_processor()


def L(name: str) -> TermSpec:
    """A single-factor lookup term."""
    return TermSpec((FactorSpec(Lookup(name)),))


def G(*names: str, relation: Relation = Relation.SINGLE) -> GroupingSpec:
    return GroupingSpec(
        factors=tuple(FactorSpec(Lookup(n)) for n in names),
        relation=relation,
    )


def parse(process, formula: str):
    """Process a formula, silencing the backend-awareness warnings (asserted
    separately) so golden comparisons stay uncluttered."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', BackendWarning)
        return process(formula)


def random_of(process, formula: str) -> tuple[RandomEffectSpec, ...]:
    return parse(process, formula).nodes[0].spec.random


# ---------------------------------------------------------------------------
# the §4.2 surface -> spec table
# ---------------------------------------------------------------------------


def test_random_intercept_scalar(process):
    assert random_of(process, 'y ~ x + (1|g)') == (
        RandomEffectSpec(
            group=G('g'), terms=(INTERCEPT,), structure=Structure.SCALAR
        ),
    )


def test_random_effect_kept_out_of_fixed(process):
    # The bar contributes nothing to the fixed design (only x + the intercept).
    spec = parse(process, 'y ~ x + (1|g)').nodes[0].spec
    assert spec.fixed == (INTERCEPT, L('x'))


@pytest.mark.parametrize('formula', ['y ~ (1+x|g)', 'y ~ (x|g)'])
def test_correlated_slope_is_unstructured(process, formula):
    # `(x|g)` carries the implicit intercept, so both forms are (1, x).
    assert random_of(process, formula) == (
        RandomEffectSpec(
            group=G('g'),
            terms=(INTERCEPT, L('x')),
            structure=Structure.UNSTRUCTURED,
        ),
    )


@pytest.mark.parametrize('formula', ['y ~ (0+x|g)', 'y ~ (x-1|g)'])
def test_slope_only_no_intercept(process, formula):
    # `0`/`-1` on the bar LHS suppresses the implicit intercept; one term left.
    assert random_of(process, formula) == (
        RandomEffectSpec(
            group=G('g'), terms=(L('x'),), structure=Structure.SCALAR
        ),
    )


def test_uncorrelated_is_diagonal(process):
    assert random_of(process, 'y ~ (1+x||g)') == (
        RandomEffectSpec(
            group=G('g'),
            terms=(INTERCEPT, L('x')),
            structure=Structure.DIAGONAL,
        ),
    )


def test_nested_expands_to_two_specs(process):
    # `(1|g1/g2)` -> (1|g1) + (1|g1:g2): a single factor then the interaction.
    assert random_of(process, 'y ~ (1|g1/g2)') == (
        RandomEffectSpec(
            group=G('g1'), terms=(INTERCEPT,), structure=Structure.SCALAR
        ),
        RandomEffectSpec(
            group=G('g1', 'g2', relation=Relation.INTERACTION),
            terms=(INTERCEPT,),
            structure=Structure.SCALAR,
        ),
    )


def test_deep_nesting(process):
    # `(1|g1/g2/g3)` -> g1, g1:g2, g1:g2:g3.
    groups = [r.group for r in random_of(process, 'y ~ (1|g1/g2/g3)')]
    assert groups == [
        G('g1'),
        G('g1', 'g2', relation=Relation.INTERACTION),
        G('g1', 'g2', 'g3', relation=Relation.INTERACTION),
    ]


def test_interaction_grouping_is_one_component(process):
    # `(1|g1:g2)` is a SINGLE interaction grouping factor (one variance
    # component over the cells), NOT crossing.
    assert random_of(process, 'y ~ (1|g1:g2)') == (
        RandomEffectSpec(
            group=G('g1', 'g2', relation=Relation.INTERACTION),
            terms=(INTERCEPT,),
            structure=Structure.SCALAR,
        ),
    )


def test_crossing_is_multiple_specs(process):
    # Genuine crossing is two bars -> two specs, never one 'crossed' relation.
    rand = random_of(process, 'y ~ x + (1|s1) + (1|s2)')
    assert [r.group for r in rand] == [G('s1'), G('s2')]
    assert all(r.structure is Structure.SCALAR for r in rand)


def test_random_effect_only_rhs_keeps_fixed_intercept(process):
    # lme4 `y ~ (1|g)` still has a fixed intercept.
    spec = parse(process, 'y ~ (1|g)').nodes[0].spec
    assert spec.fixed == (INTERCEPT,)
    assert spec.random == (
        RandomEffectSpec(
            group=G('g'), terms=(INTERCEPT,), structure=Structure.SCALAR
        ),
    )


def test_random_effect_in_frame(process):
    # A bar inside a `[ ... ]` frame binds to that frame's node, not the root.
    g = parse(process, 'y ~ x + [z ~ w + (1|g)]')
    assert g.nodes[0].spec.random == ()
    assert g.nodes[1].spec.random == (
        RandomEffectSpec(
            group=G('g'), terms=(INTERCEPT,), structure=Structure.SCALAR
        ),
    )


# ---------------------------------------------------------------------------
# malformed groupings are rejected (a grouping is a factor / `:` / `/` only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('formula', ['y ~ (1|g1*g2)', 'y ~ (1|2)'])
def test_unsupported_grouping_raises(process, formula):
    with pytest.raises(NwxError):
        process(formula)


# ---------------------------------------------------------------------------
# reserved-name disambiguation: names are special only after a bar / in a call
# ---------------------------------------------------------------------------


def test_bare_name_is_a_lookup_not_a_grouping(process):
    # `g` on a normal RHS is a fixed lookup; it is a grouping factor only
    # after a bar.
    spec = parse(process, 'y ~ g').nodes[0].spec
    assert spec.fixed == (INTERCEPT, L('g'))
    assert spec.random == ()


def test_smooth_name_is_a_plain_lookup_in_phase3(process):
    # `s` is reserved only in call position (Phase 4); bare, it is a lookup.
    spec = parse(process, 'y ~ s').nodes[0].spec
    assert spec.fixed == (INTERCEPT, L('s'))


def test_same_name_fixed_and_grouping(process):
    # `g` appears both as a fixed lookup and as a grouping factor.
    spec = parse(process, 'y ~ g + (1|g)').nodes[0].spec
    assert spec.fixed == (INTERCEPT, L('g'))
    assert spec.random == (
        RandomEffectSpec(
            group=G('g'), terms=(INTERCEPT,), structure=Structure.SCALAR
        ),
    )


# ---------------------------------------------------------------------------
# lexer: `||` out-ranks `|` by longest match (R7)
# ---------------------------------------------------------------------------


def _lex_types(grammar, text):
    grammar.input(text)
    types = []
    while True:
        tok = grammar._lexer.token()
        if tok is None:
            break
        types.append(tok.type)
    return types


def test_double_bar_lexes_as_ranef_uncorr():
    grammar = NwxGrammar()
    assert _lex_types(grammar, 'x || g') == ['NAME', 'RANEF_UNCORR', 'NAME']


def test_single_bar_lexes_as_parts_separator():
    grammar = NwxGrammar()
    assert _lex_types(grammar, 'a | b') == ['NAME', 'PARTS_SEPARATOR', 'NAME']


def test_residual_operator_still_lexes():
    # `~|` must still out-rank `|` (it is longer); the new `||` does not steal
    # its prefix.
    grammar = NwxGrammar()
    assert _lex_types(grammar, 'y ~| n') == [
        'NAME',
        'LHS_RESIDUAL_RHS',
        'NAME',
    ]


# ---------------------------------------------------------------------------
# backend-awareness warnings (spec §7 / nitrix v3 §1.1)
# ---------------------------------------------------------------------------


def test_scalar_single_effect_does_not_warn(process):
    with warnings.catch_warnings():
        warnings.simplefilter('error', BackendWarning)
        process('y ~ x + (1|g)')


def test_slope_only_does_not_warn(process):
    # A random slope without its fixed effect is legal (lme4 permits it) and a
    # single scalar component -> no warning.
    with warnings.catch_warnings():
        warnings.simplefilter('error', BackendWarning)
        process('y ~ (0+x|g)')


@pytest.mark.parametrize('formula', ['y ~ (1+x|g)', 'y ~ (1+x||g)'])
def test_non_scalar_structure_warns(process, formula):
    with pytest.warns(BackendWarning, match='lme_fit'):
        process(formula)


def test_nested_grouping_warns(process):
    with pytest.warns(BackendWarning, match='[Nn]ested'):
        process('y ~ (1|g1/g2)')


def test_interaction_grouping_does_not_warn(process):
    # One scalar component on an interaction grouping factor is shipped.
    with warnings.catch_warnings():
        warnings.simplefilter('error', BackendWarning)
        process('y ~ (1|g1:g2)')


# ---------------------------------------------------------------------------
# value-equality / hashability of the random-effect IR
# ---------------------------------------------------------------------------


def test_random_effect_ir_is_hashable_and_value_equal(process):
    g1 = parse(process, 'y ~ x + (1+x|g)')
    g2 = parse(process, 'y ~ x + (1+x|g)')
    assert g1 == g2
    assert hash(g1) == hash(g2)
