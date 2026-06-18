# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
``nwx`` grammar extensions (Phase 3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``NwxGrammar`` is the Wilkinson term algebra plus the ``nwx``-specific syntax.
Phase 3 adds the **lme4 bar-in-parens** random-effects idiom via
``RanefComponent``: a new token ``RANEF_UNCORR (||)`` and the productions

::

    factor   : LPAREN ranef RPAREN
    ranef    : expression PARTS_SEPARATOR grouping    # correlated / scalar
             | expression RANEF_UNCORR    grouping    # uncorrelated (diagonal)
    grouping : term                                   # g, g1/g2, g1:g2

A bar *inside parentheses* is the random-effects grouping operator; a bar at
the *blocks (top) level* (``blocks : blocks PARTS_SEPARATOR block``) is the
parts separator. The two never collide: the parts-bar production lives at
``blocks`` level, which is unreachable from inside ``LPAREN expression RPAREN``
(spec §4.2). The PR gate asserts the merged grammar is conflict-free
(``NwxGrammar().conflicts == ()``) -- PLY exposes no ``parser.conflicts``, so
``DynamicGrammar`` captures the construction ``errorlog`` instead.

``||`` and ``|`` are both function-free string tokens, so PLY orders them by
descending regex length (``\\|\\|`` before ``\\|``) -- maximal munch picks
``RANEF_UNCORR`` for ``||`` and ``PARTS_SEPARATOR`` for a lone ``|``. A lexer
test pins this.

The new AST primitives ``RANDOM_EFFECT`` / ``GROUPING`` are interpreted in the
disjoint :mod:`gramform.grammars.nwx.transform_ranef` module (registered into
the same ``spec`` interpreter group). This module is pure syntax: it imports
neither ``jax`` nor ``nitrix``.
"""

from dataclasses import dataclass
from typing import Tuple

from gramform.core import (
    DynamicGrammar,
    GrammarComponent,
    ProductionRule,
    Token,
    enter_group,
    unit_lift,
)
from gramform.core import (
    Primitive as CorePrimitive,
)
from gramform.grammars.wilkinson.grammar import (
    BasicOperatorsComponent,
    ExecutionComponent,
    LiteralTerminalsComponent,
    NamesComponent,
    StructureComponent,
    from_sequence,
)

# New AST primitives (the Wilkinson registry is closed, so these use the core
# ``Primitive`` directly; the interpreter dispatches purely by ``.name``).
RANDOM_EFFECT = CorePrimitive('RANDOM_EFFECT', is_associative=False)
GROUPING = CorePrimitive('GROUPING', is_associative=False)
PROGRAM_DIRECTIVES = CorePrimitive('PROGRAM_DIRECTIVES', is_associative=False)
FRAME_DIRECTIVES = CorePrimitive('FRAME_DIRECTIVES', is_associative=False)
PIPELINE = CorePrimitive('PIPELINE', is_associative=True)


def _directive_token(t, grammar):
    """A no-op lexer action: makes ``DIRECTIVE_BLOCK`` a *function* token so it
    out-ranks the string-defined ``EXECUTE`` (``{...}``), giving ``{{`` maximal
    munch over a single ``{``."""
    return t


def _ranef_bind(correlated: bool):
    """Bind a ``ranef`` production to a ``RANDOM_EFFECT`` node carrying the
    bar-LHS expression, the grouping, and whether the bar correlates the
    effects (``|`` correlated, ``||`` uncorrelated)."""

    def _inner(expr, _, grouping):
        return RANDOM_EFFECT.bind(expr, grouping, correlated)

    return _inner


@dataclass(frozen=True)
class RanefComponent(GrammarComponent):
    """Component for lme4 bar-in-parens random effects."""

    tokens: Tuple[Token, ...] = (
        # `||` (uncorrelated). Must out-rank `|`; both are function-free string
        # tokens, so PLY orders them by descending regex length automatically.
        Token(
            'RANEF_UNCORR',
            r'\|\|',
            precedence=from_sequence,
            category='STRUCTURE',
        ),
    )

    production_rules: Tuple[ProductionRule, ...] = (
        ProductionRule(
            'factor_random_effect',
            'factor : LPAREN ranef RPAREN',
            enter_group(),
        ),
        ProductionRule(
            'ranef_correlated',
            'ranef : expression PARTS_SEPARATOR grouping',
            _ranef_bind(True),
        ),
        ProductionRule(
            'ranef_uncorrelated',
            'ranef : expression RANEF_UNCORR grouping',
            _ranef_bind(False),
        ),
        ProductionRule(
            'grouping_term',
            'grouping : term',
            lambda term: GROUPING.bind(term),
        ),
    )


@dataclass(frozen=True)
class DirectiveComponent(GrammarComponent):
    """Component for the trailing ``{{ ... }}`` directive block (spec §4.5).

    The block is captured as one ``DIRECTIVE_BLOCK`` token (its inner text is
    parsed by :mod:`gramform.grammars.nwx.directives`), so the directive
    mini-language's ``:`` / ``=`` never reach the term-algebra lexer -- no
    collision, no separate lexer state needed. Its grammatical *position*
    carries the scope: a block after the whole formula binds to the implicit
    outermost node (``PROGRAM_DIRECTIVES``); a block inside a frame's brackets
    binds to that frame's node (``FRAME_DIRECTIVES``). The multi-level
    pipeline (graph-level placement) is added in a later component.
    """

    tokens: Tuple[Token, ...] = (
        Token(
            'DIRECTIVE_BLOCK',
            r'\{\{.*?\}\}',
            function=_directive_token,
            precedence=from_sequence,
            category='DIRECTIVE',
        ),
    )

    production_rules: Tuple[ProductionRule, ...] = (
        ProductionRule(
            'program_plain',
            'program : pipeline',
            unit_lift(),
        ),
        ProductionRule(
            'program_directives',
            'program : pipeline DIRECTIVE_BLOCK',
            lambda pipeline, block: PROGRAM_DIRECTIVES.bind(pipeline, block),
        ),
        ProductionRule(
            'frame_directives',
            'factor : LBRACKET formula DIRECTIVE_BLOCK RBRACKET',
            lambda _, formula, block, __: FRAME_DIRECTIVES.bind(
                formula, block
            ),
        ),
    )


@dataclass(frozen=True)
class PipelineComponent(GrammarComponent):
    """Component for the multi-level pipeline ``>>`` (spec §4.4).

    ``STAGE_PIPE (>>)`` connects frames/formulae into a graph of stages
    (BIDS-SM ``Nodes`` + ``Edges``). A single-stage pipeline is just a formula;
    a multi-stage pipeline is an associative ``PIPELINE`` of stages. The
    trailing directive block (handled by ``program``) is graph-level."""

    tokens: Tuple[Token, ...] = (
        Token(
            'STAGE_PIPE',
            r'\>\>',
            precedence=from_sequence,
            category='STRUCTURE',
        ),
    )

    production_rules: Tuple[ProductionRule, ...] = (
        ProductionRule(
            'pipeline_one',
            'pipeline : formula',
            unit_lift(),
        ),
        ProductionRule(
            'pipeline_seq',
            'pipeline : pipeline STAGE_PIPE formula',
            lambda left, _, right: PIPELINE.bind(left, right),
        ),
    )


class NwxGrammar(DynamicGrammar):
    """The ``nwx`` grammar: the Wilkinson components + ``nwx`` extensions.

    Phase 3 adds :class:`RanefComponent`; Phase 5 adds
    :class:`DirectiveComponent` (and, later, the multi-level pipeline). The
    start symbol is ``program`` = a ``formula`` plus an optional trailing
    directive block. Each merge is guarded by the parser-conflict gate.
    """

    def __init__(self):
        super().__init__(
            start_symbol='program',
            components=(
                LiteralTerminalsComponent(),
                BasicOperatorsComponent(),
                NamesComponent(),
                ExecutionComponent(),
                StructureComponent(),
                RanefComponent(),
                DirectiveComponent(),
                PipelineComponent(),
            ),
        )
