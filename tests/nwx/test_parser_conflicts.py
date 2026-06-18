# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Parser-conflict gate. Every built grammar must be free of LALR shift/reduce
and reduce/reduce conflicts. ``DynamicGrammar`` captures PLY's construction
warnings (PLY exposes no ``parser.conflicts`` attribute) and surfaces them via
the ``conflicts`` property. This gate guards the nwx grammar extensions
(bar-in-parens random effects, directive block, pipeline) as they are added.
"""

import pytest

from gramform.grammars.minimaltest.grammar import MinimalGrammar
from gramform.grammars.nwx.grammar import NwxGrammar
from gramform.grammars.wilkinson.grammar import WilkinsonGrammar

GRAMMARS = [WilkinsonGrammar, MinimalGrammar, NwxGrammar]


@pytest.mark.parametrize(
    'grammar_cls',
    GRAMMARS,
    ids=[g.__name__ for g in GRAMMARS],
)
def test_grammar_has_no_conflicts(grammar_cls):
    grammar = grammar_cls()
    assert grammar.conflicts == (), (
        f'{grammar_cls.__name__} has LALR conflicts:\n'
        + '\n'.join(grammar.conflicts)
    )
