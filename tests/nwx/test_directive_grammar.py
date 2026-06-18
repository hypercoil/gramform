# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Phase-5b grammar-integrated directives (spec §4.5).

The trailing ``{{ ... }}`` block is now captured by the grammar as one
``DIRECTIVE_BLOCK`` token (its text parsed by the directive mini-parser) and
attaches by *position*: a block after the whole formula binds to the outermost
node (``PROGRAM_DIRECTIVES``); a block inside a frame's brackets binds to that
frame's node (``FRAME_DIRECTIVES``). Because the block is opaque to the term
lexer, the directive ``:`` / ``=`` never collide with the term algebra's
``INTERACTION_ONLY`` / ``ASSIGN`` -- the isolation the spec asked an exclusive
lexer state to provide, achieved by a single blob token.
"""

import warnings

import pytest

from gramform.grammars.nwx.grammar import NwxGrammar
from gramform.grammars.nwx.spec import (
    INTERCEPT,
    BackendWarning,
    Combine,
    FactorSpec,
    Family,
    Level,
    Lookup,
    TermSpec,
)
from gramform.grammars.nwx.transform import NwxError, get_processor


@pytest.fixture(scope='module')
def process():
    return get_processor()


def L(name: str) -> TermSpec:
    return TermSpec((FactorSpec(Lookup(name)),))


def parse(process, formula: str):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', BackendWarning)
        return process(formula)


# ---------------------------------------------------------------------------
# outermost directives, via the grammar (not the old textual split)
# ---------------------------------------------------------------------------


def test_outermost_directives(process):
    spec = parse(process, 'y ~ x {{ family=binomial; estimator=reml }}').nodes[
        0
    ].spec
    assert spec.family.family is Family.BINOMIAL
    assert spec.estimation.estimator == 'reml'


def test_no_directives_is_plain(process):
    spec = process('y ~ x + z').nodes[0].spec
    assert spec.fixed == (INTERCEPT, L('x'), L('z'))
    assert spec.estimands == ()


# ---------------------------------------------------------------------------
# exclusive isolation: directive `:`/`=` do not collide with the term algebra
# ---------------------------------------------------------------------------


def test_interaction_colon_and_contrast_colon_coexist(process):
    # `a:b` is a term interaction (`:`); the `contrasts:` clause also uses `:`
    # -- the directive block is opaque to the term lexer, so neither leaks.
    g = parse(process, 'y ~ a:b {{ contrasts: c = a (t) }}')
    spec = g.nodes[0].spec
    ab = TermSpec((FactorSpec(Lookup('a')), FactorSpec(Lookup('b'))))
    assert ab in spec.fixed
    assert spec.estimands[0].name == 'c'
    assert spec.estimands[0].weights == (('a', 1.0),)


def test_directive_equals_does_not_become_a_term(process):
    # The `=` inside the block is key/value, not a stray token in the design.
    spec = parse(process, 'y ~ x {{ family=poisson }}').nodes[0].spec
    assert spec.fixed == (INTERCEPT, L('x'))
    assert spec.family.family is Family.POISSON


def test_execute_block_still_works_with_directives(process):
    # `{...}` is EXECUTE (a Python transform); `{{...}}` is a directive block.
    spec = parse(process, 'y ~ x + {np.log(z)} {{ family=poisson }}').nodes[
        0
    ].spec
    assert spec.family.family is Family.POISSON
    assert any(
        getattr(f.source, 'code', None) == 'np.log(z)'
        for t in spec.fixed
        for f in t.factors
    )


# ---------------------------------------------------------------------------
# frame-level directives (the new capability over the textual split)
# ---------------------------------------------------------------------------


def test_frame_level_directives(process):
    g = parse(process, 'y ~ x + [z ~ w {{ estimator=reml; level=subject }}]')
    root, frame = g.nodes[0], g.nodes[1]
    # root is unaffected by the frame's directives
    assert root.spec.estimation.estimator == 'ols'
    assert root.level is Level.DATASET
    # the frame node carries them
    assert frame.spec.estimation.estimator == 'reml'
    assert frame.level is Level.SUBJECT


def test_frame_combine_and_group_by(process):
    g = parse(
        process,
        'y ~ x + [cope ~ cond {{ group_by=subject; combine=mixed }}]',
    )
    frame = g.nodes[1]
    assert frame.group_by == ('subject',)
    assert frame.combine is Combine.MIXED


def test_frame_residualise_validation_runs(process):
    # The frame sub-node's residualise spec is validated too: nonaggressive
    # without a signal() set is a hard error even inside a frame.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', BackendWarning)
        with pytest.raises(NwxError, match='nonaggressive requires a signal'):
            process('y ~ x + [bold ~| n {{ residualise=nonaggressive }}]')


# ---------------------------------------------------------------------------
# placement: a directive block is only valid in the two grammar positions
# ---------------------------------------------------------------------------


def test_misplaced_directive_block_is_a_parse_error(process):
    with pytest.raises(ValueError):
        process('y ~ {{ family=binomial }} + x')


# ---------------------------------------------------------------------------
# lexer: `{{...}}` is one DIRECTIVE_BLOCK; `{...}` is EXECUTE
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


def test_directive_block_lexes_as_one_token():
    grammar = NwxGrammar()
    assert _lex_types(grammar, '{{ family=x; se=robust }}') == [
        'DIRECTIVE_BLOCK'
    ]


def test_single_brace_is_execute():
    grammar = NwxGrammar()
    assert _lex_types(grammar, '{np.log(z)}') == ['EXECUTE']
