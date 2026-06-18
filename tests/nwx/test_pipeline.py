# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Phase-5c multi-level pipeline (``>>``) — spec §4.4 / §11.

``>>`` connects stages into a ``ModelGraph`` of nodes + dataflow edges
(BIDS-SM-shaped). Each ``[...]`` stage becomes a node (not an inline ``_hat``
referent); a ``.`` on a downstream stage's LHS is the inbound cope from the
prior stage (the position-sensitive meaning of ``.``, spec §8); an edge carries
the upstream cope/varcope; a trailing ``{{ ... }}`` after the whole pipeline is
graph-level (its inference).
"""

import warnings

import pytest

from gramform.grammars.nwx.grammar import NwxGrammar
from gramform.grammars.nwx.spec import (
    INTERCEPT,
    BackendWarning,
    Combine,
    FactorSpec,
    Level,
    Lookup,
    Referent,
    TermSpec,
)
from gramform.grammars.nwx.transform import get_processor


@pytest.fixture(scope='module')
def process():
    return get_processor()


def L(name: str) -> TermSpec:
    return TermSpec((FactorSpec(Lookup(name)),))


def parse(process, formula: str):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', BackendWarning)
        warnings.simplefilter('ignore', UserWarning)
        return process(formula)


# ---------------------------------------------------------------------------
# graph structure
# ---------------------------------------------------------------------------


def test_two_stage_pipeline_nodes_and_edge(process):
    g = parse(process, '[cope ~ cond] >> [. ~ 1]')
    assert [n.name for n in g.nodes] == ['stage0', 'stage1']
    assert len(g.edges) == 1
    e = g.edges[0]
    assert (e.source, e.dest) == ('stage0', 'stage1')
    assert e.carry.quantities == 'cope_varcope'


def test_stage_is_a_node_not_a_referent(process):
    # A top-level `[...]` stage becomes a graph node; its design is the stage's
    # own (no `_hat` referent injected into a parent).
    g = parse(process, '[cope ~ cond] >> [. ~ 1]')
    assert g.nodes[0].spec.fixed == (INTERCEPT, L('cond'))
    assert g.nodes[0].spec.response.terms == (L('cope'),)


def test_inbound_cope_referent(process):
    # `.` on the downstream LHS resolves to the prior stage's cope.
    g = parse(process, '[a ~ b] >> [. ~ 1]')
    resp = g.nodes[1].spec.response.terms
    assert resp == (
        TermSpec((FactorSpec(Referent(stage='stage0', kind='cope')),)),
    )


def test_three_stage_chain(process):
    g = parse(process, '[a ~ b] >> [. ~ 1] >> [. ~ 1]')
    assert [n.name for n in g.nodes] == ['stage0', 'stage1', 'stage2']
    assert [(e.source, e.dest) for e in g.edges] == [
        ('stage0', 'stage1'),
        ('stage1', 'stage2'),
    ]
    # each `.` chains to the immediately prior stage
    assert g.nodes[2].spec.response.terms == (
        TermSpec((FactorSpec(Referent(stage='stage1', kind='cope')),)),
    )


# ---------------------------------------------------------------------------
# per-stage directives + graph-level directives
# ---------------------------------------------------------------------------


def test_stage_directives_set_level_and_combine(process):
    g = parse(
        process,
        '[cope ~ cond {{ level=subject; group_by=subject; combine=fixed }}] '
        '>> [. ~ 1 {{ level=dataset; combine=mixed; estimator=flame }}]',
    )
    s0, s1 = g.nodes
    assert s0.level is Level.SUBJECT
    assert s0.group_by == ('subject',)
    assert s0.combine is Combine.FIXED
    assert s1.level is Level.DATASET
    assert s1.combine is Combine.MIXED
    assert s1.spec.estimation.estimator == 'flame'


def test_graph_level_inference(process):
    g = parse(
        process,
        '[a ~ b] >> [. ~ 1] {{ inference=permutation(cluster_mass, n=5000) }}',
    )
    assert g.inference is not None
    assert g.inference.kind == 'permutation'
    assert g.inference.enhancement == 'cluster_mass'
    assert g.inference.n_perm == 5000


def test_edge_carry_binds_upstream_contrast(process):
    g = parse(
        process,
        '[a ~ b {{ contrasts: eff = b (t) }}] >> [. ~ 1]',
    )
    assert g.edges[0].carry.contrast == 'eff'


# ---------------------------------------------------------------------------
# `.` is position-sensitive: a cope only inside a pipeline downstream stage
# ---------------------------------------------------------------------------


def test_dot_outside_pipeline_is_complement(process):
    # On a normal RHS, `.` is the complement set (a reserved lookup), NOT a
    # cope referent.
    spec = parse(process, 'y ~ .').nodes[0].spec
    assert TermSpec((FactorSpec(Lookup('.')),)) in spec.fixed
    assert all(
        not isinstance(f.source, Referent)
        for t in spec.fixed
        for f in t.factors
    )


def test_first_stage_dot_is_not_a_cope(process):
    # The first stage has no upstream, so its `.` stays a complement lookup.
    g = parse(process, '[. ~ b] >> [c ~ d]')
    s0_resp = g.nodes[0].spec.response.terms
    assert s0_resp == (TermSpec((FactorSpec(Lookup('.')),)),)


# ---------------------------------------------------------------------------
# single-node formulae are unchanged
# ---------------------------------------------------------------------------


def test_single_node_unchanged(process):
    g = parse(process, 'y ~ x + z {{ family=binomial }}')
    assert [n.name for n in g.nodes] == ['root']
    assert g.edges == ()
    assert g.nodes[0].spec.fixed == (INTERCEPT, L('x'), L('z'))


# ---------------------------------------------------------------------------
# lexer
# ---------------------------------------------------------------------------


def test_stage_pipe_lexes():
    grammar = NwxGrammar()
    grammar.input('[a] >> [b]')
    types = []
    while True:
        tok = grammar._lexer.token()
        if tok is None:
            break
        types.append(tok.type)
    assert 'STAGE_PIPE' in types


def test_pipeline_ir_is_hashable_and_value_equal(process):
    g1 = parse(process, '[a ~ b] >> [. ~ 1]')
    g2 = parse(process, '[a ~ b] >> [. ~ 1]')
    assert g1 == g2
    assert hash(g1) == hash(g2)
