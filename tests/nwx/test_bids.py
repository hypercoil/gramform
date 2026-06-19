# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Phase-7 read-direction BIDS Stats Models importer (spec §10).

``import_bids_model(model.json dict) -> ModelGraph``: Nodes -> ModelNodes,
Edges -> Edges, Model.X -> fixed design, Contrasts / DummyContrasts ->
ContrastSpec, Edge.Filter['contrast'] -> Edge.carry. The imported graph feeds
straight into ``validate`` and the dry-run dispatcher -- a full
model.json -> IR -> static report + planned nitrix calls pipeline.
"""

import json

import pytest

from gramform.grammars.nwx.bids import (
    BidsImportError,
    import_bids_file,
    import_bids_model,
)
from gramform.grammars.nwx.spec import (
    INTERCEPT,
    Combine,
    FactorSpec,
    Level,
    Lookup,
    TermSpec,
)
from gramform.grammars.nwx.spec import (
    Test as _Test,  # aliased so pytest does not collect the `Test*` enum
)
from gramform.grammars.nwx.validate import validate

from .dryrun import dry_run


def L(name: str) -> TermSpec:
    return TermSpec((FactorSpec(Lookup(name)),))


TWO_LEVEL = {
    'Name': 'two-level',
    'BIDSModelVersion': '1.0.0',
    'Nodes': [
        {
            'Level': 'Subject',
            'Name': 'subject',
            'GroupBy': ['subject'],
            'Model': {'Type': 'glm', 'X': ['1', 'trial_type']},
            'Contrasts': [
                {
                    'Name': 'effect',
                    'ConditionList': ['trial_type'],
                    'Weights': [1],
                    'Test': 't',
                }
            ],
        },
        {
            'Level': 'Dataset',
            'Name': 'dataset',
            'GroupBy': ['contrast'],
            'Model': {'Type': 'meta', 'X': ['1']},
            'DummyContrasts': {'Test': 't'},
        },
    ],
    'Edges': [
        {
            'Source': 'subject',
            'Destination': 'dataset',
            'Filter': {'contrast': ['effect']},
        }
    ],
}


# ---------------------------------------------------------------------------
# node / edge mapping
# ---------------------------------------------------------------------------


def test_nodes_map_level_combine_and_design():
    g = import_bids_model(TWO_LEVEL)
    subject, dataset = g.nodes
    assert subject.name == 'subject'
    assert subject.level is Level.SUBJECT
    assert subject.group_by == ('subject',)
    assert subject.combine is Combine.FIXED  # Model.Type == glm
    assert subject.spec.fixed == (INTERCEPT, L('trial_type'))
    # Model.Type == meta -> mixed-effects group node
    assert dataset.level is Level.DATASET
    assert dataset.combine is Combine.MIXED


def test_contrast_maps_conditionlist_and_weights():
    subject = import_bids_model(TWO_LEVEL).nodes[0]
    (effect,) = subject.spec.estimands
    assert effect.name == 'effect'
    assert effect.weights == (('trial_type', 1.0),)
    assert effect.test is _Test.T


def test_edge_filter_contrast_becomes_carry():
    g = import_bids_model(TWO_LEVEL)
    (edge,) = g.edges
    assert (edge.source, edge.dest) == ('subject', 'dataset')
    assert edge.carry.contrast == 'effect'
    assert edge.carry.quantities == 'cope_varcope'
    assert edge.filter == ()


def test_non_contrast_filter_becomes_filter_pairs():
    g = import_bids_model(
        {
            'Nodes': [
                {'Level': 'Run', 'Name': 'a', 'GroupBy': [], 'Model': {}},
                {'Level': 'Subject', 'Name': 'b', 'GroupBy': [], 'Model': {}},
            ],
            'Edges': [
                {
                    'Source': 'a',
                    'Destination': 'b',
                    'Filter': {'session': ['pre', 'post'], 'contrast': ['c']},
                }
            ],
        }
    )
    (edge,) = g.edges
    assert edge.carry.contrast == 'c'
    assert edge.filter == (('session', 'pre'), ('session', 'post'))


# ---------------------------------------------------------------------------
# contrasts: dummy, F-test; design: formula fallback; levels
# ---------------------------------------------------------------------------


def test_dummy_contrasts_make_unit_contrasts():
    g = import_bids_model(
        {
            'Nodes': [
                {
                    'Level': 'Run',
                    'Name': 'r',
                    'GroupBy': [],
                    'Model': {'X': ['1', 'a', 'b']},
                    'DummyContrasts': {'Test': 'F'},
                }
            ]
        }
    )
    estimands = g.nodes[0].spec.estimands
    assert {c.name for c in estimands} == {'a', 'b'}  # '1' excluded
    assert all(c.test is _Test.F for c in estimands)
    assert all(c.weights == ((c.name, 1.0),) for c in estimands)


def test_formula_fallback_when_no_X():
    g = import_bids_model(
        {
            'Nodes': [
                {
                    'Level': 'Run',
                    'Name': 'r',
                    'GroupBy': [],
                    'Model': {'Formula': '0 + a + b'},
                }
            ]
        }
    )
    # `0 +` suppresses the intercept; the nwx processor parses the RHS
    assert g.nodes[0].spec.fixed == (L('a'), L('b'))


@pytest.mark.parametrize(
    'bids_level,level',
    [
        ('Run', Level.RUN),
        ('Session', Level.SESSION),
        ('Subject', Level.SUBJECT),
        ('Dataset', Level.DATASET),
    ],
)
def test_all_levels(bids_level, level):
    g = import_bids_model(
        {'Nodes': [{'Level': bids_level, 'Name': 'n', 'Model': {}}]}
    )
    assert g.nodes[0].level is level


# ---------------------------------------------------------------------------
# the imported graph feeds validate + the dry-run dispatcher
# ---------------------------------------------------------------------------


def test_imported_graph_validates_clean():
    assert validate(import_bids_model(TWO_LEVEL)) == ()


def test_imported_graph_dispatches_to_flame():
    run = dry_run(import_bids_model(TWO_LEVEL))
    assert [n.route for n in run.nodes] == ['glm_fit', 'flame_two_level']
    assert run.edges and 'effect' in run.edges[0]


def test_import_from_file(tmp_path):
    path = tmp_path / 'model.json'
    path.write_text(json.dumps(TWO_LEVEL), encoding='utf-8')
    g = import_bids_file(path)
    assert [n.name for n in g.nodes] == ['subject', 'dataset']


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


def test_unknown_level_errors():
    with pytest.raises(BidsImportError, match='Level'):
        import_bids_model({'Nodes': [{'Level': 'Galaxy', 'Name': 'n'}]})


def test_node_without_name_errors():
    with pytest.raises(BidsImportError, match='Name'):
        import_bids_model({'Nodes': [{'Level': 'Run', 'Model': {}}]})


def test_edge_without_endpoints_errors():
    with pytest.raises(BidsImportError, match='Source'):
        import_bids_model(
            {
                'Nodes': [{'Level': 'Run', 'Name': 'a', 'Model': {}}],
                'Edges': [{'Source': 'a'}],
            }
        )


def test_non_object_document_errors():
    with pytest.raises(BidsImportError):
        import_bids_model([1, 2, 3])
