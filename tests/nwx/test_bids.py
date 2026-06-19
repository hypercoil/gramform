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
import warnings

import pytest

from gramform.grammars.nwx.bids import (
    BidsExportError,
    BidsImportError,
    export_bids_file,
    export_bids_model,
    import_bids_file,
    import_bids_model,
    validate_exportable,
)
from gramform.grammars.nwx.spec import (
    INTERCEPT,
    Combine,
    FactorSpec,
    Level,
    Lookup,
    Severity,
    TermSpec,
)
from gramform.grammars.nwx.spec import (
    Test as _Test,  # aliased so pytest does not collect the `Test*` enum
)
from gramform.grammars.nwx.transform import get_processor
from gramform.grammars.nwx.validate import validate

from .dryrun import dry_run


def L(name: str) -> TermSpec:
    return TermSpec((FactorSpec(Lookup(name)),))


@pytest.fixture(scope='module')
def process():
    return get_processor()


def parse(process, formula: str):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return process(formula)


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


# ===========================================================================
# export direction: ModelGraph -> BIDS Stats Models (strict)
# ===========================================================================


# ---------------------------------------------------------------------------
# round-trip + the document shape
# ---------------------------------------------------------------------------


def test_import_export_round_trips():
    # import -> export -> import is identity on the representable subset.
    g = import_bids_model(TWO_LEVEL)
    assert import_bids_model(export_bids_model(g, name='two-level')) == g


def test_export_is_json_idempotent():
    g = import_bids_model(TWO_LEVEL)
    doc1 = export_bids_model(g)
    doc2 = export_bids_model(import_bids_model(doc1))
    assert doc1 == doc2


def test_export_document_shape():
    doc = export_bids_model(import_bids_model(TWO_LEVEL), name='m')
    assert doc['Name'] == 'm'
    assert doc['BIDSModelVersion'] == '1.0.0'
    assert [n['Name'] for n in doc['Nodes']] == ['subject', 'dataset']


def test_export_node_level_type_and_design():
    doc = export_bids_model(import_bids_model(TWO_LEVEL))
    subject, dataset = doc['Nodes']
    assert subject['Level'] == 'Subject'  # capitalised back
    assert subject['Model'] == {'Type': 'glm', 'X': ['1', 'trial_type']}
    assert subject['GroupBy'] == ['subject']
    # Combine.MIXED -> Type 'meta' (the inverse of the importer)
    assert dataset['Model']['Type'] == 'meta'


def test_export_contrast_round_trips_fields():
    subject = export_bids_model(import_bids_model(TWO_LEVEL))['Nodes'][0]
    (contrast,) = subject['Contrasts']
    assert contrast == {
        'Name': 'effect',
        'ConditionList': ['trial_type'],
        'Weights': [1.0],
        'Test': 't',
    }


def test_export_edge_carry_becomes_filter_contrast():
    (edge,) = export_bids_model(import_bids_model(TWO_LEVEL))['Edges']
    assert edge['Source'] == 'subject' and edge['Destination'] == 'dataset'
    assert edge['Filter'] == {'contrast': ['effect']}


def test_export_multi_value_filter_round_trips():
    doc = {
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
    g = import_bids_model(doc)
    (edge,) = export_bids_model(g)['Edges']
    assert edge['Filter'] == {'session': ['pre', 'post'], 'contrast': ['c']}


def test_export_to_file(tmp_path):
    g = import_bids_model(TWO_LEVEL)
    path = tmp_path / 'out.json'
    export_bids_file(g, path, name='two-level')
    assert import_bids_file(path) == g


# ---------------------------------------------------------------------------
# the GLM-graph subset exports; multi-level pipelines export
# ---------------------------------------------------------------------------


def test_native_formula_exports(process):
    doc = export_bids_model(
        parse(process, 'y ~ x + z {{ contrasts: b = x (t) }}')
    )
    (node,) = doc['Nodes']
    assert node['Model']['X'] == ['1', 'x', 'z']
    assert node['Contrasts'][0]['Name'] == 'b'


def test_interaction_renders_as_colon(process):
    (node,) = export_bids_model(parse(process, 'y ~ a*b'))['Nodes']
    assert node['Model']['X'] == ['1', 'a', 'b', 'a:b']


def test_pipeline_exports_nodes_and_edge(process):
    doc = export_bids_model(
        parse(process, '[a ~ b] >> [. ~ 1 {{ combine=mixed }}]')
    )
    assert [n['Model']['Type'] for n in doc['Nodes']] == ['glm', 'meta']
    assert len(doc['Edges']) == 1


def test_dropped_inference_does_not_block_export(process):
    # node-level inference is out of BIDS-SM scope -> dropped, not refused.
    doc = export_bids_model(
        parse(
            process,
            'y ~ x {{ contrasts: c = x (t); inference=permutation(tfce) }}',
        )
    )
    (node,) = doc['Nodes']
    assert node['Contrasts'][0]['Name'] == 'c'
    assert 'Inference' not in node and 'Inference' not in doc


# ---------------------------------------------------------------------------
# strict refusal of unrepresentable constructs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'formula,needle',
    [
        ('y ~ x + (1|g)', 'random effects'),
        ('y ~ s(age)', 'smooth'),
        ('bold ~| n', 'residualisation'),
        ('y ~ x + noise(z)', 'partial'),
        ('y ~ x {{ family=binomial }}', 'Gaussian/identity'),
        ('y ~ x {{ correlation=ar1(t|g) }}', 'error structure'),
    ],
)
def test_strict_refuses_unrepresentable(process, formula, needle):
    with pytest.raises(BidsExportError, match=needle):
        export_bids_model(parse(process, formula))


def test_refusal_message_names_the_node(process):
    g = parse(process, 'y ~ x + [bold ~| n]')
    with pytest.raises(BidsExportError, match=r'\[frame0\]'):
        export_bids_model(g)


def test_frame_referent_design_is_refused(process):
    # `y ~ x + [z ~ w]` lifts a frame fit (`_hat`) into the design -> no plain
    # column form -> refused.
    g = parse(process, 'y ~ x + [z ~ w]')
    with pytest.raises(BidsExportError, match='plain column'):
        export_bids_model(g)


# ---------------------------------------------------------------------------
# validate_exportable: the static check behind the raise
# ---------------------------------------------------------------------------


def test_validate_exportable_clean_is_empty(process):
    assert validate_exportable(parse(process, 'y ~ x + z')) == ()


def test_validate_exportable_flags_each_node(process):
    diags = validate_exportable(parse(process, 'y ~ s(age) + (1|g)'))
    assert diags  # at least the smooth + random
    assert all(d.severity is Severity.ERROR for d in diags)
    assert all(d.code == 'bids-export-unrepresentable' for d in diags)


def test_imported_then_validated_then_exported(process):
    # the full bridge: model.json -> IR -> (clean) -> model.json.
    g = import_bids_model(TWO_LEVEL)
    assert validate_exportable(g) == ()
    assert validate(g) == ()
    assert export_bids_model(g)['Nodes']
