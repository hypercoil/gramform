# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Phase-7 static validation (spec §8).

``validate(graph)`` returns Diagnostics for a parsed ``ModelGraph``: a smooth
``by=`` factor missing its main effect, aggressive residualisation given a
``signal()`` set, the backend-awareness roll-up (v3-gated features), and
multi-level graph integrity (edge endpoints, carry resolution, acyclicity).
"""

import warnings

import pytest

from gramform.grammars.nwx.spec import (
    Carry,
    Combine,
    ContrastSpec,
    Diagnostic,
    Edge,
    Level,
    ModelGraph,
    ModelNode,
    ModelSpec,
    ResponseSpec,
    Severity,
)
from gramform.grammars.nwx.spec import (
    Test as _Test,  # aliased so pytest does not collect the `Test*` enum
)
from gramform.grammars.nwx.transform import get_processor
from gramform.grammars.nwx.validate import validate


@pytest.fixture(scope='module')
def process():
    return get_processor()


def parse(process, formula: str):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return process(formula)


def codes(diags) -> list[str]:
    return [d.code for d in diags]


# ---------------------------------------------------------------------------
# node-level: smooth by=, aggressive+signal
# ---------------------------------------------------------------------------


def test_clean_model_has_no_diagnostics(process):
    assert validate(parse(process, 'y ~ x + z')) == ()


def test_smooth_by_without_main_effect_warns(process):
    diags = validate(parse(process, 'y ~ s(age, by=dx)'))
    hit = [d for d in diags if d.code == 'smooth-by-main-effect']
    assert hit and hit[0].severity is Severity.WARNING
    assert 'dx' in hit[0].message


def test_smooth_by_with_main_effect_is_clean(process):
    diags = validate(parse(process, 'y ~ s(age, by=dx) + dx'))
    assert 'smooth-by-main-effect' not in codes(diags)


def test_re_basis_by_is_exempt(process):
    # `s(g, x, bs="re")` uses `by` as a slope variable, not a factor.
    diags = validate(parse(process, 'y ~ s(g, x, bs="re")'))
    assert 'smooth-by-main-effect' not in codes(diags)


def test_aggressive_residualise_with_signal_warns(process):
    diags = validate(parse(process, 'bold ~| signal(s) + n'))
    assert 'residualise-aggressive-signal' in codes(diags)


# ---------------------------------------------------------------------------
# backend-awareness roll-up
# ---------------------------------------------------------------------------


# nitrix v3 ships nwx's whole v1 scope, so the roll-up is now a short residual:
# non-canonical links, the `soft` residualise mode, RFT inference, and a
# non-Gaussian random slope (glmm_fit is scalar-RE only).
@pytest.mark.parametrize(
    'formula,detail',
    [
        ('y ~ x {{ link=probit }}', "link 'probit'"),
        ('bold ~| n {{ residualise=soft }}', "residualise='soft'"),
        ('y ~ x {{ inference=parametric(rft) }}', "correction 'rft'"),
        (
            'y ~ x + (1+x|g) {{ family=binomial }}',
            'non-Gaussian random slope',
        ),
    ],
)
def test_backend_awareness(process, formula, detail):
    diags = validate(parse(process, formula))
    hit = [d for d in diags if d.code == 'backend-unshipped']
    assert any(detail in d.message for d in hit), [d.message for d in hit]


def test_core_model_has_no_backend_warning(process):
    diags = validate(parse(process, 'y ~ x {{ family=binomial; link=logit }}'))
    assert 'backend-unshipped' not in codes(diags)


@pytest.mark.parametrize(
    'formula',
    [
        'y ~ x {{ family=gamma }}',  # v3 §4
        'y ~ x {{ family=tweedie }}',
        'y ~ x {{ se=robust(hc3) }}',  # v3 §6.2
        'y ~ x {{ dof=kr }}',  # v3 §1.3
        'y ~ x + (1+x|g)',  # unstructured, lme_fit R2
        'y ~ x + (1|g1/g2)',  # nested, lme_fit R3
        'y ~ s(age, bs="cr")',  # v3 §3.2
        'y ~ s(age, bs="gp")',
        'y ~ x {{ correlation=ar1(t|g) }}',  # v3 §1.4
        'y ~ x {{ weights=varPower(z) }}',
        'bold ~| noise(n) + signal(s) {{ residualise=nonaggressive }}',  # §5.1
        'y ~ x + (1+x|g) {{ family=gaussian }}',  # Gaussian random slope: R2
    ],
)
def test_shipped_features_have_no_backend_warning(process, formula):
    # everything nitrix v3 now ships must NOT roll up as backend-unshipped.
    diags = validate(parse(process, formula))
    assert 'backend-unshipped' not in codes(diags), [
        d.message for d in diags if d.code == 'backend-unshipped'
    ]


# ---------------------------------------------------------------------------
# graph-level integrity (hand-built graphs the parser would never emit)
# ---------------------------------------------------------------------------


def _node(name, estimands=()):
    return ModelNode(
        name=name,
        level=Level.DATASET,
        group_by=(),
        combine=Combine.FIXED,
        spec=ModelSpec(
            response=ResponseSpec(terms=()),
            fixed=(),
            estimands=estimands,
        ),
    )


def _carry(contrast=''):
    return Carry(contrast=contrast, quantities='cope_varcope')


def test_edge_to_unknown_node_errors():
    g = ModelGraph(
        nodes=(_node('a'),),
        edges=(Edge('a', 'ghost', (), _carry()),),
    )
    hit = [d for d in validate(g) if d.code == 'edge-unknown-node']
    assert hit and hit[0].severity is Severity.ERROR


def test_edge_carry_unresolved_errors():
    g = ModelGraph(
        nodes=(_node('a'), _node('b')),
        edges=(Edge('a', 'b', (), _carry('missing')),),
    )
    assert 'edge-carry-unresolved' in codes(validate(g))


def test_edge_carry_resolves_is_clean():
    src = _node('a', estimands=(ContrastSpec('eff', (('x', 1.0),), _Test.T),))
    g = ModelGraph(
        nodes=(src, _node('b')),
        edges=(Edge('a', 'b', (), _carry('eff')),),
    )
    assert 'edge-carry-unresolved' not in codes(validate(g))


def test_graph_cycle_errors():
    g = ModelGraph(
        nodes=(_node('a'), _node('b')),
        edges=(
            Edge('a', 'b', (), _carry()),
            Edge('b', 'a', (), _carry()),
        ),
    )
    assert 'graph-cycle' in codes(validate(g))


def test_pipeline_carry_resolves(process):
    g = parse(process, '[a ~ b {{ contrasts: eff = b (t) }}] >> [. ~ 1]')
    assert 'edge-carry-unresolved' not in codes(validate(g))
    assert 'graph-cycle' not in codes(validate(g))


def test_diagnostics_are_frozen_dataclasses(process):
    diags = validate(parse(process, 'y ~ s(age, by=dx)'))
    assert all(isinstance(d, Diagnostic) for d in diags)
    with pytest.raises(Exception):
        diags[0].code = 'mutated'  # frozen
