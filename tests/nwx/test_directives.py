# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for the directive mini-parser (full Phase-4 key set).
"""

import warnings

import pytest

from gramform.grammars.nwx.directives import parse_directives
from gramform.grammars.nwx.spec import (
    BackendWarning,
    Combine,
    CorrelationSpec,
    FactorSpec,
    Family,
    Level,
    Link,
    Lookup,
    Severity,
    WeightSpec,
)
from gramform.grammars.nwx.transform import get_processor


def _quiet(text: str):
    """Parse, silencing backend-awareness warnings (asserted separately)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', BackendWarning)
        return parse_directives(text)


def test_family_and_link():
    d = parse_directives('family=binomial; link=logit')
    assert d.family is not None
    assert d.family.family is Family.BINOMIAL
    assert d.family.link is Link.LOGIT


def test_estimation_keys_merge_into_one_spec():
    d = _quiet('estimator=reml; se=robust; dof=satterthwaite')
    assert d.estimation is not None
    assert d.estimation.estimator == 'reml'
    assert d.estimation.se == 'robust'
    assert d.estimation.dof == 'satterthwaite'


def test_contrasts_linear_combinations():
    d = parse_directives(
        'contrasts: a = dx (t), b = g1 - g2 (F), c = 2*x + y'
    )
    names = {c.name: c for c in d.estimands}
    assert names['a'].weights == (('dx', 1.0),)
    assert names['a'].test.value == 't'
    assert names['b'].weights == (('g1', 1.0), ('g2', -1.0))
    assert names['b'].test.value == 'F'
    assert names['c'].weights == (('x', 2.0), ('y', 1.0))
    # No explicit test defaults to t.
    assert names['c'].test.value == 't'


def test_inference_permutation():
    d = parse_directives('inference=permutation(tfce, n=5000)')
    assert d.inference is not None
    assert d.inference.kind == 'permutation'
    assert d.inference.enhancement == 'tfce'
    assert d.inference.n_perm == 5000


def test_inference_cluster_mass_alias_and_parametric():
    d = parse_directives('inference=permutation(cluster_mass, n=100)')
    assert d.inference.enhancement == 'cluster_mass'
    with warnings.catch_warnings():  # rft warns (intentional); not under test
        warnings.simplefilter('ignore', BackendWarning)
        p = parse_directives('inference=parametric(rft)')
    assert p.inference.kind == 'parametric'
    assert p.inference.correction == 'rft'


def test_unknown_key_warns_not_errors():
    d = parse_directives('family=gaussian; wibble=3')
    assert d.family is not None
    assert any(
        x.severity is Severity.WARNING and 'wibble' in x.message
        for x in d.diagnostics
    )


def test_unknown_enum_value_warns():
    d = parse_directives('family=martian')
    assert d.family is None
    assert any(x.severity is Severity.WARNING for x in d.diagnostics)


def test_empty_block_is_inert():
    d = parse_directives('   ')
    assert d.family is None
    assert d.estimation is None
    assert d.estimands == ()
    assert d.inference is None
    assert d.diagnostics == ()


# ---------------------------------------------------------------------------
# full Phase-4 key set: robust/cluster se, error structures, node-level keys
# ---------------------------------------------------------------------------


def test_robust_se_variant():
    d = _quiet('se=robust(hc3)')
    assert d.estimation.se == 'robust'
    assert d.estimation.robust_variant == 'hc3'


def test_cluster_se():
    d = _quiet('se=cluster(subject)')
    assert d.estimation.se == 'cluster'
    assert d.estimation.cluster_by == 'subject'


def test_correlation_structure():
    d = _quiet('correlation=ar1(session | subject)')
    assert d.errors is not None
    assert d.errors.correlation == CorrelationSpec(
        kind='ar1',
        index=FactorSpec(Lookup('session')),
        group=FactorSpec(Lookup('subject')),
    )


def test_weights_structure():
    d = _quiet('weights=varPower(meanFD)')
    assert d.errors is not None
    assert d.errors.heteroscedasticity == WeightSpec(
        kind='varPower', arg=FactorSpec(Lookup('meanFD'))
    )


def test_node_level_keys():
    d = _quiet('level=subject; group_by=subject; combine=fixed')
    assert d.level is Level.SUBJECT
    assert d.group_by == ('subject',)
    assert d.combine is Combine.FIXED


def test_group_by_multiple():
    d = _quiet('group_by=site, subject')
    assert d.group_by == ('site', 'subject')


def test_malformed_correlation_warns():
    d = _quiet('correlation=ar1(time)')  # missing the `| group`
    assert d.errors is None
    assert any(x.where == 'correlation' for x in d.diagnostics)


# ---------------------------------------------------------------------------
# backend-awareness warnings (spec §7)
# ---------------------------------------------------------------------------


# The nitrix GP branch ships every directive-reachable axis -- families,
# non-canonical links (Family.with_link), robust/cluster SEs, Satterthwaite/KR
# dof, error correlation + heteroscedasticity, and all residualise modes. The
# *sole* residual reachable from a directive is RFT inference, intentionally
# omitted for its known failure modes.
@pytest.mark.parametrize(
    'text',
    [
        'inference=parametric(rft)',
    ],
)
def test_backend_awareness_warns(text):
    with pytest.warns(BackendWarning, match='intentionally not shipped'):
        parse_directives(text)


@pytest.mark.parametrize(
    'text',
    [
        'family=binomial; link=logit; estimator=reml',
        'family=gamma',  # §4
        'link=probit',  # non-canonical link via Family.with_link
        'link=inverse',
        'link=sqrt',
        'se=robust(hc3)',  # §6.2
        'se=cluster(subject)',
        'dof=satterthwaite',  # §1.3
        'dof=kr',
        'correlation=ar1(time | g)',  # §1.4
        'weights=varPower(x)',
        'residualise=nonaggressive',  # §5.1
        'residualise=soft',  # FR §5.2 (ridge / James-Stein shrunk)
        'inference=parametric(fdr)',  # FDR-BH ships
    ],
)
def test_shipped_directives_do_not_warn(text):
    with warnings.catch_warnings():
        warnings.simplefilter('error', BackendWarning)
        parse_directives(text)


# ---------------------------------------------------------------------------
# end-to-end: directives land on the ModelSpec / ModelNode via the processor
# ---------------------------------------------------------------------------


def test_error_structure_lands_on_modelspec():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', BackendWarning)
        g = get_processor()(
            'y ~ x {{ correlation=ar1(session | subject) }}'
        )
    errors = g.nodes[0].spec.errors
    assert errors is not None
    assert errors.correlation is not None
    assert errors.correlation.kind == 'ar1'


def test_node_level_directives_land_on_node():
    g = get_processor()(
        'cope ~ cond {{ level=subject; group_by=subject; combine=mixed }}'
    )
    node = g.nodes[0]
    assert node.level is Level.SUBJECT
    assert node.group_by == ('subject',)
    assert node.combine is Combine.MIXED
