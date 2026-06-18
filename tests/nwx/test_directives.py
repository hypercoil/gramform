# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for the minimal Phase-1 directive mini-parser.
"""

from gramform.grammars.nwx.directives import parse_directives
from gramform.grammars.nwx.spec import Family, Link, Severity


def test_family_and_link():
    d = parse_directives('family=binomial; link=logit')
    assert d.family is not None
    assert d.family.family is Family.BINOMIAL
    assert d.family.link is Link.LOGIT


def test_estimation_keys_merge_into_one_spec():
    d = parse_directives('estimator=reml; se=robust; dof=satterthwaite')
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
