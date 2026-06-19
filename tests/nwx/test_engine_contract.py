# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Phase-7 engine contract (spec §9), round-tripped by the dry-run dispatcher.

``dry_run`` turns a parsed ``ModelGraph`` into the nitrix call sequence the
engine would make, WITHOUT importing nitrix. These tests pin the §9 dispatch
table and the §11 examples, and the shipped / v3-gated reachability annotation.
"""

import warnings

import pytest

from gramform.grammars.nwx.transform import get_processor

from .dryrun import Call, dry_run, render


@pytest.fixture(scope='module')
def process():
    return get_processor()


def plan(process, formula: str):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return dry_run(process(formula))


def node(run, name: str):
    return next(n for n in run.nodes if n.node == name)


def routines(node_plan) -> list[str]:
    return [c.routine for c in node_plan.calls]


# ---------------------------------------------------------------------------
# the §9 dispatch table
# ---------------------------------------------------------------------------


def test_only_fixed_dispatches_to_glm(process):
    run = plan(process, 'y ~ x + z {{ contrasts: b = x (t) }}')
    root = node(run, 'root')
    assert root.route == 'glm_fit'
    assert 't_contrast' in routines(root)


def test_partial_is_glm_fwl(process):
    root = node(plan(process, 'y ~ x + noise(z)'), 'root')
    assert root.route == 'glm_fit'
    assert 'FWL' in root.calls[0].detail


def test_scalar_random_is_reml(process):
    root = node(plan(process, 'y ~ x + (1|g)'), 'root')
    assert root.route == 'reml_fit'
    assert root.calls[0].shipped  # R1 ships today


def test_nonscalar_random_is_lme(process):
    root = node(plan(process, 'y ~ x + (1+x|g)'), 'root')
    assert root.route == 'lme_fit'
    assert root.calls[0].shipped  # R2 (block-Woodbury) ships in nitrix v3


def test_nongaussian_random_slope_is_unshipped_glmm(process):
    # glmm_fit ships a scalar RE only -> a non-Gaussian random slope is the
    # one random-effect case still gated.
    root = node(plan(process, 'y ~ x + (1+x|g) {{ family=binomial }}'), 'root')
    assert root.route == 'glmm_fit'
    assert not root.calls[0].shipped


def test_smooth_is_gam(process):
    root = node(plan(process, 'y ~ s(age, k=6)'), 'root')
    assert root.route == 'gam_fit'


def test_residualise_frame_is_linalg_residualise(process):
    run = plan(process, 'y ~ x + [bold ~| n1 + n2]')
    assert node(run, 'frame0').route == 'linalg.residualise'


def test_dataset_node_fed_by_copes_is_flame(process):
    run = plan(process, '[cope ~ cond] >> [. ~ 1 {{ combine=mixed }}]')
    assert node(run, 'stage0').route == 'glm_fit'
    assert node(run, 'stage1').route == 'flame_two_level'
    assert 'mixed-effects' in node(run, 'stage1').calls[0].detail
    assert run.edges  # the cope/varcope edge is planned


# ---------------------------------------------------------------------------
# family / contrast / inference routing
# ---------------------------------------------------------------------------


def test_f_contrast_for_f_test(process):
    root = node(plan(process, 'y ~ a + b {{ contrasts: ab = a (F) }}'), 'root')
    assert 'f_contrast' in routines(root)


def test_glm_families_ship_but_noncanonical_link_does_not(process):
    # all nwx families ship in v3; the residual is a non-canonical link.
    for family in ('binomial', 'gamma', 'tweedie'):
        glm = node(plan(process, f'y ~ x {{{{ family={family} }}}}'), 'root')
        assert glm.calls[0].routine == 'glm_fit' and glm.calls[0].shipped
    probit = node(plan(process, 'y ~ x {{ link=probit }}'), 'root')
    assert not probit.calls[0].shipped


@pytest.mark.parametrize(
    'directive,routine',
    [
        ('inference=permutation(tfce, n=200)', 'permutation_test'),
        ('inference=parametric(fdr)', 'fdr_bh'),
        ('inference=parametric(bonferroni)', 'bonferroni'),
    ],
)
def test_inference_routing(process, directive, routine):
    root = node(plan(process, f'y ~ x {{{{ {directive} }}}}'), 'root')
    assert routine in routines(root)


# ---------------------------------------------------------------------------
# §11 worked examples round-trip
# ---------------------------------------------------------------------------


def test_m1_example_is_fully_shipped(process):
    run = plan(
        process,
        'thk ~ dx + sex + noise(meanFD) '
        '{{ contrasts: dx = dx (t); inference=permutation(tfce, n=5000) }}',
    )
    root = node(run, 'root')
    assert routines(root) == ['glm_fit', 't_contrast', 'permutation_test']
    # the single most-used neuroimaging model runs on shipped kernels today
    assert all(c.shipped for c in root.calls)


def test_multilevel_flame_example(process):
    run = plan(
        process,
        '[cope ~ cond {{ level=subject; group_by=subject; combine=fixed }}] '
        '>> [. ~ 1 {{ level=dataset; combine=mixed; estimator=flame; '
        'contrasts: grp = 1 (t) }}] '
        '{{ inference=permutation(cluster_mass, n=5000) }}',
    )
    assert [n.route for n in run.nodes] == ['glm_fit', 'flame_two_level']
    assert len(run.edges) == 1
    assert run.inference and run.inference[0].routine == 'permutation_test'


def test_gamm_example_now_ships(process):
    # the GAMM (s(by=) + a random effect) that v1 could not run is shipped in
    # v3 (by_factor_smooth §3.1 + the re_smooth GAMM bridge §2).
    run = plan(
        process,
        'thk ~ s(age, k=6, by=dx) + dx + noise(meanFD) + (1|site) '
        '{{ inference=permutation(tfce) }}',
    )
    root = node(run, 'root')
    assert root.route == 'gam_fit'
    assert root.calls[0].shipped


# ---------------------------------------------------------------------------
# render + value semantics
# ---------------------------------------------------------------------------


def test_render_is_a_transcript(process):
    text = render(plan(process, 'thk ~ dx + noise(meanFD)'))
    assert 'node root: [glm_fit]' in text
    assert 'glm_fit' in text


def test_calls_are_frozen_hashable(process):
    run = plan(process, 'y ~ x + (1|g)')
    assert isinstance(run.nodes[0].calls[0], Call)
    assert hash(run.nodes[0].calls[0]) == hash(run.nodes[0].calls[0])
