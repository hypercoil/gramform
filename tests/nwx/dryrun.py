# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Dry-run engine dispatcher (the executable form of the spec §9 contract).

``dry_run(graph)`` turns a validated :class:`ModelGraph` into the *sequence of
nitrix calls the engine would make* -- structurally, by which IR fields are
populated -- WITHOUT importing ``nitrix`` (or any array library). It proves the
IR -> engine dispatch and round-trips the §11 examples.

It lives **test-side, not in the nwx package**: the populated-fields -> routine
dispatch is the engine's concern, not nwx's (spec §1). Each call is tagged with
whether its nitrix kernel ships today (`shipped`) or is gated on the nitrix v3
feature request (per spec §7), so the contract doubly documents reachability.
See ``docs/nwx/engine-contract.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

from gramform.grammars.nwx import backend
from gramform.grammars.nwx.spec import (
    Combine,
    Family,
    Mode,
    ModelGraph,
    ModelNode,
    ModelSpec,
    Structure,
    Test,
)


@dataclass(frozen=True)
class Call:
    """One intended nitrix routine call."""

    routine: str
    detail: str
    shipped: bool = True


@dataclass(frozen=True)
class NodePlan:
    """The ordered nitrix calls for one model node."""

    node: str
    route: str  # the primary fit routine
    calls: tuple[Call, ...]


@dataclass(frozen=True)
class DryRun:
    nodes: tuple[NodePlan, ...]
    edges: tuple[str, ...]  # cope/varcope propagation descriptions
    inference: tuple[Call, ...]  # graph-level inference


def dry_run(graph: ModelGraph) -> DryRun:
    """Plan the engine's nitrix calls for ``graph`` (spec §9)."""
    fed_by_copes = {edge.dest for edge in graph.edges}
    nodes = tuple(
        _node_plan(node, node.name in fed_by_copes) for node in graph.nodes
    )
    edges = tuple(
        f'propagate {{cope, varcope}} {edge.source} -> {edge.dest} '
        f'(carry {edge.carry.contrast or "<default cope>"!s}, '
        f'{edge.carry.quantities})'
        for edge in graph.edges
    )
    inference = (_inference_call(graph.inference),) if graph.inference else ()
    return DryRun(nodes=nodes, edges=edges, inference=inference)


# ---------------------------------------------------------------------------
# per-node dispatch (the §9 "cheapest exact route" table)
# ---------------------------------------------------------------------------


def _node_plan(node: ModelNode, fed_by_copes: bool) -> NodePlan:
    spec = node.spec
    fit = _fit_call(spec, fed_by_copes, node.combine)
    calls = [fit]
    calls.extend(_contrast_calls(spec))
    if spec.inference is not None:
        calls.append(_inference_call(spec.inference))
    return NodePlan(node=node.name, route=fit.routine, calls=tuple(calls))


def _fit_call(
    spec: ModelSpec,
    fed_by_copes: bool,
    combine: Combine,
) -> Call:
    # A higher-level node fed by lower-level copes -> FLAME.
    if fed_by_copes:
        mode = 'mixed-effects' if combine is Combine.MIXED else 'fixed-effects'
        return Call(
            'flame_two_level',
            f'{mode} group model over inbound copes/varcopes',
            shipped=True,
        )
    # A residualise frame -> project onto the noise complement.
    if spec.residualise:
        res = spec.residualise[0]
        if res.mode is Mode.AGGRESSIVE:
            return Call(
                'linalg.residualise',
                'aggressive: project onto the noise complement (_tilde)',
                shipped=True,
            )
        return Call(
            'partial_residualise',
            'non-aggressive (ICA-AROMA): remove only the noise-unique fit',
            shipped=backend.residualise_mode_shipped(res.mode),
        )
    # Within-node fit: smooth > random > partial > plain (the cheapest exact).
    if spec.smooth:
        bridge = ' + re/fs GAMM blocks' if spec.random else ''
        shipped = all(backend.basis_shipped(s.basis) for s in spec.smooth)
        return Call(
            'gam_fit',
            f'penalised smooth bases{bridge}',
            shipped=shipped,
        )
    if spec.random:
        family = spec.family.family
        # A non-Gaussian family + random effect -> GLMM (glmm_fit, scalar RE
        # only); a Gaussian random effect -> reml_fit (R1) / lme_fit (R2-R4).
        if family is not Family.GAUSSIAN:
            structures = {re.structure for re in spec.random}
            slope = any(
                backend.glmm_random_slope_unshipped(s, family)
                for s in structures
            )
            return Call(
                'glmm_fit',
                f'{family.value} GLMM, scalar RE (PQL / Laplace)',
                shipped=not slope,
            )
        scalar = (
            len(spec.random) == 1
            and spec.random[0].structure is Structure.SCALAR
        )
        if scalar:
            return Call('reml_fit', 'single scalar variance component (R1)')
        return Call(
            'lme_fit',
            'structure-dispatch (R2-R4): non-scalar / nested / crossed',
            shipped=True,
        )
    family_shipped = backend.family_shipped(
        spec.family.family
    ) and backend.link_shipped(spec.family.link)
    if spec.partial:
        return Call(
            'glm_fit',
            'FWL: design includes the partial nuisance block; contrast loads '
            'only on the signal columns',
            shipped=family_shipped,
        )
    return Call(
        'glm_fit',
        f'{spec.family.family.value}/{spec.family.link.value} GLM',
        shipped=family_shipped,
    )


def _contrast_calls(spec: ModelSpec) -> list[Call]:
    # GLM (t/f_contrast) and LME (lme_t/f_contrast, Satterthwaite + KR dof)
    # contrasts both ship in nitrix v3.
    lme = bool(spec.random)
    calls = []
    for estimand in spec.estimands:
        base = 'f_contrast' if estimand.test is Test.F else 't_contrast'
        routine = f'lme_{base}' if lme else base
        calls.append(
            Call(routine, f'estimand {estimand.name!r}', shipped=True)
        )
    return calls


def _inference_call(inf) -> Call:
    if inf.kind == 'permutation':
        enh = inf.enhancement or 'voxel'
        n = inf.n_perm or 1000
        return Call(
            'permutation_test',
            f'Freedman-Lane, {enh} max-stat FWE (n={n})',
            shipped=True,
        )
    correction = inf.correction
    if correction == 'fdr':
        return Call('fdr_bh', 'Benjamini-Hochberg FDR')
    if correction == 'bonferroni':
        return Call('bonferroni', 'Bonferroni FWE')
    if correction == 'rft':
        return Call(
            'rft',
            'random-field-theory FWE',
            shipped=backend.inference_correction_shipped(correction),
        )
    return Call('parametric', f'parametric correction={correction}')


# ---------------------------------------------------------------------------
# rendering (for the engine-contract doc / human output)
# ---------------------------------------------------------------------------


def render(plan: DryRun) -> str:
    """A human-readable transcript of the planned nitrix call sequence."""
    lines: list[str] = []
    for node in plan.nodes:
        lines.append(f'node {node.node}: [{node.route}]')
        for call in node.calls:
            flag = '' if call.shipped else '  (nitrix v3)'
            lines.append(f'    {call.routine}: {call.detail}{flag}')
    for edge in plan.edges:
        lines.append(f'edge: {edge}')
    for call in plan.inference:
        flag = '' if call.shipped else '  (nitrix v3)'
        lines.append(f'graph inference: {call.routine}: {call.detail}{flag}')
    return '\n'.join(lines)
