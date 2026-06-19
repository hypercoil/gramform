# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
``nwx`` static validation (spec §8)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``validate(graph)`` walks a *successfully parsed* :class:`ModelGraph` and
returns a tuple of :class:`Diagnostic` s -- a complete static report, before
any data. It is pure (no array, no jax/nitrix).

It covers the §8 checks that survive parsing. Several §8 items are already
**hard errors at parse time** (the structural-singularity guard;
``residualise=nonaggressive`` without a ``signal()`` set; ``signal()`` off a
``~|`` RHS), so a graph that reaches ``validate`` is past them. Others are
**parse-resolved and unambiguous in the IR** (reserved-name shadowing, the
``.`` complement-vs-inbound-cope disambiguation, directive scoping) or
**data-dependent** (grouping-factor categoricality, ``group_by`` column
resolution), which a data-free layer cannot check. What remains, and is
checked here:

- **WARNING** -- a smooth ``by=`` factor without its parametric main effect
  (the identifiability check deferred from Phase 4); aggressive residualisation
  given a ``signal()`` set (ignored); and a **backend-awareness roll-up**:
  every IR feature whose nitrix kernel is not yet shipped (spec §7), as
  Diagnostics so a report is complete without capturing the parse-time
  warnings.
- **ERROR** -- multi-level graph integrity: each ``Edge`` endpoint names a real
  node, each ``Edge.carry.contrast`` resolves to a real upstream
  ``ContrastSpec``, and the edges form a DAG.
"""

from __future__ import annotations

from collections.abc import Iterator

from gramform.grammars.nwx import backend
from gramform.grammars.nwx.spec import (
    BasisKind,
    Diagnostic,
    Lookup,
    Mode,
    ModelGraph,
    ModelNode,
    ModelSpec,
    Severity,
)

#: Smooth bases whose ``by=`` second arg is a slope variable, not a factor.
_RE_BASES = frozenset({BasisKind.RE, BasisKind.FS})


def validate(graph: ModelGraph) -> tuple[Diagnostic, ...]:
    """Statically validate a parsed :class:`ModelGraph` (spec §8)."""
    diagnostics: list[Diagnostic] = []
    for node in graph.nodes:
        diagnostics.extend(_validate_node(node))
    diagnostics.extend(_validate_graph(graph))
    return tuple(diagnostics)


def _warn(code: str, message: str, where: str) -> Diagnostic:
    return Diagnostic(Severity.WARNING, code, message, where)


def _error(code: str, message: str, where: str) -> Diagnostic:
    return Diagnostic(Severity.ERROR, code, message, where)


# ---------------------------------------------------------------------------
# node-level checks
# ---------------------------------------------------------------------------


def _validate_node(node: ModelNode) -> Iterator[Diagnostic]:
    yield from _check_smooth_by(node.name, node.spec)
    yield from _check_residualise(node.name, node.spec)
    yield from _check_backend_awareness(node.name, node.spec)


def _main_effects(spec: ModelSpec) -> frozenset[str]:
    """The names of the single-factor ``Lookup`` main effects in the design."""
    return frozenset(
        term.factors[0].source.name
        for term in spec.fixed
        if len(term.factors) == 1
        and isinstance(term.factors[0].source, Lookup)
    )


def _check_smooth_by(where: str, spec: ModelSpec) -> Iterator[Diagnostic]:
    """A factor ``by=`` smooth needs its parametric main effect for
    identifiability (spec §4.3/§8). ``by_kind`` is data-dependent so we warn
    whenever the ``by`` variable is absent from the design -- the harmless
    continuous-``by`` case aside, this is the onboarding catch from §11
    (``y ~ s(age, by=dx)`` without ``dx``). The ``re``/``fs`` GAMM bridges use
    ``by`` as a *slope* variable, not a factor, and are exempt."""
    mains = _main_effects(spec)
    for smooth in spec.smooth:
        by = smooth.by
        if by is None or smooth.basis in _RE_BASES:
            continue
        if isinstance(by.source, Lookup) and by.source.name not in mains:
            yield _warn(
                'smooth-by-main-effect',
                f'smooth by={by.source.name!r} has no parametric main effect; '
                f'a factor-by smooth is unidentifiable without it (spec §4.3)',
                where,
            )


def _check_residualise(where: str, spec: ModelSpec) -> Iterator[Diagnostic]:
    """Aggressive residualisation ignores any ``signal()`` set (spec §4.6).
    (Non-aggressive without ``signal()`` is a hard error at parse time.)"""
    for res in spec.residualise:
        if res.mode is Mode.AGGRESSIVE and res.signal:
            yield _warn(
                'residualise-aggressive-signal',
                'aggressive residualisation ignores the signal() set; use '
                '{{ residualise=nonaggressive }} to preserve shared variance',
                where,
            )


def _check_backend_awareness(
    where: str,
    spec: ModelSpec,
) -> Iterator[Diagnostic]:
    """Roll up every IR feature the reference backend cannot yet run, so
    ``validate`` gives a complete forward-compatibility report. nitrix
    stats-suite v3 ships nwx's whole v1 scope (see
    :mod:`gramform.grammars.nwx.backend`), so this is now a short residual:
    non-canonical links, the ``soft`` residualise mode, RFT inference, and a
    non-Gaussian random *slope*."""
    fam = spec.family
    if not backend.link_shipped(fam.link):
        yield _backend(
            f'link {fam.link.value!r} is not a nitrix built-in (only the '
            'canonical identity/log/logit ship; others need a hand-built '
            'Family)',
            where,
        )

    # GLMM random slope: glmm_fit ships scalar RE only; a non-scalar random
    # effect under a non-Gaussian family is the Tier-2 deferral (a Gaussian
    # random slope is shipped via lme_fit R2, so it is not flagged).
    for re in spec.random:
        if backend.glmm_random_slope_unshipped(re.structure, fam.family):
            yield _backend(
                f'a {re.structure.value!r} random effect under family '
                f'{fam.family.value!r} (a non-Gaussian random slope) is not '
                'yet shipped; glmm_fit fits a scalar random effect only',
                where,
            )

    for res in spec.residualise:
        if not backend.residualise_mode_shipped(res.mode):
            yield _backend(
                f'residualise={res.mode.value!r} is not yet shipped (nitrix '
                'ships aggressive + nonaggressive)',
                where,
            )

    inf = spec.inference
    if inf is not None and not backend.inference_correction_shipped(
        inf.correction
    ):
        yield _backend(
            f'inference correction {inf.correction!r} is not yet shipped '
            '(nitrix ships permutation/TFCE/cluster/FDR/Bonferroni)',
            where,
        )


def _backend(detail: str, where: str) -> Diagnostic:
    return _warn('backend-unshipped', detail, where)


# ---------------------------------------------------------------------------
# graph-level checks (multi-level integrity)
# ---------------------------------------------------------------------------


def _validate_graph(graph: ModelGraph) -> Iterator[Diagnostic]:
    names = {node.name for node in graph.nodes}
    by_name = {node.name: node for node in graph.nodes}

    for edge in graph.edges:
        for role, end in (('source', edge.source), ('dest', edge.dest)):
            if end not in names:
                yield _error(
                    'edge-unknown-node',
                    f'edge {role} {end!r} is not a node in the graph',
                    f'{edge.source}->{edge.dest}',
                )
        # the carried contrast must name a real upstream ContrastSpec
        contrast = edge.carry.contrast
        source = by_name.get(edge.source)
        if contrast and source is not None:
            estimands = {c.name for c in source.spec.estimands}
            if contrast not in estimands:
                yield _error(
                    'edge-carry-unresolved',
                    f'edge carries contrast {contrast!r} but its source node '
                    f'{edge.source!r} defines no such contrast',
                    f'{edge.source}->{edge.dest}',
                )

    if _has_cycle(graph):
        yield _error(
            'graph-cycle',
            'the multi-level graph has a cycle; stage edges must form a DAG',
            'graph',
        )


def _has_cycle(graph: ModelGraph) -> bool:
    adjacency: dict[str, list[str]] = {}
    for edge in graph.edges:
        adjacency.setdefault(edge.source, []).append(edge.dest)
    visiting: set[str] = set()
    done: set[str] = set()

    def _descend(node: str) -> bool:
        visiting.add(node)
        for nxt in adjacency.get(node, ()):
            if nxt in visiting:
                return True
            if nxt not in done and _descend(nxt):
                return True
        visiting.discard(node)
        done.add(node)
        return False

    return any(
        node.name not in done and _descend(node.name) for node in graph.nodes
    )
