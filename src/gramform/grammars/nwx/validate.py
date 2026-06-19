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

from gramform.grammars.nwx.spec import (
    BasisKind,
    Diagnostic,
    Family,
    Link,
    Lookup,
    Mode,
    ModelGraph,
    ModelNode,
    ModelSpec,
    Severity,
    Structure,
)

#: Features whose nitrix kernel ships in v1 (the rest warn, spec §7).
_SHIPPED_FAMILIES = frozenset(
    {Family.GAUSSIAN, Family.BINOMIAL, Family.POISSON}
)
_SHIPPED_LINKS = frozenset({Link.IDENTITY, Link.LOG, Link.LOGIT})
_SHIPPED_BASES = frozenset(
    {BasisKind.PS, BasisKind.CC, BasisKind.TPRS, BasisKind.TENSOR}
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
    """Roll up every IR feature whose nitrix kernel is not yet shipped (spec
    §7), so ``validate`` gives a complete forward-compatibility report."""
    fam = spec.family
    if fam.family not in _SHIPPED_FAMILIES:
        yield _backend(f'family {fam.family.value!r} (FR §4)', where)
    if fam.link not in _SHIPPED_LINKS:
        yield _backend(f'link {fam.link.value!r} (FR §4)', where)

    if len(spec.random) > 1:
        yield _backend(
            'multiple random effects (nested/crossed) lower onto lme_fit '
            '(FR §1.1 R3/R4)',
            where,
        )
    for re in spec.random:
        if re.structure is not Structure.SCALAR:
            yield _backend(
                f'random-effect structure {re.structure.value!r} lowers onto '
                'lme_fit (FR §1.1 R2)',
                where,
            )

    for smooth in spec.smooth:
        if smooth.basis not in _SHIPPED_BASES:
            yield _backend(
                f'smooth basis {smooth.basis.value!r} (FR §2 / §3.1)', where
            )

    if spec.errors is not None:
        if spec.errors.correlation is not None:
            yield _backend(
                f'correlation={spec.errors.correlation.kind} (FR §1.4)', where
            )
        if spec.errors.heteroscedasticity is not None:
            yield _backend(
                f'weights={spec.errors.heteroscedasticity.kind} (FR §6)', where
            )

    est = spec.estimation
    if est.se in ('robust', 'cluster'):
        yield _backend(f'se={est.se} (FR §6.2)', where)
    if est.dof in ('satterthwaite', 'kr'):
        yield _backend(f'dof={est.dof} (FR §1.3)', where)

    for res in spec.residualise:
        if res.mode is not Mode.AGGRESSIVE:
            yield _backend(f'residualise={res.mode.value} (FR §5)', where)


def _backend(detail: str, where: str) -> Diagnostic:
    return _warn(
        'backend-unshipped',
        f'{detail} is gated on the nitrix v3 feature request, not yet shipped',
        where,
    )


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
