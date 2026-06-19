# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
``nwx`` <-> BIDS Stats Models bridge (spec §10)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``import_bids_model(model)`` reads a BIDS Stats Models ``model.json`` (as a
dict) into the typed :class:`~gramform.grammars.nwx.spec.ModelGraph` IR;
``export_bids_model(graph)`` writes the reverse. The IR is already node + edge
shaped (it deliberately mirrors BIDS-SM), so the mapping is direct. Pure: no
array, no ``jax``/``nitrix``.

**Export is narrowing, not symmetric.** A ``ModelSpec`` carries far more than a
BIDS-SM ``Model`` can hold, so export is **strict**: a graph that uses a
construct BIDS-SM has no representation for is *refused* (``BidsExportError``),
never silently lowered. :func:`validate_exportable` is the static check;
:func:`export_bids_model` raises on its ERROR diagnostics. The
BIDS-SM-representable subset is the multi-level GLM graph -- a
Gaussian/identity fixed design (``Model.X``) + ``Contrasts`` + cope-carrying
``Edges`` -- which is exactly what BIDS-SM exists to express and what ``>>``
pipelines / imported graphs already are.

REFUSED on export (no faithful BIDS-SM form -- would change the model):
random effects, smooths, residualisation (``~|``), in-model ``noise()``
partials, a non-Gaussian family / non-identity link, an explicit error
structure, and a design factor that is not a plain column (a frame
``Referent``, a ``CovariateRef``, or a ``PyExpr`` -- the last awaits a
``Model.Formula`` exporter). INTENTIONALLY DROPPED (out of BIDS-SM's scope, not
a model property it represents): the response (implicit in BIDS-SM), node- and
graph-level ``inference`` (the executor's concern), and ``estimation`` details.

Mapping (schema: ``bsmschema``; field names verbatim from the BIDS Stats Models
spec):

==========================  ===================================================
BIDS Stats Models           nwx IR
==========================  ===================================================
``Nodes[i]``                a :class:`ModelNode`
``Node.Level``              :class:`Level` (``Run``/``Session``/``Subject``/
                            ``Dataset`` -> the lower-cased enum)
``Node.Name``               ``ModelNode.name``
``Node.GroupBy``            ``ModelNode.group_by``
``Node.Model.Type``         ``meta`` -> ``Combine.MIXED``, else ``FIXED``
``Node.Model.X``            ``ModelSpec.fixed`` (``"1"`` -> the intercept; each
                            other column -> a ``Lookup`` term)
``Node.Model.Formula``      parsed via the nwx processor when ``X`` is absent
``Node.Contrasts[i]``       a :class:`ContrastSpec` (``ConditionList`` zipped
                            with ``Weights``; ``Test`` ``t``/``F``)
``Node.DummyContrasts``     one unit :class:`ContrastSpec` per design column
``Edges[i]``                an :class:`Edge`
``Edge.Source``/``.Destination``  ``Edge.source``/``.dest``
``Edge.Filter``             non-``contrast`` keys -> ``Edge.filter`` pairs; the
                            special ``"contrast"`` -> ``Edge.carry.contrast``
==========================  ===================================================

**Not mapped** (documented limitations): ``Node.Transformations`` (general BIDS
design transforms -- ``Scale``/``Convolve``/``Factor``/... -- are not the nwx
confound-formula vocabulary, so there is no faithful ``CovariateProgram`` map
yet; a confound-style transform string would go through
:func:`~gramform.grammars.nwx.covariate.lower_covariates`); ``Input``;
``Model.Options``. The BIDS-SM field names here are taken from the live
``bsmschema`` schema (verified over the web, 2026-06).
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from gramform.grammars.nwx.spec import (
    INTERCEPT,
    Carry,
    Combine,
    Const,
    ContrastSpec,
    CovariateRef,
    Diagnostic,
    Edge,
    FactorSpec,
    Family,
    Level,
    Link,
    Lookup,
    ModelGraph,
    ModelNode,
    ModelSpec,
    PyExpr,
    Referent,
    ResponseSpec,
    Severity,
    TermSpec,
    Test,
)


class BidsImportError(ValueError):
    """A malformed / unsupported BIDS Stats Models document."""


class BidsExportError(ValueError):
    """A ``ModelGraph`` that cannot be faithfully written as BIDS Stats Models
    (it uses a construct BIDS-SM has no representation for -- see
    :func:`validate_exportable`)."""


def import_bids_file(path: str | Path) -> ModelGraph:
    """Read a BIDS Stats Models ``model.json`` file into a ``ModelGraph``."""
    with open(path, encoding='utf-8') as handle:
        return import_bids_model(json.load(handle))


def import_bids_model(model: Mapping[str, Any]) -> ModelGraph:
    """Map a parsed BIDS Stats Models document onto a ``ModelGraph``."""
    if not isinstance(model, Mapping):
        raise BidsImportError('a BIDS Stats Models document must be an object')
    nodes = tuple(_node(n) for n in model.get('Nodes', ()))
    edges = tuple(_edge(e) for e in model.get('Edges', ()))
    return ModelGraph(nodes=nodes, edges=edges)


# ---------------------------------------------------------------------------
# nodes
# ---------------------------------------------------------------------------


def _node(node: Mapping[str, Any]) -> ModelNode:
    name = node.get('Name')
    if not name:
        raise BidsImportError('a Node needs a Name')
    level = _level(node.get('Level'))
    model = node.get('Model') or {}
    x = list(model.get('X', ()))
    fixed = _x_to_fixed(x) if x else _formula_to_fixed(model.get('Formula'))
    combine = Combine.MIXED if model.get('Type') == 'meta' else Combine.FIXED
    estimands = _contrasts(node.get('Contrasts')) + _dummy_contrasts(
        node.get('DummyContrasts'), x
    )
    spec = ModelSpec(
        response=ResponseSpec(terms=()),
        fixed=fixed,
        estimands=estimands,
    )
    return ModelNode(
        name=str(name),
        level=level,
        group_by=tuple(str(g) for g in node.get('GroupBy', ())),
        combine=combine,
        spec=spec,
    )


def _level(value: Any) -> Level:
    try:
        return Level(str(value).lower())
    except ValueError:
        raise BidsImportError(
            f'unknown BIDS Level {value!r} (Run/Session/Subject/Dataset)'
        )


def _x_to_fixed(x: Sequence[str]) -> tuple[TermSpec, ...]:
    """``Model.X`` columns -> fixed terms (``"1"`` is the intercept)."""
    return tuple(
        INTERCEPT
        if str(col) == '1'
        else TermSpec((FactorSpec(Lookup(str(col))),))
        for col in x
    )


def _formula_to_fixed(formula: Any) -> tuple[TermSpec, ...]:
    """A ``Model.Formula`` (Wilkinson RHS) parsed via the nwx processor. Best
    effort -- patsy-only constructs (e.g. ``C(x)``) are not supported."""
    if not formula:
        return ()
    from gramform.grammars.nwx.transform import get_processor

    try:
        graph = get_processor()(f'__bids_response__ ~ {formula}')
    except ValueError as error:
        raise BidsImportError(
            f'could not parse Model.Formula {formula!r}: {error}'
        )
    return graph.nodes[0].spec.fixed


# ---------------------------------------------------------------------------
# contrasts
# ---------------------------------------------------------------------------


def _contrasts(contrasts: Any) -> tuple[ContrastSpec, ...]:
    if not contrasts:
        return ()
    return tuple(_contrast(c) for c in contrasts)


def _contrast(contrast: Mapping[str, Any]) -> ContrastSpec:
    name = contrast.get('Name')
    if not name:
        raise BidsImportError('a Contrast needs a Name')
    conditions = contrast.get('ConditionList', ())
    weights = contrast.get('Weights', ())
    pairs = tuple(
        (str(cond), float(weight)) for cond, weight in zip(conditions, weights)
    )
    return ContrastSpec(name=str(name), weights=pairs, test=_test(contrast))


def _dummy_contrasts(
    dummy: Any,
    x: Sequence[str],
) -> tuple[ContrastSpec, ...]:
    """``DummyContrasts`` -> one unit (indicator) contrast per named condition,
    or per non-intercept design column when no explicit list is given."""
    if not dummy:
        return ()
    conditions = dummy.get('Contrasts') or [c for c in x if str(c) != '1']
    test = _test(dummy)
    return tuple(
        ContrastSpec(
            name=str(cond),
            weights=((str(cond), 1.0),),
            test=test,
        )
        for cond in conditions
    )


def _test(obj: Mapping[str, Any]) -> Test:
    return Test.F if str(obj.get('Test', 't')) == 'F' else Test.T


# ---------------------------------------------------------------------------
# edges
# ---------------------------------------------------------------------------


def _edge(edge: Mapping[str, Any]) -> Edge:
    source, dest = edge.get('Source'), edge.get('Destination')
    if not source or not dest:
        raise BidsImportError('an Edge needs a Source and a Destination')
    carry_contrast = ''
    pairs: list[tuple[str, str]] = []
    for key, values in (edge.get('Filter') or {}).items():
        seq = values if isinstance(values, (list, tuple)) else [values]
        if str(key).lower() == 'contrast':
            carry_contrast = str(seq[0]) if seq else ''
        else:
            pairs.extend((str(key), str(value)) for value in seq)
    return Edge(
        source=str(source),
        dest=str(dest),
        filter=tuple(pairs),
        carry=Carry(contrast=carry_contrast, quantities='cope_varcope'),
    )


# ===========================================================================
# export direction: ModelGraph -> BIDS Stats Models (strict)
# ===========================================================================

#: The only family/link a BIDS-SM ``Model`` (a GLM/OLS design) represents.
_EXPORTABLE_FAMILY = Family.GAUSSIAN
_EXPORTABLE_LINK = Link.IDENTITY


def export_bids_file(
    graph: ModelGraph,
    path: str | Path,
    *,
    name: str = 'nwx-model',
    indent: int = 2,
) -> None:
    """Write ``graph`` to a BIDS Stats Models ``model.json`` file (strict)."""
    document = export_bids_model(graph, name=name)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(document, handle, indent=indent)


def export_bids_model(
    graph: ModelGraph,
    *,
    name: str = 'nwx-model',
) -> dict[str, Any]:
    """Map a ``ModelGraph`` onto a BIDS Stats Models document (strict).

    Refuses, with :class:`BidsExportError`, any graph that uses a construct
    BIDS-SM cannot represent (see :func:`validate_exportable`); the message
    lists every offending node + reason. ``ModelGraph`` carries no name, so one
    is synthesised from ``name``.
    """
    errors = [
        d for d in validate_exportable(graph) if d.severity is Severity.ERROR
    ]
    if errors:
        detail = '\n'.join(f'  - [{d.where}] {d.message}' for d in errors)
        raise BidsExportError(
            'graph is not representable as BIDS Stats Models:\n' + detail
        )
    return {
        'Name': name,
        'BIDSModelVersion': '1.0.0',
        'Nodes': [_export_node(node) for node in graph.nodes],
        'Edges': [_export_edge(edge) for edge in graph.edges],
    }


def validate_exportable(graph: ModelGraph) -> tuple[Diagnostic, ...]:
    """Statically check that ``graph`` is BIDS-SM-representable. Returns an
    ERROR :class:`Diagnostic` per node-level construct BIDS-SM has no faithful
    form for; an empty tuple means :func:`export_bids_model` will succeed."""
    diagnostics: list[Diagnostic] = []
    for node in graph.nodes:
        diagnostics.extend(_node_export_errors(node))
    return tuple(diagnostics)


def _node_export_errors(node: ModelNode) -> Iterator[Diagnostic]:
    spec = node.spec
    where = node.name

    def error(message: str) -> Diagnostic:
        return Diagnostic(
            Severity.ERROR, 'bids-export-unrepresentable', message, where
        )

    if spec.random:
        yield error(
            'random effects have no BIDS-SM form; mixed models are expressed '
            'through the node graph (a meta node over copes), not lme4 bars'
        )
    if spec.smooth:
        yield error('smooth (GAM) terms have no BIDS-SM design representation')
    if spec.residualise:
        yield error(
            'residualisation (`~|`) is a data transform with no BIDS-SM form'
        )
    if spec.partial:
        yield error(
            'an in-model noise() partial has no BIDS-SM form (BIDS-SM has no '
            'computed-but-unreported nuisance block)'
        )
    fam = spec.family
    if (
        fam.family is not _EXPORTABLE_FAMILY
        or fam.link is not _EXPORTABLE_LINK
    ):
        yield error(
            f'family {fam.family.value!r}/link {fam.link.value!r} is not the '
            'Gaussian/identity GLM that a BIDS-SM Model represents'
        )
    if spec.errors is not None:
        yield error('an explicit error structure has no BIDS-SM form')
    for term in spec.fixed:
        if _term_label(term) is None:
            yield error(
                'a design term is not a plain column (a frame referent, a '
                'covariate reference, or a Python-expression column); it '
                'needs a Model.Formula exporter, which is a follow-up'
            )
            break


def _export_node(node: ModelNode) -> dict[str, Any]:
    spec = node.spec
    model: dict[str, Any] = {
        # MIXED -> a meta (FLAME) group node; FIXED -> a GLM node (the inverse
        # of the importer's Type mapping, so import/export round-trips).
        'Type': 'meta' if node.combine is Combine.MIXED else 'glm',
        'X': [_term_label(term) for term in spec.fixed],
    }
    out: dict[str, Any] = {
        'Level': node.level.value.capitalize(),
        'Name': node.name,
        'GroupBy': list(node.group_by),
        'Model': model,
    }
    contrasts = [_export_contrast(c) for c in spec.estimands]
    if contrasts:
        out['Contrasts'] = contrasts
    return out


def _export_contrast(contrast: ContrastSpec) -> dict[str, Any]:
    return {
        'Name': contrast.name,
        'ConditionList': [name for name, _ in contrast.weights],
        'Weights': [weight for _, weight in contrast.weights],
        'Test': contrast.test.value,
    }


def _export_edge(edge: Edge) -> dict[str, Any]:
    filt: dict[str, list[str]] = {}
    for key, value in edge.filter:
        filt.setdefault(key, []).append(value)
    if edge.carry.contrast:
        # the special "contrast" key carries the upstream cope (spec §10)
        filt['contrast'] = [edge.carry.contrast]
    out: dict[str, Any] = {'Source': edge.source, 'Destination': edge.dest}
    if filt:
        out['Filter'] = filt
    return out


def _term_label(term: TermSpec) -> str | None:
    """A fixed term -> its BIDS-SM ``Model.X`` column label, or ``None`` if it
    is not a plain design column. ``"1"`` is the intercept; a single ``Lookup``
    is its name; an interaction is ``"a:b"``. A unit scaling constant is
    absorbed; a frame ``Referent`` / ``CovariateRef`` / ``PyExpr`` factor (or a
    non-unit constant) has no plain-column form -- ``None``."""
    consts = [f.source for f in term.factors if isinstance(f.source, Const)]
    variables = [
        f.source for f in term.factors if not isinstance(f.source, Const)
    ]
    if any(c.value != 1.0 for c in consts):
        return None
    if not variables:
        return '1' if consts else None
    labels: list[str] = []
    for source in variables:
        if isinstance(source, Lookup):
            labels.append(source.name)
        elif isinstance(source, (Referent, CovariateRef, PyExpr)):
            return None
        else:
            return None
    return ':'.join(labels)
