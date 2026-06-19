# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
``nwx`` <- BIDS Stats Models importer (read direction, spec §10)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``import_bids_model(model)`` reads a BIDS Stats Models ``model.json`` (as a
dict) into the typed :class:`~gramform.grammars.nwx.spec.ModelGraph` IR. The IR
is already node + edge shaped (it deliberately mirrors BIDS-SM), so the mapping
is direct. Pure: no array, no ``jax``/``nitrix``. Export is a follow-up.

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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from gramform.grammars.nwx.spec import (
    INTERCEPT,
    Carry,
    Combine,
    ContrastSpec,
    Edge,
    FactorSpec,
    Level,
    Lookup,
    ModelGraph,
    ModelNode,
    ModelSpec,
    ResponseSpec,
    TermSpec,
    Test,
)


class BidsImportError(ValueError):
    """A malformed / unsupported BIDS Stats Models document."""


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
