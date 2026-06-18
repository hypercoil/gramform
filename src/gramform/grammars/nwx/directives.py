# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
``nwx`` directive mini-parser (minimal, Phase 1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Parses the *text* inside a ``{{ ... }}`` directive block into the typed IR
fragments it populates -- :class:`FamilySpec`, :class:`EstimationSpec`,
:class:`ContrastSpec`, :class:`InferenceSpec`. Phase 1 honours only the keys
the runnable slice needs (``family``/``link``/``estimator``/``se``/``dof``/
``contrasts``/``inference``); the full key set + an integrated *exclusive*
lexer state land in Phase 4. Unknown keys warn (forward-compat), never error.

The directive grammar (spec §4.5):

    directives : directive (";" directive)*
    directive  : key "=" value
               | "contrasts" ":" contrast ("," contrast)*
    contrast   : NAME "=" contrast_expr ("(" test ")")?

In this Phase-1 text parser, ``;`` separates directives and (inside a
``contrasts`` clause) ``,`` separates contrasts; neither appears inside a value
in the minimal key set, so a split is unambiguous.
"""

import re
from dataclasses import dataclass, field
from typing import Literal, TypeVar, cast, get_args

from gramform.grammars.nwx.spec import (
    ContrastSpec,
    Diagnostic,
    EstimationSpec,
    Family,
    FamilySpec,
    InferenceSpec,
    Link,
    Severity,
    Test,
)

_Estimator = Literal['ols', 'wls', 'irls', 'reml', 'ml', 'flame']
_SE = Literal['model', 'robust', 'cluster']
_Dof = Literal['residual', 'satterthwaite', 'kr']
_InferKind = Literal['parametric', 'permutation']
_Enhancement = Literal['voxel', 'tfce', 'cluster_extent', 'cluster_mass']
_Correction = Literal['fdr', 'bonferroni', 'fwe', 'rft']

_ESTIMATORS: tuple[str, ...] = get_args(_Estimator)
_SES: tuple[str, ...] = get_args(_SE)
_DOFS: tuple[str, ...] = get_args(_Dof)
_ENHANCEMENTS: tuple[str, ...] = get_args(_Enhancement)
_CORRECTIONS: tuple[str, ...] = get_args(_Correction)

#: Surface aliases for enhancement names.
_ENHANCEMENT_ALIASES: dict[str, str] = {'cluster': 'cluster_extent'}

#: Recognised v1 directive keys (Phase-1 subset).
_KNOWN_KEYS: frozenset[str] = frozenset(
    {'family', 'link', 'estimator', 'se', 'dof', 'inference'}
)


@dataclass(frozen=True)
class DirectiveSet:
    """The parsed result of a single ``{{ ... }}`` block. Fields left ``None``
    were not specified and should not override IR defaults."""

    family: FamilySpec | None = None
    estimation: EstimationSpec | None = None
    estimands: tuple[ContrastSpec, ...] = ()
    inference: InferenceSpec | None = None
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)


def parse_directives(text: str) -> DirectiveSet:
    """Parse the content of a ``{{ ... }}`` block (braces already stripped)."""
    diagnostics: list[Diagnostic] = []
    family_val: Family | None = None
    link_val: Link | None = None
    estimator_val: _Estimator | None = None
    se_val: _SE | None = None
    dof_val: _Dof | None = None
    estimands: list[ContrastSpec] = []
    inference: InferenceSpec | None = None

    for raw in _split_top(text, ';'):
        clause = raw.strip()
        if not clause:
            continue
        head, sep, rest = clause.partition(':')
        if sep and head.strip() == 'contrasts':
            estimands.extend(_parse_contrasts(rest, diagnostics))
            continue
        key, eq, value = clause.partition('=')
        key, value = key.strip(), value.strip()
        if not eq:
            diagnostics.append(
                _warn('directive', f'malformed directive: {clause!r}')
            )
            continue
        if key == 'family':
            family_val = _enum(Family, value, 'family', diagnostics)
        elif key == 'link':
            link_val = _enum(Link, value, 'link', diagnostics)
        elif key == 'estimator':
            estimator_val = cast(
                '_Estimator | None',
                _literal(value, _ESTIMATORS, 'estimator', diagnostics),
            )
        elif key == 'se':
            se_val = cast(
                '_SE | None', _literal(value, _SES, 'se', diagnostics)
            )
        elif key == 'dof':
            dof_val = cast(
                '_Dof | None', _literal(value, _DOFS, 'dof', diagnostics)
            )
        elif key == 'inference':
            inference = _parse_inference(value, diagnostics)
        else:
            diagnostics.append(
                _warn('directive', f'unknown directive key: {key!r}')
            )

    family = (
        FamilySpec(
            family=family_val if family_val is not None else Family.GAUSSIAN,
            link=link_val if link_val is not None else Link.IDENTITY,
        )
        if (family_val is not None or link_val is not None)
        else None
    )
    estimation = (
        EstimationSpec(
            estimator=estimator_val or 'ols',
            se=se_val or 'model',
            dof=dof_val,
        )
        if (estimator_val or se_val or dof_val)
        else None
    )
    return DirectiveSet(
        family=family,
        estimation=estimation,
        estimands=tuple(estimands),
        inference=inference,
        diagnostics=tuple(diagnostics),
    )


# ---------------------------------------------------------------------------
# contrasts
# ---------------------------------------------------------------------------

_TEST_RE = re.compile(r'\(\s*([tF])\s*\)\s*$')
_LINEAR_TERM_RE = re.compile(
    r'\s*([+-]?)\s*'  # optional sign
    r'(?:([0-9]*\.?[0-9]+)\s*\*\s*)?'  # optional scalar coefficient
    r'([^+\-\s*]+)\s*'  # coefficient name
)


def _parse_contrasts(
    clause: str,
    diagnostics: list[Diagnostic],
) -> list[ContrastSpec]:
    out: list[ContrastSpec] = []
    for piece in _split_top(clause, ','):
        body = piece.strip()
        if not body:
            continue
        name, eq, expr = body.partition('=')
        name, expr = name.strip(), expr.strip()
        if not eq or not name:
            diagnostics.append(
                _warn('contrasts', f'malformed contrast: {body!r}')
            )
            continue
        test = Test.T
        m = _TEST_RE.search(expr)
        if m:
            test = Test.T if m.group(1) == 't' else Test.F
            expr = expr[: m.start()].strip()
        weights = _parse_linear(expr, name, diagnostics)
        out.append(ContrastSpec(name=name, weights=weights, test=test))
    return out


def _parse_linear(
    expr: str,
    where: str,
    diagnostics: list[Diagnostic],
) -> tuple[tuple[str, float], ...]:
    weights: list[tuple[str, float]] = []
    pos = 0
    for m in _LINEAR_TERM_RE.finditer(expr):
        if m.start() != pos:  # a gap means an unparsed character
            break
        pos = m.end()
        sign = -1.0 if m.group(1) == '-' else 1.0
        scalar = float(m.group(2)) if m.group(2) else 1.0
        weights.append((m.group(3), sign * scalar))
    if pos != len(expr.strip()) and expr.strip():
        diagnostics.append(
            _warn(f'contrasts:{where}', f'unparsed contrast expr: {expr!r}')
        )
    return tuple(weights)


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------

_CALL_RE = re.compile(r'^\s*([A-Za-z_]\w*)\s*(?:\((.*)\)\s*)?$')


def _parse_inference(
    value: str,
    diagnostics: list[Diagnostic],
) -> InferenceSpec | None:
    m = _CALL_RE.match(value)
    if not m:
        diagnostics.append(
            _warn('inference', f'malformed inference: {value!r}')
        )
        return None
    kind_raw = m.group(1)
    if kind_raw not in get_args(_InferKind):
        diagnostics.append(
            _warn('inference', f'unknown inference kind: {kind_raw!r}')
        )
        return None
    kind = cast(_InferKind, kind_raw)
    enhancement: str | None = None
    correction: str | None = None
    n_perm: int | None = None
    threshold: float | None = None

    for arg in _split_top(m.group(2) or '', ','):
        arg = arg.strip()
        if not arg:
            continue
        k, eq, v = arg.partition('=')
        k, v = k.strip(), v.strip()
        if not eq:  # positional: an enhancement or a correction name
            token = _ENHANCEMENT_ALIASES.get(k, k)
            if token in _ENHANCEMENTS:
                enhancement = token
            elif token in _CORRECTIONS:
                correction = token
            else:
                diagnostics.append(
                    _warn('inference', f'unknown inference arg: {k!r}')
                )
        elif k in ('n', 'n_perm'):
            n_perm = _to_int(v, 'inference', diagnostics)
        elif k in ('threshold', 'cluster_threshold'):
            threshold = _to_float(v, 'inference', diagnostics)
        elif k == 'correction' and v in _CORRECTIONS:
            correction = v
        elif k == 'enhancement':
            token = _ENHANCEMENT_ALIASES.get(v, v)
            if token in _ENHANCEMENTS:
                enhancement = token
        else:
            diagnostics.append(
                _warn('inference', f'unknown inference arg: {k!r}')
            )

    return InferenceSpec(
        kind=kind,
        enhancement=cast(
            '_Enhancement | None',
            enhancement,
        ),
        cluster_threshold=threshold,
        n_perm=n_perm,
        correction=cast('_Correction | None', correction),
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _split_top(text: str, sep: str) -> list[str]:
    """Split ``text`` on ``sep`` at parenthesis depth zero."""
    parts: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(text):
        if ch in '([':
            depth += 1
        elif ch in ')]':
            depth = max(0, depth - 1)
        elif ch == sep and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return parts


_E = TypeVar('_E', Family, Link)


def _enum(
    enum_cls: type[_E],
    value: str,
    field_name: str,
    diagnostics: list[Diagnostic],
) -> _E | None:
    try:
        return enum_cls(value.lower())
    except ValueError:
        diagnostics.append(
            _warn(field_name, f'unknown {field_name}: {value!r}')
        )
        return None


def _literal(
    value: str,
    allowed: tuple[str, ...],
    field_name: str,
    diagnostics: list[Diagnostic],
) -> str | None:
    if value.lower() in allowed:
        return value.lower()
    diagnostics.append(_warn(field_name, f'unknown {field_name}: {value!r}'))
    return None


def _to_int(
    value: str,
    where: str,
    diagnostics: list[Diagnostic],
) -> int | None:
    try:
        return int(value)
    except ValueError:
        diagnostics.append(_warn(where, f'expected integer: {value!r}'))
        return None


def _to_float(
    value: str,
    where: str,
    diagnostics: list[Diagnostic],
) -> float | None:
    try:
        return float(value)
    except ValueError:
        diagnostics.append(_warn(where, f'expected number: {value!r}'))
        return None


def _warn(where: str, message: str) -> Diagnostic:
    return Diagnostic(
        severity=Severity.WARNING,
        code='directive',
        message=message,
        where=where,
    )
