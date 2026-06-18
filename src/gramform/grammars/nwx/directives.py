# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
``nwx`` directive mini-parser (full key set, Phase 4)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Parses the *text* inside a ``{{ ... }}`` directive block into the typed IR
fragments it populates -- :class:`FamilySpec`, :class:`EstimationSpec`,
:class:`ErrorSpec`, :class:`ContrastSpec`, :class:`InferenceSpec`, and the
node-level ``level`` / ``group_by`` / ``combine``. Recognised v1 keys (spec
§4.5): ``family``, ``link``, ``estimator``, ``correlation``, ``weights``,
``se``, ``dof``, ``level``, ``group_by``, ``combine``, ``contrasts``,
``inference``. Unknown keys warn (forward-compat), never error; directives
whose nitrix kernel is not yet shipped warn (:class:`BackendWarning`, §7).

The directive grammar (spec §4.5):

    directives : directive (";" directive)*
    directive  : key "=" value
               | "contrasts" ":" contrast ("," contrast)*
    contrast   : NAME "=" contrast_expr ("(" test ")")?

In this text parser, ``;`` separates directives and (inside a ``contrasts``
clause) ``,`` separates contrasts at parenthesis depth zero. The integrated
*exclusive* ``{{ }}`` lexer state is a separate change; this parser is the
content layer it (and the current trailing-block split) feed.
"""

import re
import warnings
from dataclasses import dataclass, field
from typing import Literal, TypeVar, cast, get_args

from gramform.grammars.nwx.spec import (
    BackendWarning,
    Combine,
    ContrastSpec,
    CorrelationSpec,
    Diagnostic,
    ErrorSpec,
    EstimationSpec,
    FactorSpec,
    Family,
    FamilySpec,
    InferenceSpec,
    Level,
    Link,
    Lookup,
    Mode,
    Severity,
    Test,
    WeightSpec,
)

_Estimator = Literal['ols', 'wls', 'irls', 'reml', 'ml', 'flame']
_SE = Literal['model', 'robust', 'cluster']
_Robust = Literal['hc0', 'hc1', 'hc2', 'hc3']
_Dof = Literal['residual', 'satterthwaite', 'kr']
_CorrKind = Literal['ar1', 'car1', 'cs']
_WeightKind = Literal['varIdent', 'varPower']
_InferKind = Literal['parametric', 'permutation']
_Enhancement = Literal['voxel', 'tfce', 'cluster_extent', 'cluster_mass']
_Correction = Literal['fdr', 'bonferroni', 'fwe', 'rft']

_ESTIMATORS: tuple[str, ...] = get_args(_Estimator)
_SES: tuple[str, ...] = get_args(_SE)
_ROBUSTS: tuple[str, ...] = get_args(_Robust)
_DOFS: tuple[str, ...] = get_args(_Dof)
_CORR_KINDS: tuple[str, ...] = get_args(_CorrKind)
_WEIGHT_KINDS: tuple[str, ...] = get_args(_WeightKind)
_ENHANCEMENTS: tuple[str, ...] = get_args(_Enhancement)
_CORRECTIONS: tuple[str, ...] = get_args(_Correction)

#: Surface aliases for enhancement names.
_ENHANCEMENT_ALIASES: dict[str, str] = {'cluster': 'cluster_extent'}

#: Families / links whose nitrix kernel ships in v1 (the rest warn, spec §7).
_SHIPPED_FAMILIES: frozenset[Family] = frozenset(
    {Family.GAUSSIAN, Family.BINOMIAL, Family.POISSON}
)
_SHIPPED_LINKS: frozenset[Link] = frozenset(
    {Link.IDENTITY, Link.LOG, Link.LOGIT}
)


@dataclass(frozen=True)
class DirectiveSet:
    """The parsed result of a single ``{{ ... }}`` block. Fields left ``None``
    were not specified and should not override IR defaults."""

    family: FamilySpec | None = None
    estimation: EstimationSpec | None = None
    errors: ErrorSpec | None = None
    estimands: tuple[ContrastSpec, ...] = ()
    inference: InferenceSpec | None = None
    residualise: Mode | None = None
    level: Level | None = None
    group_by: tuple[str, ...] = ()
    combine: Combine | None = None
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)


def parse_directives(text: str) -> DirectiveSet:
    """Parse the content of a ``{{ ... }}`` block (braces already stripped)."""
    diagnostics: list[Diagnostic] = []
    family_val: Family | None = None
    link_val: Link | None = None
    estimator_val: _Estimator | None = None
    se_val: _SE | None = None
    robust_val: _Robust | None = None
    cluster_by: str | None = None
    dof_val: _Dof | None = None
    correlation: CorrelationSpec | None = None
    heteroscedasticity: WeightSpec | None = None
    estimands: list[ContrastSpec] = []
    inference: InferenceSpec | None = None
    residualise_val: Mode | None = None
    level_val: Level | None = None
    group_by: tuple[str, ...] = ()
    combine_val: Combine | None = None

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
            se_val, robust_val, cluster_by = _parse_se(value, diagnostics)
        elif key == 'dof':
            dof_val = cast(
                '_Dof | None', _literal(value, _DOFS, 'dof', diagnostics)
            )
        elif key == 'correlation':
            correlation = _parse_correlation(value, diagnostics)
        elif key == 'weights':
            heteroscedasticity = _parse_weights(value, diagnostics)
        elif key == 'level':
            level_val = _enum(Level, value, 'level', diagnostics)
        elif key == 'group_by':
            group_by = tuple(
                v.strip() for v in _split_top(value, ',') if v.strip()
            )
        elif key == 'combine':
            combine_val = _enum(Combine, value, 'combine', diagnostics)
        elif key == 'residualise':
            residualise_val = _enum(Mode, value, 'residualise', diagnostics)
        elif key == 'inference':
            inference = _parse_inference(value, diagnostics)
        else:
            diagnostics.append(
                _warn('directive', f'unknown directive key: {key!r}')
            )

    _backend_awareness(
        family_val, link_val, se_val, dof_val, correlation, heteroscedasticity
    )
    if residualise_val in (Mode.NONAGGRESSIVE, Mode.SOFT):
        _backend(
            f'residualise={residualise_val.value} is gated on nitrix v3 §5'
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
            robust_variant=robust_val,
            cluster_by=cluster_by,
            dof=dof_val,
        )
        if (estimator_val or se_val or dof_val)
        else None
    )
    errors = (
        ErrorSpec(
            correlation=correlation,
            heteroscedasticity=heteroscedasticity,
        )
        if (correlation is not None or heteroscedasticity is not None)
        else None
    )
    return DirectiveSet(
        family=family,
        estimation=estimation,
        errors=errors,
        estimands=tuple(estimands),
        inference=inference,
        residualise=residualise_val,
        level=level_val,
        group_by=group_by,
        combine=combine_val,
        diagnostics=tuple(diagnostics),
    )


# ---------------------------------------------------------------------------
# standard-error, correlation, weights, backend awareness
# ---------------------------------------------------------------------------


def _parse_se(
    value: str,
    diagnostics: list[Diagnostic],
) -> tuple[_SE | None, _Robust | None, str | None]:
    """``se=robust`` / ``se=robust(hc3)`` / ``se=cluster(subject)``."""
    m = _CALL_RE.match(value)
    if not m:
        diagnostics.append(_warn('se', f'malformed se: {value!r}'))
        return None, None, None
    kind = cast('_SE | None', _literal(m.group(1), _SES, 'se', diagnostics))
    arg = (m.group(2) or '').strip()
    robust: _Robust | None = None
    cluster_by: str | None = None
    if kind == 'robust' and arg:
        robust = cast(
            '_Robust | None', _literal(arg, _ROBUSTS, 'se', diagnostics)
        )
    elif kind == 'cluster' and arg:
        cluster_by = arg
    return kind, robust, cluster_by


def _parse_correlation(
    value: str,
    diagnostics: list[Diagnostic],
) -> CorrelationSpec | None:
    """``correlation=ar1(time | g)`` -> a within-group correlation."""
    m = _CALL_RE.match(value)
    if not m or m.group(2) is None:
        diagnostics.append(
            _warn('correlation', f'malformed correlation: {value!r}')
        )
        return None
    kind = m.group(1).lower()
    if kind not in _CORR_KINDS:
        diagnostics.append(
            _warn('correlation', f'unknown correlation: {kind!r}')
        )
        return None
    index_raw, sep, group_raw = m.group(2).partition('|')
    index_raw, group_raw = index_raw.strip(), group_raw.strip()
    if not sep or not index_raw or not group_raw:
        diagnostics.append(
            _warn('correlation', f'expected `kind(index | group)`: {value!r}')
        )
        return None
    return CorrelationSpec(
        kind=cast('_CorrKind', kind),
        index=FactorSpec(Lookup(index_raw)),
        group=FactorSpec(Lookup(group_raw)),
    )


def _parse_weights(
    value: str,
    diagnostics: list[Diagnostic],
) -> WeightSpec | None:
    """``weights=varPower(x)`` / ``weights=varIdent(g)``."""
    m = _CALL_RE.match(value)
    if not m or m.group(2) is None:
        diagnostics.append(_warn('weights', f'malformed weights: {value!r}'))
        return None
    kind = m.group(1)
    if kind not in _WEIGHT_KINDS:
        diagnostics.append(_warn('weights', f'unknown weights: {kind!r}'))
        return None
    arg = m.group(2).strip()
    if not arg:
        diagnostics.append(_warn('weights', 'weights needs an argument'))
        return None
    return WeightSpec(
        kind=cast('_WeightKind', kind), arg=FactorSpec(Lookup(arg))
    )


def _backend_awareness(
    family: Family | None,
    link: Link | None,
    se: _SE | None,
    dof: _Dof | None,
    correlation: CorrelationSpec | None,
    weights: WeightSpec | None,
) -> None:
    """Emit a :class:`BackendWarning` for valid IR whose nitrix kernel is not
    yet shipped (spec §7), so specs stay forward-compatible."""
    if family is not None and family not in _SHIPPED_FAMILIES:
        _backend(f'family {family.value!r} is gated on nitrix v3 §4')
    if link is not None and link not in _SHIPPED_LINKS:
        _backend(f'link {link.value!r} is gated on nitrix v3 §4')
    if se in ('robust', 'cluster'):
        _backend(f'se={se} is gated on nitrix v3 §6.2')
    if dof in ('satterthwaite', 'kr'):
        _backend(f'dof={dof} is gated on nitrix v3 §1.3')
    if correlation is not None:
        _backend(f'correlation={correlation.kind} is gated on nitrix v3 §1.4')
    if weights is not None:
        _backend(f'weights={weights.kind} (heteroscedasticity) is gated on v3')


def _backend(message: str) -> None:
    warnings.warn(message, BackendWarning, stacklevel=3)


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


_E = TypeVar('_E', Family, Link, Level, Combine, Mode)


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
