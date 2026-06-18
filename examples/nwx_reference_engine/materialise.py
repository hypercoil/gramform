# -*- coding: utf-8 -*-
"""
Design materialisation: a one-node ``ModelGraph`` + a covariate frame -> the
numeric design ``(Y, X)`` plus the coefficient bookkeeping contrasts need.

Supported Phase-2 term sources: ``Const`` (intercept), ``Lookup`` (numeric
main effect, or treatment-coded categorical), and interactions thereof.
``partial`` (``noise()``) terms are materialised into the *same* design matrix
(so a full-model fit yields the FWL partial coefficient) but tracked separately
so contrasts never load on them. Unsupported sources raise ``EngineError``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gramform.grammars.nwx.spec import (
    Const,
    ContrastSpec,
    FactorSpec,
    Lookup,
    ModelNode,
    TermSpec,
)


class EngineError(ValueError):
    """The reference engine cannot lower this IR / bind this data."""


@dataclass(frozen=True)
class Design:
    """A materialised design ready for the GLM."""

    Y: np.ndarray  # (n_obs, n_mass)
    X: np.ndarray  # (n_obs, n_cols) -- full design incl. partial columns
    columns: tuple[str, ...]  # a label per design column
    signal_cols: tuple[int, ...]  # reported (non-partial) column indices
    partial_cols: tuple[int, ...]  # in-model nuisance column indices
    coef_index: dict[str, tuple[int, ...]]  # coefficient name -> columns

    @property
    def n_obs(self) -> int:
        return self.X.shape[0]

    @property
    def n_cols(self) -> int:
        return self.X.shape[1]

    @property
    def n_mass(self) -> int:
        return self.Y.shape[1]

    def contrast_vector(self, contrast: ContrastSpec) -> np.ndarray:
        """Resolve a :class:`ContrastSpec`'s named weights into a length-``p``
        contrast vector over the design columns (zero on partial columns)."""
        c = np.zeros(self.n_cols)
        for name, weight in contrast.weights:
            cols = self.coef_index.get(name)
            if cols is None and name in self.columns:
                cols = (self.columns.index(name),)
            if cols is None:
                raise EngineError(
                    f'contrast {contrast.name!r} references unknown '
                    f'coefficient {name!r}; design columns are '
                    f'{self.columns}'
                )
            if len(cols) != 1:
                raise EngineError(
                    f'contrast {contrast.name!r} loads on multi-column factor '
                    f'{name!r}; name an explicit level (Phase 2 limitation)'
                )
            c[cols[0]] += weight
        if not c.any():
            raise EngineError(
                f'contrast {contrast.name!r} resolved to an all-zero vector'
            )
        return c


def _is_categorical(series: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(
        series
    ) or pd.api.types.is_bool_dtype(series)


def _factor_columns(
    factor: FactorSpec,
    data: pd.DataFrame,
    n_obs: int,
) -> list[tuple[str, np.ndarray]]:
    """Materialise one factor into one or more (label, column) pairs."""
    source = factor.source
    if isinstance(source, Const):
        return [('Intercept', np.full(n_obs, float(source.value)))]
    if isinstance(source, Lookup):
        if source.name not in data.columns:
            raise EngineError(f'covariate {source.name!r} not found in data')
        series = data[source.name]
        if _is_categorical(series):
            levels = sorted(map(str, pd.unique(series.astype(str))))
            reference, rest = levels[0], levels[1:]
            cols = []
            as_str = series.astype(str).to_numpy()
            for level in rest:
                label = f'{source.name}[T.{level}]'
                cols.append((label, (as_str == level).astype(float)))
            if not cols:  # a constant column carries no information
                raise EngineError(
                    f'categorical {source.name!r} has a single level '
                    f'{reference!r}; it is collinear with the intercept'
                )
            return cols
        return [(source.name, series.to_numpy(dtype=float))]
    raise EngineError(
        f'term source {type(source).__name__} is not supported by the '
        f'Phase-2 reference engine (Lookup / Const only)'
    )


def _term_columns(
    term: TermSpec,
    data: pd.DataFrame,
    n_obs: int,
) -> list[tuple[str, np.ndarray]]:
    """Materialise a term (a product of factors) into design columns.

    An interaction is the elementwise product across factors; a categorical
    factor contributes its set of treatment-coded columns, so the interaction
    is the Cartesian product of the per-factor column sets.
    """
    blocks = [_factor_columns(f, data, n_obs) for f in term.factors]
    columns = blocks[0]
    for block in blocks[1:]:
        columns = [
            (
                f'{a_label}:{b_label}' if a_label != 'Intercept' else b_label,
                a_col * b_col,
            )
            for a_label, a_col in columns
            for b_label, b_col in block
        ]
    return columns


def materialise(
    node: ModelNode,
    data: pd.DataFrame,
    imaging: np.ndarray | None,
) -> Design:
    """Assemble the design for one model node."""
    spec = node.spec
    n_obs = len(data)

    if imaging is not None:
        Y = np.asarray(imaging, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        if Y.shape[0] != n_obs:
            raise EngineError(
                f'imaging has {Y.shape[0]} rows but data has {n_obs}'
            )
    else:
        if len(spec.response.terms) != 1:
            raise EngineError(
                'without imaging, the response must be a single covariate term'
            )
        resp = _term_columns(spec.response.terms[0], data, n_obs)
        if len(resp) != 1:
            raise EngineError('response must resolve to a single column')
        Y = resp[0][1][:, None]

    labels: list[str] = []
    columns: list[np.ndarray] = []
    signal_cols: list[int] = []
    partial_cols: list[int] = []
    coef_index: dict[str, list[int]] = {}

    def _add(term: TermSpec, *, partial: bool) -> None:
        for label, col in _term_columns(term, data, n_obs):
            idx = len(columns)
            labels.append(label)
            columns.append(col)
            (partial_cols if partial else signal_cols).append(idx)
            coef_index.setdefault(label, []).append(idx)
            # Register the bare factor name as a contrast alias, but only for
            # main effects (single-factor terms) so an interaction column does
            # not shadow the main-effect coefficient.
            if len(term.factors) == 1 and isinstance(
                term.factors[0].source, Lookup
            ):
                name = term.factors[0].source.name
                if name != label:
                    coef_index.setdefault(name, []).append(idx)

    for term in spec.fixed:
        _add(term, partial=False)
    for term in spec.partial:
        _add(term, partial=True)

    if not columns:
        raise EngineError('design has no columns')

    X = np.column_stack(columns)
    return Design(
        Y=Y,
        X=X,
        columns=tuple(labels),
        signal_cols=tuple(signal_cols),
        partial_cols=tuple(partial_cols),
        coef_index={k: tuple(v) for k, v in coef_index.items()},
    )
