# -*- coding: utf-8 -*-
"""
The orchestrator: a formula string (or a pre-built ``ModelGraph``) plus bound
data -> corrected statistical maps. This is the *engine* half of the nwx
contract (spec §9): it dispatches structurally on which IR fields are
populated and runs the numerics. Phase 2 covers the Gaussian mass-univariate
GLM path; it errors helpfully on IR it does not yet lower.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from gramform.grammars.nwx.spec import (
    Family,
    Link,
    ModelGraph,
    ModelNode,
    Test,
)
from gramform.grammars.nwx.transform import get_processor
from nwx_reference_engine.glm import f_contrast, glm_fit, t_contrast
from nwx_reference_engine.inference import (
    Adjacency,
    bonferroni,
    fdr_bh,
    permutation_test,
)
from nwx_reference_engine.materialise import Design, EngineError, materialise

__all__ = ['EngineError', 'Result', 'run', 'fit_node']


@dataclass(frozen=True)
class Result:
    """Per-contrast result over the mass axis."""

    contrast: str
    stat_kind: str  # 't' or 'F'
    stat: np.ndarray  # observed statistic map (n_mass,)
    pvalue: np.ndarray  # uncorrected parametric p-map
    corrected: np.ndarray  # corrected p-map
    correction: str  # human-readable description of the correction
    dof: int
    n_obs: int
    design_columns: tuple[str, ...]


def _check_supported(node: ModelNode) -> None:
    spec = node.spec
    fam = spec.family
    if fam.family is not Family.GAUSSIAN or fam.link is not Link.IDENTITY:
        raise EngineError(
            f'Phase-2 engine only lowers Gaussian/identity; got '
            f'{fam.family.value}/{fam.link.value}'
        )
    for field_name in ('random', 'smooth', 'residualise'):
        if getattr(spec, field_name):
            raise EngineError(
                f'IR field {field_name!r} is populated but the Phase-2 engine '
                'does not lower it yet (random effects / smooths / '
                'residualisation are later phases)'
            )
    if spec.errors is not None:
        raise EngineError('error/correlation structures are not yet lowered')


def _cluster_threshold(
    stat_kind: str,
    dof: int,
    override: float | None,
) -> float:
    if override is not None:
        return override
    # default cluster-forming threshold at p ~ 0.001 (one-sided in magnitude)
    if stat_kind == 'F':
        return float(stats.f.isf(0.001, 1, dof))
    return float(stats.t.isf(0.001, dof))


def _apply_inference(
    node: ModelNode,
    design: Design,
    contrast_vec: np.ndarray,
    stat: np.ndarray,
    pvalue: np.ndarray,
    stat_kind: str,
    *,
    adjacency: Adjacency | None,
    n_perm: int | None,
    cluster_threshold: float | None,
    seed: int,
    dof: int,
) -> tuple[np.ndarray, str]:
    spec = node.spec.inference
    if spec is None:
        return pvalue, 'uncorrected (parametric, no inference directive)'

    if spec.kind == 'permutation':
        enhancement = spec.enhancement or 'voxel'
        n = n_perm if n_perm is not None else (spec.n_perm or 1000)
        threshold = (
            _cluster_threshold(stat_kind, dof, cluster_threshold)
            if enhancement in ('cluster_extent', 'cluster_mass')
            else None
        )
        result = permutation_test(
            design.Y,
            design.X,
            contrast_vec,
            stat_kind=stat_kind,
            enhancement=enhancement,
            n_perm=n,
            adjacency=adjacency,
            threshold=threshold,
            seed=seed,
        )
        return result.corrected, f'permutation-{enhancement} FWE (n={n})'

    # parametric
    correction = spec.correction
    if correction == 'fdr':
        return fdr_bh(pvalue), 'FDR (Benjamini-Hochberg)'
    if correction == 'bonferroni':
        return bonferroni(pvalue), 'Bonferroni'
    if correction is None:
        return pvalue, 'uncorrected (parametric)'
    raise EngineError(
        f'parametric correction {correction!r} is not supported by the '
        'Phase-2 engine'
    )


def fit_node(
    node: ModelNode,
    data: pd.DataFrame,
    imaging: np.ndarray | None,
    *,
    adjacency: Adjacency | None = None,
    n_perm: int | None = None,
    cluster_threshold: float | None = None,
    seed: int = 0,
) -> dict[str, Result]:
    """Fit one model node and evaluate every estimand."""
    _check_supported(node)
    design = materialise(node, data, imaging)
    fit = glm_fit(design.Y, design.X)

    if not node.spec.estimands:
        raise EngineError(
            'no contrasts to evaluate; add a `{{ contrasts: ... }}` directive'
        )

    results: dict[str, Result] = {}
    for contrast in node.spec.estimands:
        c = design.contrast_vector(contrast)
        if contrast.test is Test.F:
            stat_kind = 'F'
            stat, pvalue = f_contrast(fit, c[None, :])
        else:
            stat_kind = 't'
            stat, pvalue = t_contrast(fit, c)
        corrected, correction = _apply_inference(
            node,
            design,
            c,
            stat,
            pvalue,
            stat_kind,
            adjacency=adjacency,
            n_perm=n_perm,
            cluster_threshold=cluster_threshold,
            seed=seed,
            dof=fit.dof,
        )
        results[contrast.name] = Result(
            contrast=contrast.name,
            stat_kind=stat_kind,
            stat=stat,
            pvalue=pvalue,
            corrected=corrected,
            correction=correction,
            dof=fit.dof,
            n_obs=design.n_obs,
            design_columns=design.columns,
        )
    return results


def run(
    formula: str,
    data: pd.DataFrame,
    imaging: np.ndarray | None = None,
    *,
    adjacency: Adjacency | None = None,
    n_perm: int | None = None,
    cluster_threshold: float | None = None,
    seed: int = 0,
) -> dict[str, Result]:
    """Parse ``formula`` to a ``ModelGraph`` (via ``nwx``) and fit it.

    The single most-used neuroimaging model, end to end::

        run('thk ~ dx + sex + noise(meanFD) '
            '{{ contrasts: dx = dx (t); '
            'inference=permutation(tfce, n=1000) }}',
            data, imaging, adjacency=adj)
    """
    graph: ModelGraph = get_processor()(formula)
    if len(graph.nodes) != 1:
        raise EngineError(
            f'Phase-2 engine handles single-node models; got '
            f'{len(graph.nodes)} nodes (frames / multi-level are later phases)'
        )
    return fit_node(
        graph.nodes[0],
        data,
        imaging,
        adjacency=adjacency,
        n_perm=n_perm,
        cluster_threshold=cluster_threshold,
        seed=seed,
    )
