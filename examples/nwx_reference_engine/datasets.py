# -*- coding: utf-8 -*-
"""
Synthetic vertexwise dataset for the worked example and the tests: a small
1-D "surface" (a chain mesh) with a localised group effect, a categorical
nuisance (sex), and a continuous confound (mean framewise displacement) that is
correlated with the imaging signal -- so the engine must partial it out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Dataset:
    data: pd.DataFrame  # covariates (n_obs rows)
    imaging: np.ndarray  # (n_obs, n_vertices)
    adjacency: tuple[np.ndarray, ...]  # chain-mesh vertex adjacency
    true_mask: np.ndarray  # boolean (n_vertices,): where the dx effect is real


def chain_adjacency(n_vertices: int) -> tuple[np.ndarray, ...]:
    """Adjacency of a 1-D chain mesh (each vertex links its neighbours)."""
    return tuple(
        np.array([j for j in (i - 1, i + 1) if 0 <= j < n_vertices])
        for i in range(n_vertices)
    )


def synthetic_vertexwise(
    *,
    seed: int = 0,
    n_obs: int = 60,
    n_vertices: int = 60,
    effect: tuple[int, int] = (24, 36),
    effect_size: float = 1.5,
    confound_loading: float = 0.6,
    noise: float = 1.0,
) -> Dataset:
    """A reproducible vertexwise dataset with a known dx effect cluster."""
    rng = np.random.default_rng(seed)
    dx = rng.integers(0, 2, n_obs)
    sex = np.where(rng.integers(0, 2, n_obs) == 0, 'F', 'M')
    mean_fd = rng.normal(0.0, 1.0, n_obs)

    data = pd.DataFrame({'dx': dx, 'sex': sex, 'meanFD': mean_fd})

    true_mask = np.zeros(n_vertices, dtype=bool)
    true_mask[effect[0] : effect[1]] = True

    Y = rng.normal(0.0, noise, (n_obs, n_vertices))
    Y += np.outer(dx.astype(float), effect_size * true_mask)
    Y += np.outer(mean_fd, confound_loading * np.ones(n_vertices))

    return Dataset(
        data=data,
        imaging=Y,
        adjacency=chain_adjacency(n_vertices),
        true_mask=true_mask,
    )
