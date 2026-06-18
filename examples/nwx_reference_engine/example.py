# -*- coding: utf-8 -*-
"""
Worked end-to-end example: a formula string -> a corrected statistical map.

Run it (from the gramform repo root)::

    PYTHONPATH=src:examples python -m nwx_reference_engine.example

It builds a small synthetic vertexwise dataset with a known group-difference
cluster and a correlated confound, then fits the single most-used neuroimaging
model straight from a formula string and prints the recovered, TFCE-corrected
map as an ASCII strip.
"""

from __future__ import annotations

import numpy as np

from nwx_reference_engine import run
from nwx_reference_engine.datasets import synthetic_vertexwise

FORMULA = (
    'thk ~ dx + sex + noise(meanFD) '
    '{{ contrasts: dx = dx (t); inference=permutation(tfce, n=1000) }}'
)


def _strip(values: np.ndarray, mark: np.ndarray, glyph: str) -> str:
    return ''.join(glyph if m else '.' for m in mark)


def main() -> None:
    ds = synthetic_vertexwise(seed=0)
    n_obs, n_vtx = ds.imaging.shape

    print('nwx reference engine -- worked vertical slice')
    print('=' * 60)
    print(f'formula : {FORMULA}')
    print(f'data    : {n_obs} subjects x {n_vtx} vertices (chain mesh)')
    print(
        '          dx (group), sex (categorical), meanFD (confound, '
        'partialled out via noise())'
    )
    print()

    results = run(FORMULA, ds.data, ds.imaging, adjacency=ds.adjacency, seed=1)
    r = results['dx']

    print(f'design  : {list(r.design_columns)}')
    print(f'dof     : {r.dof}   correction: {r.correction}')
    print()

    sig = r.corrected < 0.05
    print('true dx-effect cluster (where the signal really is):')
    print('   ' + _strip(ds.true_mask, ds.true_mask, '#'))
    print('recovered (TFCE-FWE corrected p < 0.05):')
    print('   ' + _strip(sig, sig, '*'))
    print()

    detected = int((sig & ds.true_mask).sum())
    missed = int((~sig & ds.true_mask).sum())
    false_pos = int((sig & ~ds.true_mask).sum())
    print(
        f'recovered {detected}/{int(ds.true_mask.sum())} true vertices, '
        f'{missed} missed, {false_pos} false positives.'
    )
    peak = int(np.argmax(np.abs(r.stat)))
    print(
        f'peak |t| = {abs(r.stat[peak]):.2f} at vertex {peak} '
        f'(corrected p = {r.corrected[peak]:.3f}).'
    )


if __name__ == '__main__':
    main()
