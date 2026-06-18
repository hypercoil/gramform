# -*- coding: utf-8 -*-
"""
``nwx`` reference engine (Phase 2 vertical slice)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A *separate consumer* of the ``nwx`` ``ModelSpec`` IR -- it lives **outside**
``gramform`` so the model-specification layer stays array-free (the jax/nitrix
firewall). It binds data to a one-node ``ModelGraph`` and lowers it onto
numerics, proving the contract end-to-end on the most-used model:

    Gaussian mass-univariate GLM + confounds + t/F contrast
        + permutation (voxel / cluster / TFCE) or FDR / Bonferroni inference.

The intended substrate is ``nitrix``; this reference path uses ``numpy`` /
``scipy`` so it runs anywhere, and the tests pin its output against an
independent linear-algebra oracle. The engine reads exactly these IR fields:
``ModelSpec.{response, fixed, partial, estimands, family, inference}`` -- and
errors helpfully on IR it does not yet lower (random effects, smooths,
non-Gaussian families, residualisation).
"""

from nwx_reference_engine.engine import EngineError, Result, run

__all__ = ['EngineError', 'Result', 'run']
