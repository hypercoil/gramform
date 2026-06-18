# nwx reference engine — Phase 2 vertical slice

A **separate consumer** of the `nwx` `ModelSpec` IR that proves the contract
end to end on the single most-used neuroimaging model:

> **Gaussian mass-univariate GLM + confounds + t/F contrast + permutation
> (voxel / cluster / TFCE) or FDR / Bonferroni inference.**

It lives **outside `gramform`** on purpose. `nwx` is a pure specification layer
(`string → immutable ModelGraph`, no `jax`/`nitrix`/arrays); the engine is the
*other* side of that boundary — it binds data and runs numerics. The
[CI firewall](../../tests/nwx/test_import_firewall.py) asserts the spec layer
never imports `jax`/`nitrix`; this engine is never imported by `gramform`, so
the firewall is unaffected even though the engine imports `numpy`/`scipy`.

The intended numerical substrate is **`nitrix`** (per the engine-contract
dispatch table in `docs/nwx/spec.md` §9); this reference path uses
`numpy`/`scipy` so it runs anywhere, and its outputs are pinned against
independent linear-algebra oracles in the tests.

## Run it

```bash
# from the gramform repo root (the spec layer lives under src/, the engine
# under examples/ — neither is pip-installed here)
PYTHONPATH=src:examples python -m nwx_reference_engine.example

# tests
PYTHONPATH=src:examples python -m pytest examples/nwx_reference_engine/tests -q
```

Example output (a known group-difference cluster, recovered exactly):

```
design  : ['Intercept', 'dx', 'sex[T.M]', 'meanFD']
dof     : 56   correction: permutation-tfce FWE (n=1000)
true dx-effect cluster:   ........................############........................
recovered (FWE p<0.05):   ........................************........................
recovered 12/12 true vertices, 0 missed, 0 false positives.
```

## Programmatic use

```python
from nwx_reference_engine import run

results = run(
    'thk ~ dx + sex + noise(meanFD) '
    '{{ contrasts: dx = dx (t); inference=permutation(tfce, n=1000) }}',
    data,                 # pandas DataFrame of covariates (n_obs rows)
    imaging,              # np.ndarray (n_obs, n_vertices) — the mass response
    adjacency=adjacency,  # tuple of neighbour-index arrays (for cluster/TFCE)
    seed=1,
)
r = results['dx']
r.stat        # observed t-map        (n_vertices,)
r.pvalue      # uncorrected parametric p-map
r.corrected   # FWE-corrected p-map
r.correction  # 'permutation-tfce FWE (n=1000)'
```

## What it lowers (the contract)

The engine dispatches structurally on which `ModelSpec` fields are populated and
reads exactly:

| IR field | Engine behaviour |
|---|---|
| `response` | the mass response (imaging array), or a single covariate column |
| `fixed` | the reported design (intercept, numeric / treatment-coded categorical, interactions) |
| `partial` (`noise()`) | added to the design; reported coefficients become the FWL partial (`glm_fit` + contrast on signal columns only) |
| `family` | Gaussian / identity only (else `EngineError`) |
| `estimands` | each `ContrastSpec` → t (single linear combination) or F (= t² for one row) |
| `inference` | `permutation(voxel\|cluster_extent\|cluster_mass\|tfce)` → Freedman–Lane max-statistic FWE; `parametric` + `fdr`/`bonferroni` |

Permutation uses **Freedman–Lane** (remove the tested effect, permute the
reduced-model residuals, refit, recompute the statistic) — the standard scheme
for a contrast with nuisance regressors. Spatial enhancement (clusters / TFCE)
uses the supplied vertex adjacency and is two-sided (computed per sign).

## Phase-2 limitations (later phases / `nitrix` v3)

It raises a helpful `EngineError` (or `NotImplementedError`) on IR it does not
yet lower: non-Gaussian families, random effects, smooths, residualisation
frames, error/correlation structures, multi-node (frame / `>>`) graphs,
`PyExpr`/`CovariateRef`/`Referent` term sources, and multi-column-factor
contrasts without an explicit level. These are the surfaces the later nwx
phases and the `nitrix` v3 kernels fill in.

## Layout

```
nwx_reference_engine/
  materialise.py   # ModelGraph + DataFrame -> Design (Y, X, contrast bookkeeping)
  glm.py           # mass-univariate Gaussian OLS + t/F contrasts
  inference.py     # Freedman–Lane permutation (voxel/cluster/TFCE) + FDR/Bonferroni
  engine.py        # run(formula, data, imaging, ...) -> {contrast: Result}
  datasets.py      # synthetic vertexwise dataset (example + tests)
  example.py       # the worked end-to-end run
  tests/           # GLM oracle, inference, end-to-end + contract tests
```
