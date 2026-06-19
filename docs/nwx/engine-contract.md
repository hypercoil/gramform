# `nwx` — engine contract

> **Status (2026-06-18): the IR→engine contract (spec §9), made executable.**
> `nwx` guarantees a complete, validated `ModelGraph`; an external **engine**
> (a separate consumer — *not* `gramform`, spec §1) guarantees the numerics. This
> doc is the dispatch table the engine implements. It is **informative**: no
> dispatch logic lives in the `nwx` package — `nwx` never decides which `nitrix`
> routine fires (that would couple the spec layer to a backend).
>
> The contract is **executable**: `tests/nwx/dryrun.py` (`dry_run(graph)`) is a
> pure function that emits the intended `nitrix` call sequence for any graph
> *without importing nitrix*, and `tests/nwx/test_engine_contract.py` round-trips
> the §11 examples against it.

## What the engine receives

`(ModelGraph, covariate_df, imaging_data)`. The engine:

1. materialises the node's `CovariateProgram` into covariate columns (expanding
   `Shorthand`/`Derivative`/`Power`/… — `nwx` emitted the program, never the
   array; a term factor that is a `CovariateRef` indexes into it);
2. assembles, per node, the fixed design `X`, the random design `Z_g` (one per
   grouping term), the smooth bases, and the `partial` nuisance block;
3. **dispatches structurally** to the cheapest exact `nitrix` routine (below);
4. **propagates `{cope, varcope}` along `Edge`s** for multi-level FLAME;
5. runs the node-level and graph-level `InferenceSpec`.

## Dispatch table (populated IR fields → `nitrix` route)

Checked **in order** — the first match is the cheapest exact route. `✅` = the
kernel ships today; `⚠️ v3` = gated on the nitrix v3 feature request (spec §7).

| Populated (in priority order) | Route | Status |
|---|---|---|
| node is an `Edge.dest` (fed by copes) | `flame_two_level` (FE/ME by `combine`) | ✅ |
| `residualise` (a `~\|` frame) | `linalg.residualise` (aggressive) | ✅ |
| ″ with `mode=nonaggressive` | `partial_residualise` | ⚠️ v3 §5.1 |
| `smooth` present | `gam_fit` (+ `re`/`fs` GAMM blocks if `random`) | ✅ ps/cc/tp/te; ⚠️ v3 otherwise |
| `random` present, one `scalar` spec | `reml_fit` (R1) | ✅ |
| `random` non-scalar / nested / crossed | `lme_fit` structure-dispatch (R2–R4) | ⚠️ v3 §1.1 |
| `partial` present | `glm_fit` — FWL (design includes the partial block; the contrast loads only on the signal columns) | ✅ |
| only `fixed` | `glm_fit` (family/link) | ✅ Gaussian/Binomial/Poisson; ⚠️ v3 otherwise |

Then, per node: each `ContrastSpec` → `t_contrast` / `f_contrast` (✅ for a GLM;
⚠️ v3 for an LME contrast, which needs the dof machinery, §1.3); and the
node-level `InferenceSpec` → `permutation_test` (Freedman–Lane, voxel / TFCE /
cluster max-stat FWE, ✅) / `fdr_bh` / `bonferroni` (✅) / `rft` (⚠️ v3 §7).

## Multi-level: cope/varcope propagation, FE vs ME

An `Edge` carries `{cope, varcope}` from `source` to `dest`. FLAME needs the
**known** within-level variance: the engine binds the upstream `ContrastSpec`'s
varcope (`c'(XᵀX)⁻¹c·φ`) to `flame_two_level(var_within=…)`. The `Edge.carry`
names which upstream estimand is carried (`nwx` validates it resolves, §8).

`ModelNode.combine` selects the group model:

- **`FIXED`** — fixed-effects: precision-weighted, no between-level variance
  (typical run → subject).
- **`MIXED`** — mixed-effects: estimate the between-level variance σ_b²
  (typical subject → dataset).

A three-level RUN→SUBJECT→DATASET design is **two chained two-level FLAME fits**
with varcope propagation through the middle node (nitrix ships only
`flame_two_level`).

## Worked §11 transcripts (from `dry_run`)

The single most-used neuroimaging model — fully on shipped kernels (**M1**):

```
thk ~ dx + sex + noise(meanFD)
      {{ contrasts: dx = dx (t); inference=permutation(tfce, n=5000) }}

node root: [glm_fit]
    glm_fit: FWL: design includes the partial nuisance block; contrast loads only on the signal columns
    t_contrast: estimand 'dx'
    permutation_test: Freedman-Lane, tfce max-stat FWE (n=5000)
```

Multi-level run→subject→dataset (FLAME):

```
[ cope ~ cond {{ level=subject; combine=fixed }} ]
  >> [ . ~ 1 {{ level=dataset; combine=mixed; estimator=flame; contrasts: grp = 1 (t) }} ]
     {{ inference=permutation(cluster_mass, n=5000) }}

node stage0: [glm_fit]
    glm_fit: gaussian/identity GLM
node stage1: [flame_two_level]
    flame_two_level: mixed-effects group model over inbound copes/varcopes
    t_contrast: estimand 'grp'
edge: propagate {cope, varcope} stage0 -> stage1 (carry <default cope>, cope_varcope)
graph inference: permutation_test: Freedman-Lane, cluster_mass max-stat FWE (n=5000)
```

Forward-looking GAMM (`s(by=)` + a random effect → nitrix v3):

```
thk ~ s(age, k=6, by=dx) + dx + noise(meanFD) + (1 | site) {{ ... }}

node root: [gam_fit]
    gam_fit: penalised smooth bases + re/fs GAMM blocks  (nitrix v3)
```

## The no-regression invariant

`nwx` guarantees the IR; the engine guarantees the numerics **and** the
no-regression dispatch (nitrix v3 §0.1): adding a structure-dispatch ladder must
never change the result of a model that an earlier, simpler kernel already fit
exactly. The `shipped` flags above let a consumer see, statically, exactly which
formulae are runnable today versus pending a v3 kernel — the same information
`validate(graph)` rolls up as backend-awareness Diagnostics.

## Cross-references

- Spec: `docs/nwx/spec.md` §7 (device→IR→nitrix map + status), §9 (this contract).
- Validation: `gramform.grammars.nwx.validate` (the §8 static checks, incl. the
  backend-awareness roll-up and `Edge.carry` resolution).
- The executable contract: `tests/nwx/dryrun.py` + `test_engine_contract.py`.
- The reference engine (a real consumer, GLM slice): `examples/nwx_reference_engine/`.
- nitrix kernels: `nitrix/docs/feature-requests/stats-modelling-suite-v3.md`.
