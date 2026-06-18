# nwx — session resume prompt

> Hand this file to a fresh Claude Code session started in
> `/root/capsule/code/gramform` (branch `nwx`). It is a self-contained kickoff
> prompt: context, goal, current state, and the next task. Everything it
> references is on disk in this repo unless noted.

---

## Who you are / what this is

We are building an ecosystem for accelerated, differentiable neuroimaging in
JAX. Three libraries matter here:

- **`nitrix`** — the high-performance numerical substrate (XLA primitives;
  GLM/GAM/LME/permutation kernels). *Already mature.*
- **`gramform`** — a substrate for building small DSLs on a `ply` (lex/yacc)
  core. *This repo.* Mature enough to host real grammars.

**`nwx` (neuroimaging Wilkinson extension)** is the DSL we are building inside
`gramform`, extending the Wilkinson-formula proof-of-concept in
`src/gramform/grammars/wilkinson/`.

## The overall goal

`nwx` turns a **formula string** into a **validated, immutable `ModelSpec` IR**
(a *model graph*, not a design matrix) plus a **`CovariateProgram`**. It must
support **GL(A)MM, residualisation, and multi-level (node-graph) models** for
neuroimaging. It is a **pure specification layer**.

### Three-layer concern boundary (locked)
- **(A) Syntax** = `gramform.core` (the `ply` engine) — unchanged.
- **(B) `nwx`** = `string -> immutable ModelSpec IR + CovariateProgram`. **PURE.**
  Never imports `jax`, `nitrix`, arrays, or the mass axis. (There is a test that
  enforces this firewall.)
- **(C) Engine** = a *separate consumer* (lives outside `gramform`, e.g. in a
  reference engine / hypercoil) that lowers the `ModelSpec` onto `nitrix` calls
  and owns the mass-univariate vmap.

This mirrors BIDS-StatsModels (spec vs fitlins executor) and formulaic (Formula
vs ModelMatrix): **a model > a design matrix.** The interpreter emits a typed
`ModelSpec` IR, NOT `formulaic` objects.

## Locked decisions — do NOT relitigate

**Round 1 (scope):**
1. Scope = GL(A)MM + residualisation + multi-level node graph. The IR *carries*
   multivariate response (mvbind) as a future extension, not a v1 promise.
   Connectivity (cov/glasso/PCA) is **out of nwx scope.**
2. Model directives (family/link/estimator/error-structure/contrasts) live in a
   trailing `{{ key=val; ... }}` block.
3. Random effects use lme4 bar-in-parens: `(1+x|g)`, `(x||g)`, `(1|g1/g2)`,
   `(1|g1:g2)`. `|` inside parens = random effects; `|` at top level = parts
   separator. The `||` token is free to claim.
4. `nwx` **emits a `CovariateProgram` only**; the engine materialises columns.
   `nwx` stays array-free.

**Round 2 (after a four-lens review — architecture approved):**
1. Residualisation is **aggressive-by-default** (`~|` = full projection); a
   non-aggressive partial-regression (AROMA) mode is an explicit
   `{{residualise=nonaggressive}}` opt-in that **requires** a `signal()` set
   (hard error otherwise). FWL in-model partialling (`ModelSpec.partial`) is
   kept distinct from residualisation.
2. A **runnable vertical slice is Phase 2** via a **real reference engine that
   lives OUTSIDE gramform** (preserves the firewall).
3. The old `-ops` modules are **deleted with no shims** (done in Phase 0);
   downstream `-ops` consumers (`entense`, …) are being rebuilt — breakage
   accepted.
4. A **read-direction BIDS Stats Models importer** (model.json → ModelGraph)
   joins the roadmap (Phase 7).

## Current state (recovered after a crash)

> ⚠️ **Provenance.** ~19 h of work was lost in a system crash. This `nwx`
> branch was **reconstructed** from the session transcript + Claude Code
> file-history blobs. The source was restored from the post-`ruff` file-history
> blobs (the faithful crash-time bytes), and verified green. **Commit hashes
> differ** from any referenced in older notes/memory (the original `nwx` branch
> and its commits — e.g. `04881e1`, `0af810c`, `d1d4adc` — no longer exist).

**Branch `nwx`** (off `ply`), recovered commits:
- `(DOC) nwx: grammar specification + design` — `docs/nwx/spec.md`
- `(DOC) nwx: implementation plan (phases 0-7) + review-driven spec revision`
  — `docs/nwx/implementation-plan.md`
- `(MAINT) nwx Phase 0: remove pre-ply -ops modules; modernise dev suite`

**Phase 2 is DONE and verified green (2026-06-18) — M1 reached: nwx is
usable.** A real reference engine lives **outside `gramform`** at
`examples/nwx_reference_engine/` (preserving the firewall — it is never
imported by `gramform`, so the import-firewall subprocess stays clean even
though the engine imports `numpy`/`scipy`).
- nitrix is NOT importable here (needs `jax`; only a source checkout exists at
  `/root/capsule/code/nitrix`) and statsmodels is absent — so per the plan's
  allowance the engine uses a **numpy/scipy** path, pinned against independent
  linear-algebra oracles (`np.linalg.lstsq`, `scipy.stats.linregress`, a
  QR-based OLS, an FWL hand-derivation).
- `engine.run(formula, data, imaging, adjacency=…)` → `{contrast: Result}`:
  parses via the nwx processor, dispatches structurally on populated IR fields,
  and errors helpfully on unsupported IR (non-Gaussian, random/smooth/
  residualise, multi-node, etc.). `materialise.py` (design + treatment coding +
  partial), `glm.py` (vectorised OLS + t/F, F=t² verified), `inference.py`
  (Freedman–Lane permutation: voxel/cluster_extent/cluster_mass/TFCE max-stat
  FWE + FDR-BH/Bonferroni), `datasets.py`, `example.py`, `README.md`, `tests/`.
- **Pure spec-layer addition this phase:** `noise()`/`nuisance()` on a normal
  RHS now routes (structurally, via a `NwxState.partial` accumulator consumed by
  `LHS_RHS`) to `ModelSpec.partial` — `NAMED_FUNCTION_impl` no longer raises on
  those names (still raises on smooths/other calls → Phase 4). `signal()` and
  residualise-context `noise()` remain Phase 5.
- **Acceptance gate met:** `thk ~ dx + sex + noise(meanFD) {{ contrasts: dx=dx
  (t); inference=permutation(tfce, n=…) }}` runs formula→corrected map; the
  worked example recovers the true cluster (12/12 vertices, 0 false positives).
- **Verified:** gramform 138 passed/1 xfailed (firewall + conflict gates green),
  ruff+format clean, pyright 0 on nwx, coverage ≥68; engine 27 passed, ruff
  clean. Run the engine: `PYTHONPATH=src:examples python -m
  nwx_reference_engine.example` (and `... -m pytest examples/nwx_reference_engine
  /tests`). The engine is NOT in the nox gate (it is a separate consumer).

**Phase 1 is DONE and verified green (2026-06-18):**
- `grammars/nwx/spec.py` — the full §5 IR: enums (incl. reserved members), the
  closed `TermSource` union (`Lookup|Const|PyExpr|CovariateRef|Referent`,
  matched structurally), `FactorSpec`, `TermSpec` (role structural; `.order`
  computed; `INTERCEPT` constant), `SmoothSpec`, `GroupingSpec`,
  `RandomEffectSpec`, `ErrorSpec`, `ResidualiseSpec` (default
  `Mode.AGGRESSIVE`), `FamilySpec`, `ContrastSpec`, `EstimationSpec`,
  `InferenceSpec`, `ResponseSpec`, `ModelSpec` (+`partial`), `Carry`, `Edge`,
  `ModelNode`, `ModelGraph`, `Diagnostic`/`Severity`, `Lowerable` Protocol. All
  frozen, hashable, value-comparable; no mappings (tuple-of-pairs).
- `grammars/nwx/transform.py` — the `spec` interpreter over the **reused**
  `WilkinsonGrammar`: `NwxState` (single typed `eval` slot + `deps`/`directives`
  accumulators, **no `operational_level`**), the term-algebra ops
  (`VARIABLE/NUMERIC/EXECUTE/APPEND/REMOVE/UNARY_NEGATION/INTERACTION/NESTED`),
  structure (`LHS_RHS→ModelSpec`, `RESIDUAL→ResidualiseSpec`,
  `PUSH_FRAME→Referent` + frame sub-node, top-level `|` rejected),
  `to_term`/`to_terms` (ported dedup + structural-singularity + intercept),
  reusing `ppr_add_intercept`/`ppr_associative_flatten`/
  `ppr_common_subexpression`. `NwxProcessor` splits a trailing `{{…}}` block
  textually, parses it via `directives.py`, applies it, and emits a one-node
  (+ frame deps) `ModelGraph`.
- `grammars/nwx/directives.py` — minimal mini-parser (`family`/`link`/
  `estimator`/`se`/`dof`/`contrasts`/`inference`); unknown keys/values WARN.
- `grammars/nwx/covariate.py` — minimal `CovariateOp` (`Shorthand`/`Derivative`/
  `Power`) + shorthand table (`csf` is NOT a shorthand). **Not yet wired into
  the term interpreter** (covariate→term lowering is Phase 6).
- Tests `tests/nwx/`: golden frozen IR (`y~x+z`, `y~1`, `y~group`, `x*z`,
  `a/b`, frames, residualise, directives); PoC term-set parity via a
  `TermSpec`↔`formulaic.Term` bijection helper; intercept pinned in every
  position; singularity + restricted-`|` errors; covariate + directive units.
- **Verified: 136 passed / 1 xfailed; ruff check + format clean on
  `src/gramform`; pyright 0 errors on the nwx surface; parser-conflict gate = 0;
  coverage 73% (≥68 floor).**

**Phase-1 hard-won facts (don't rediscover):**
- The PoC's `to_terms` only applies the bare-constant / scale-singularity strip
  on the *dict* branch. Porting it: per-child coercion in `APPEND`/`remove_terms`
  must use `_coerce` (no strip) so a bare **zero survives** to drive intercept
  suppression / re-insertion; the singularity guard runs only on the
  *accumulated* set. (Two separate bugs hit this.)
- A Wilkinson term is a **set** of factors: `to_term` must dedup **and
  canonically sort** variable factors (`x:x→x`; `dog:rat==rat:dog`) — formulaic
  does NOT canonicalise factor order, so without this the `^2`-on-a-multi-term
  base over-counts.
- Parenthesised-zero parity (`x+(y+0)`) needs the PoC's documented oracle
  *mapping* (`↔ x+y+0`), not the same string fed to formulaic.
- `ppr_add_intercept` de-means the nuisance set: it adds the intercept to the
  **noise/RHS** side of a `~|`, so a residualise `noise` tuple leads with
  `INTERCEPT`.
- Don't import the `Test` enum into a test module under the name `Test*` —
  pytest tries to collect it (alias or use `.value`).

**Phase 0 is DONE and verified green:**
- Deleted pre-`ply` modules (`grammar.py` top-level, `dfops.py`, `imops.py`,
  `tagops.py` + their tests); added `grammars/` package `__init__`s.
- `core.py` captures PLY shift/reduce + reduce/reduce conflicts via
  `DynamicGrammar.conflicts` (PLY has **no** `parser.conflicts` attribute) — the
  parser-conflict gate uses this.
- Dev suite modernised to match siblings: `ruff format` (single-quote, line 79)
  replaced `blue`; Python 3.12–3.14; `requires-python>=3.12`; `pydantic` a core
  dep; `pyright` scoped to `src/gramform/grammars/nwx`; CI on feature branches;
  coverage floor ratcheted 90→68 (currently ~69%, climb as nwx lands).
- New gate tests in `tests/nwx/`: import firewall (no `jax`/`nitrix`) +
  parser-conflict freedom. One pre-existing `minimaltest` transform test is
  `xfail`ed (API drift, fixed in Phase 6).
- **Verified: 80 passed / 1 xfailed; `ruff check` + `ruff format --check` on
  `src/gramform` clean; coverage gate ≥68% PASS.**

## Dev environment (venv is on /scratch — keep root clean)

The root partition is tiny (~3 GB). **Install nothing on root.** The venv lives
at `/scratch/gramform-venv` (CPython 3.12). Run tools by absolute path; run
tests with `PYTHONPATH=src` (the package is *not* installed editable, to avoid
writing build artifacts to root):

```bash
VENV=/scratch/gramform-venv/bin
export XDG_CACHE_HOME=/scratch/xdg-cache TMPDIR=/scratch/tmp UV_CACHE_DIR=/scratch/uv-cache

# tests + coverage
PYTHONPATH=src $VENV/python -m pytest tests/ -q --cov=gramform --cov-report=
$VENV/coverage report --omit='*test*,*__init__*'

# lint/format gates (the noxfile gate scope is src/gramform ONLY)
$VENV/ruff check src/gramform
$VENV/ruff format --check src/gramform

# typecheck the nwx surface (pyright is installed in the venv).
# NOTE: pyright's bundled Node needs libatomic.so.1, which isn't on the system
# default path — export this first (a copy lives on /scratch; root is too small
# to add system packages):
export LD_LIBRARY_PATH=/scratch/nperf/renv/lib:$LD_LIBRARY_PATH
$VENV/pyright src/gramform/grammars/nwx   # verified: 0 errors (docstring stub today)
```

If the venv ever breaks, rebuild it **only on /scratch**:
`uv venv --clear --python 3.12 /scratch/gramform-venv` then
`uv pip install --python /scratch/gramform-venv/bin/python ply wadler-lindig pydantic formulaic narwhals numpy pandas pytest pytest-cov "coverage[toml]" ruff pyright`
(set `UV_CACHE_DIR`/`UV_PYTHON_INSTALL_DIR`/`TMPDIR` to `/scratch/...` first).

## YOUR NEXT TASK — Phase 3: random effects (bar-in-parens)

Goal: parse the lme4 bar-in-parens idiom and emit `RandomEffectSpec`. **M**
sized. This is the **first phase that EXTENDS the grammar** (a new token + a new
`GrammarComponent`), so the **parser-conflict gate becomes load-bearing** —
PLY conflicts are a global property of the merged grammar; assert
`grammar.conflicts == ()` (the `DynamicGrammar.conflicts` capture from Phase 0).

**Files (see `implementation-plan.md` Phase 3 + spec §4.2):**
- **`grammars/nwx/grammar.py`** (new) — `RanefComponent`: token
  `RANEF_UNCORR (||)` (function-free string token, so PLY orders it before `|`
  by descending regex length — assert with a lexer test); productions
  `factor : LPAREN ranef RPAREN`, `ranef : expression (PARTS_SEPARATOR |
  RANEF_UNCORR) grouping`. New primitives `RANDOM_EFFECT`, `GROUPING`. **This
  needs a new `NwxGrammar`** = WilkinsonGrammar components + `RanefComponent`
  (the Phase-1/2 transform currently parses with the bare `WilkinsonGrammar`,
  which CANNOT parse `(1|g)` — confirmed: it errors `Unexpected '|'`). Wire the
  new grammar into `NwxProcessor`.
- **`grammars/nwx/transform_ranef.py`** (new, disjoint file) — interpreter ops
  → `RandomEffectSpec(structure ∈ {SCALAR, DIAGONAL, UNSTRUCTURED})`. Rules
  (spec §4.2 table): `(1|g)`→scalar; `(1+x|g)`/`(x|g)`→unstructured;
  `(0+x|g)`/`(x-1|g)`→slope-only; `(1+x||g)`→diagonal; `(1|g1/g2)`→**two**
  specs `(1|g1)+(1|g1:g2)`; `(1|g1:g2)`→`GroupingSpec(relation=INTERACTION)`,
  one component (NOT crossed); genuine crossing `(1|g1)+(1|g2)`→**multiple**
  specs. Intercept present unless `0`/`-1` on the bar LHS. Random slope without
  its fixed effect is legal (informational, not a warning).
- Backend-awareness WARNING for non-scalar structures (nitrix v3 §1.1 R2–R4).

**Gate (R1 in the risk register):** the §4.2 analysis says bar-in-parens is
conflict-free (the parts-bar lives at `blocks` level, unreachable from inside
`LPAREN expression RPAREN`); **assert 0 conflicts** on the merged grammar. If a
conflict appears, the documented fallback is a functional `re(1+x, g)` form.

**Tests:** each surface → expected `RandomEffectSpec`; nesting/crossing; the
conflict gate on the new grammar; `structure` correctness; **`y ~ s` (bare `s`
lookup) vs `y ~ (…|g)` grouping** and bare `g` lookup vs `(…|g)` — reserved
names are special only in call/grouping position.

> Phases 3 / 4 (smooths + full directives + exclusive `{{}}` lexer state) / 6
> (covariate→term lowering) are parallelisable on disjoint `transform_*` /
> `covariate` files; the conflict gate runs on each integration merge. Phase 5
> (residualise modes + multi-level `>>`) depends on 4. Phase 7 (validate.py +
> BIDS-SM importer) is last. The reference engine extends in lockstep as kernels
> are surfaced (random effects → `lme_fit` structure-dispatch, nitrix v3 §1.1).

## Roadmap (phases 1–7) & milestones

```
Phase 0 ✅ ─▶ 1 (IR+spec) ─▶ 2 (runnable slice) ─┬─▶ 3 (random effects) ─┐
                                                 ├─▶ 4 (smooths+directives)┼─▶ 5 ─▶ 7
                                                 └─▶ 6 (covariate prog) ───┘
```
- **M1** (end P2): nwx usable, formula→corrected stat-map via the external
  reference engine. **M2** (end P4): GAM/GLMM formulae + full directives.
  **M3** (end P5): residualisation + multi-level (the distinctive surface).
  **M4** (end P7): validated, contract-proven, BIDS-SM importable.
- After P2, phases 3/4/6 are parallelisable (disjoint files); the
  parser-conflict gate runs on **each integration merge** (conflicts are a
  global property of the merged grammar). 5 depends on 4 + the residualise
  primitive. 7 is last.

Full detail: **`docs/nwx/implementation-plan.md`** (phase-by-phase, file-by-file,
risk register). Design rationale + grammar resolutions: **`docs/nwx/spec.md`**.

## Gotchas / hard-won facts (don't rediscover these)

- **PLY has no `parser.conflicts`.** Use the `errorlog`-capture hook /
  `DynamicGrammar.conflicts` already wired in `core.py`.
- The `{{...}}` directive block must use an **EXCLUSIVE** lexer state for `nwx`
  (the `minimaltest` push/pop pattern is *inclusive* — copy the idea, make it
  exclusive so `:`/`=` don't collide with default-state tokens).
- **Harvest the confound vocabulary from `grammars/minimaltest/`**, NOT the
  deleted legacy `dfops.py`. The harvested rules are in
  `docs/nwx/covariate-vocabulary.md`. **`csf` is NOT a shorthand** (passthrough).
- **Wilkinson precedence wins on shared glyphs** (`^`, `-`, `||`) — see spec §6.
- **`ruff format` uses single-quote, line-length 79.** The lint/format gate is
  scoped to `src/gramform` only (test files are not gated).
- **Web access (WebFetch/WebSearch) is BLOCKED** in this env — BIDS-SM and
  ModelArray exact field spellings were never verbatim-verified; treat them as
  approximate until checked.
- The nitrix engine target is the `lme_fit` structure-dispatch ladder in
  `nitrix/docs/feature-requests/stats-modelling-suite-v3.md` (recovered;
  the nitrix agent added a §0.1 no-regression dispatch invariant + R0–R4 ladder
  — **do not revert that FR**).

## Key references
- `docs/nwx/spec.md` — grammar + IR + engine-contract design.
- `docs/nwx/implementation-plan.md` — phases 0–7, file-by-file.
- `docs/nwx/covariate-vocabulary.md` — Phase-6 confound vocabulary (harvested).
- `src/gramform/core.py`, `grammars/wilkinson/`, `grammars/minimaltest/`,
  `postprocessors.py` — the substrate to build on.
- `nitrix/docs/feature-requests/stats-modelling-suite-v3.md` — the engine-side
  kernel gaps nwx surfaces.
- Auto-memory: `nwx-dsl-design.md`, `dev-environment.md`.
