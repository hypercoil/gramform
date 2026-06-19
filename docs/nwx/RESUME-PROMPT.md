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

**Phase 5 is DONE and verified green (2026-06-18) — M3 reached: the
distinctive residualisation + multi-level surface.** Three sub-commits:
- **5a — residualisation modes** (`5619fb7`). `~|` is aggressive by default;
  `{{ residualise=nonaggressive }}` flips the mode and **requires** a `signal()`
  set (hard error). `signal()`/`noise()` are call-position role markers: on a
  `~|` RHS (a new `NwxState.in_residualise` flag set by
  `RESIDUAL_STRUCTURE_impl`) they route into the residualise noise/signal sets
  (unwrapped terms join noise); `noise()` on a *normal* RHS still routes to
  `ModelSpec.partial` (FWL) — distinct. `signal()` off a `~|` RHS is a hard
  error. `_validate_residualise` (root + frame sub-nodes) enforces §4.6/§8.
- **5b — grammar-integrated `{{ }}`** (`0440263`). Retired the Phase-1 textual
  split; a `DirectiveComponent` captures `{{ ... }}` as one `DIRECTIVE_BLOCK`
  function-token (out-ranks `EXECUTE` `{...}`); productions
  `program : pipeline [DIRECTIVE_BLOCK]` (outermost node) and
  `factor : LBRACKET formula DIRECTIVE_BLOCK RBRACKET` (frame node). The block
  is opaque to the term lexer, so the directive `:`/`=` never collide with the
  term algebra's — the isolation an exclusive lexer state was meant to give,
  with a single blob token (no state needed). An nwx-aware intercept
  postprocessor (`nwx_add_intercept`) descends past the directive wrappers.
- **5c — multi-level `>>`** (`4f9abb6`). A `PipelineComponent` adds the
  `STAGE_PIPE (>>)` token + `pipeline` productions (associative `PIPELINE`).
  `PIPELINE_impl` builds one `ModelNode` per stage (a `[...]` stage is a node,
  NOT an inline `_hat` referent — the R5 stage/frame position split) with an
  `Edge` between consecutive stages carrying the upstream cope/varcope. A `.`
  on a downstream stage LHS resolves to the prior stage's cope
  (`Referent(stage, 'cope')`) via `NwxState.inbound_stage`; outside a pipeline
  `.` stays the complement. A trailing block after the whole pipeline is
  graph-level (`ModelGraph.inference`). The §11 run→subject→dataset example
  parses to the right node/edge/referent graph.
- **Verified: 246 passed / 1 xfailed; ruff + format clean; pyright 0 on nwx;
  conflict gate = 0 (incl. NwxGrammar with all six components); cov 76%;
  engine 27 passed.**

**Phase-5 hard-won facts:**
- A blob `DIRECTIVE_BLOCK` token (function token, so it out-ranks the
  string-defined `EXECUTE`) gives positional directive attachment without an
  exclusive lexer state — the directive `:`/`=` never reach the term lexer.
- The wilkinson `ppr_add_intercept` is unaware of the nwx top nodes
  (`PROGRAM_DIRECTIVES`/`FRAME_DIRECTIVES`/`PIPELINE`); use the nwx
  `nwx_add_intercept` which descends past the opaque string operands. A pipeline
  *frame* stage gets its intercept INSIDE the frame (so `PIPELINE_impl` can
  still unwrap the `PUSH_FRAME`); a bare stage gets it on the formula.
- `.` disambiguation is by parse position: set `inbound_stage` before
  evaluating each downstream stage; `VARIABLE_COMPLEMENT_impl` reads it.
- `Edge.carry.contrast` binds the upstream's first contrast name (or `''`); the
  precise FLAME varcope binding is engine/Phase-7 territory.

**Phase 4 is MOSTLY DONE and verified green (2026-06-18) — M2 reached
functionally (GAM/GAMM formulae + full directives).** Three sub-commits:
- **4a — GAM/GAMM smooths** (`grammars/nwx/transform_smooth.py`, new). `s`/`te`/
  `ti`/`t2` (special only in call position, R6) lower onto `SmoothSpec`. They
  reuse the existing parameterised `NAME(...)` productions — **no grammar
  change** — via a new `NAMED_FUNCTION_HANDLERS` dispatch table in
  `transform.py` that `transform_smooth` registers into. Smooths route out of
  `fixed` into a new `NwxState.smooth` accumulator (same pattern as
  `partial`/`random`) → `ModelSpec.smooth`. `bs=` → `BasisKind` (+cyclic for
  `cc`/`cp`); default basis `tprs`(s)/`tensor`(te/ti/t2); k (default 10), m
  (penalty_order), by, fx. `bs="re"/"fs"` GAMM bridge: 2nd positional arg →
  `by`. `by_kind` is data-dependent → left `None` (engine resolves).
  `BackendWarning` for non-shipped bases.
- **4b — full directive set + error structures** (`directives.py`). Adds
  `correlation=ar1(idx|grp)`→`ErrorSpec`, `weights=varIdent/varPower(arg)`→
  heteroscedasticity, `se=robust(hc3)`/`se=cluster(by)`, and node-level
  `level`/`group_by`/`combine` (→ the root `ModelNode`). `BackendWarning`s for
  v3-gated kernels (non-core families/links, robust/cluster se, satterthwaite/
  kr dof, correlation, weights). `BackendWarning` moved to `spec.py` (shared,
  re-exported from `transform_ranef`) to avoid an import cycle.
- **Verified: 210 passed / 1 xfailed; ruff + format clean; pyright 0 on nwx;
  conflict gate = 0; coverage 75%; engine 27 passed.**

**Phase-4 hard-won facts:**
- Feature-family call handlers (smooths) register into a shared
  `NAMED_FUNCTION_HANDLERS` dict in `transform.py`; `NAMED_FUNCTION_impl`
  consults it before raising. Keeps `transform.py` from importing the
  feature-family modules (they bottom-import-register, like the ranef ops).
- The smooth-call AST is `(name, first_expr, [params_node], OperationalLevel)`
  — drop the name and the trailing level marker (`parameters[1:-1]`); the
  optional params node is `FUNCTION_PARAMETERS`/`PARAMETER`/`NAMED_PARAMETER`.
  String-literal param values arrive as bare `core.Literal` (quotes included →
  strip); numeric values as `NUMERIC_LITERAL` primitives.
- `by_kind` (factor vs continuous) genuinely cannot be set at parse time (nwx
  is data-free) — leave `None`; the engine/validator resolves it (the §8
  "factor by= without main effect" warning is a Phase-7 validate.py check).

**STILL TODO in Phase 4 — the exclusive-state `{{ }}` lexer grammar.** The
directive *content* is fully parsed (above), but via the **Phase-1 textual
trailing-block split** (`_split_directives`/`_DIRECTIVE_RE` in `transform.py`),
NOT yet the integrated **exclusive `spec` lexer state** the plan/§4.5 call for.
That rework (a `DirectiveComponent` with `('spec','exclusive')`, real
productions, bracket-scoped placement: node-level inside `[]`, graph-level after
`>>`) is **deferred to land with Phase 5**, where its scoping payoff (frame /
multi-level directive attachment) is actually exercised — the textual split
already delivers full single-node directives, so the rework is low-value until
`>>` exists. The conflict gate stays load-bearing when it lands.

**Phase 3 is DONE and verified green (2026-06-18) — first grammar EXTENSION.**
Random effects (lme4 bar-in-parens) now parse and emit `RandomEffectSpec`.
- `grammars/nwx/grammar.py` (NEW) — `RanefComponent` adds the token
  `RANEF_UNCORR (||)` and productions `factor : LPAREN ranef RPAREN`,
  `ranef : expression (PARTS_SEPARATOR|RANEF_UNCORR) grouping`,
  `grouping : term`; new AST primitives `RANDOM_EFFECT`/`GROUPING`.
  **`NwxGrammar`** = the five Wilkinson components + `RanefComponent`.
  `NwxProcessor` now parses with `NwxGrammar` (was the bare `WilkinsonGrammar`).
- `grammars/nwx/transform_ranef.py` (NEW, disjoint file, registers into the
  shared `spec` group via a bottom-import in `transform.py`) — `RANDOM_EFFECT`
  → `RandomEffectSpec`. Routed structurally OUT of `fixed` into a new
  `NwxState.random` accumulator (mirrors the Phase-2 `partial`/`noise()`
  mechanism), consumed by `LHS_RHS` into `ModelSpec.random` (works in frames
  too). Implicit lme4 intercept reuses the model's own
  `add_intercept_to_formula` so `0+x`/`x-1` suppress it identically.
  `_grouping_components` does lme4 nesting (`g1/g2` → `g1`, `g1:g2`);
  `g1:g2` → ONE interaction grouping factor (`Relation.INTERACTION`), NOT
  crossing; crossing = multiple bars → multiple specs. `structure`: 1 term →
  `SCALAR`; ≥2 terms → `UNSTRUCTURED` (`|`) / `DIAGONAL` (`||`).
  `BackendWarning` (a `UserWarning` subclass) fires for non-scalar structures
  (v3 §1.1 R2) and nested groupings (R3); scalar single/interaction effects
  (shipped `reml_fit` R1) and slope-without-fixed-effect are silent.
- **Conflict gate (R1): `NwxGrammar().conflicts == ()`** — the parts-bar lives
  at `blocks` level, unreachable from inside `LPAREN expression RPAREN`, so no
  reduce/reduce (spec §4.2). Added to `test_parser_conflicts.py`. R7: `||`
  out-ranks `|` by longest match (both function-free string tokens) — pinned by
  a lexer test; `~|` still lexes correctly.
- **Verified: 167 passed / 1 xfailed; ruff + format clean; pyright 0 on nwx;
  conflict gate = 0 (incl. NwxGrammar); coverage 74%.** Engine still 27 passed,
  M1 example intact; a `(1|g)` formula now parses and the Phase-2 engine
  rejects it with a helpful `EngineError` (the `random` field is populated).

**Phase-3 hard-won facts (don't rediscover):**
- A random effect must be routed OUT of the fixed-term stream (like `noise()`),
  not returned as a term — `RANDOM_EFFECT_impl` appends to `state.random` and
  returns `()`. `LHS_RHS`/`finalise` consume + clear it so it binds to its block.
- lme4's implicit random intercept can't be inferred from the post-eval term
  list (`(x|g)` and `(0+x|g)` both reduce to `[x]`). Reuse
  `add_intercept_to_formula` on the bar-LHS AST *before* evaluating, so the
  existing ZERO/`-1` suppression machinery cancels it for `0+x`/`x-1`.
- Interpret the grouping from the RAW term AST (under the `GROUPING` wrapper) —
  do NOT dispatch it through the normal term interpreter (that would expand
  `g1/g2` into design terms `g1 + g1:g2`). `GROUPING_impl` is a defensive raise.
- The bottom-import in `transform.py`
  (`import gramform.grammars.nwx.transform_ranef`) is what registers the ranef
  ops into the shared `INTERPRETERS`; keep it a single-line module import with
  `# noqa: E402, F401` (ruff reflows a multi-line `from ... import` and breaks
  the noqa placement).

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
  coverage floor ratcheted 90→68 in Phase 0 (later raised to 78 once the nwx
  surface landed; the legacy `grammars/dataframe` materialiser at ~31% is the
  drag — give it tests or retire it to climb toward 90).
- New gate tests in `tests/nwx/`: import firewall (no `jax`/`nitrix`) +
  parser-conflict freedom. One pre-existing `dataframe` transform test is
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

## YOUR NEXT TASK — Phase 6 (covariate program) then Phase 7 (validate + BIDS)

Phases 1–5 are DONE (M1–M3). Two phases remain:

**Phase 6 — `CovariateProgram` completeness** (`implementation-plan.md` Phase 6;
disjoint `covariate.py`, independent of the grammar). Extend the minimal Phase-1
`CovariateOp` set to the full closed union harvested from `grammars/dataframe/`
(`Shorthand`, `Derivative`, `Power`, `CompCorSelect` via `{{…}}`, `Indicator`,
`SetOp`, `Scatter`); wire covariate shorthands into the term interpreter (they
are defined but NOT yet lowered into terms — `csf` is passthrough, not a
shorthand). Shorthand expansions as a preprocessor (the
`confound_formula_preprocessor` pattern). **Emit-only** (nwx holds no array).
Tests: 36P `(dd_(rps+wm+csf+gsr))^^2`, spike `:::`/`OR_`/`I_`, aCompCor `v_`
→ expected `CovariateProgram` + term set. Fixes the one `xfail`ed `dataframe`
transform test (API drift).

**Phase 7 — validation, errors, contract, BIDS-SM importer** (last).
`validate(graph) -> tuple[Diagnostic, ...]` per §8: rank/identifiability,
intercept, RE well-formedness, **smooth factor-`by=` without its main effect**
(deferred from P4), non-aggressive-needs-signal (already enforced at parse —
move/duplicate as a Diagnostic), reserved-name shadowing, **multi-level DAG +
`Edge.carry.contrast` resolves to a real upstream `ContrastSpec`** (the carry
binding `PIPELINE_impl` left as a default), `.` disambiguation, backend-awareness
roll-up. Plus the engine-contract doc + a dry-run dispatcher under `tests/nwx/`
and the read-direction BIDS Stats Models importer (`model.json` → `ModelGraph`).

> The reference engine extends in lockstep as nitrix kernels are surfaced
> (smooths → `gam_fit`; random effects → `lme_fit`; non-aggressive →
> `partial_residualise`; FLAME two-level chaining for the `>>` graph). Today the
> Phase-2 engine rejects populated `random`/`smooth`/`residualise`/multi-node IR
> with a helpful `EngineError`.

## Roadmap (phases 1–7) & milestones

```
Phase 0 ✅ ─▶ 1 ✅ ─▶ 2 ✅ (runnable slice) ─┬─▶ 3 ✅ (random effects) ─┐
                                            ├─▶ 4 ✅ (smooths+directives)┼▶ 5 ✅ ▶ 7
                                            └─▶ 6 (covariate prog) ──────┘
  (5 ✅ = 5a residualise modes + 5b grammar-integrated {{}} + 5c multi-level >>.
   The exclusive-state {{}} lexer was realised as a blob token in 5b.
   Remaining: 6 (covariate program), 7 (validate + BIDS-SM importer).)
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
  (the `dataframe` push/pop pattern is *inclusive* — copy the idea, make it
  exclusive so `:`/`=` don't collide with default-state tokens).
- **Harvest the confound vocabulary from `grammars/dataframe/`**, NOT the
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
- `src/gramform/core.py`, `grammars/wilkinson/`, `grammars/dataframe/`,
  `postprocessors.py` — the substrate to build on.
- `nitrix/docs/feature-requests/stats-modelling-suite-v3.md` — the engine-side
  kernel gaps nwx surfaces.
- Auto-memory: `nwx-dsl-design.md`, `dev-environment.md`.
