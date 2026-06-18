# `nwx` — implementation plan (phases 0–7)

> **Status (2026-06-17): proposed; revised after the four-lens review.**
> Companion to `docs/nwx/spec.md`. Concrete, file-by-file build plan. Effort
> tags: **S** ≈ ≤1 day, **M** ≈ 2–4 days, **L** ≈ ≥1 week.
>
> **Changes from the first draft (review + decisions):** (a) a **runnable
> vertical slice** is sequenced as Phase 2 (formula→results for the Gaussian
> GLM-with-confounds path, on already-shipped nitrix kernels, via a real
> reference engine that lives *outside* `gramform`); (b) the confound vocabulary
> is harvested from **`grammars/minimaltest/`** (the new-core port), not the
> legacy `dfops.py`; (c) the parser-conflict gate is implemented via a captured
> `errorlog` (PLY has no `parser.conflicts`); (d) CI/nox discipline is made to
> actually bind; (e) the AST→IR interpreter is split by feature family; (f)
> deletion of the old `-ops` modules proceeds **without shims** — downstream
> consumers (`entense`, …) that use the old `-ops` system are being rebuilt and
> breakage is accepted; (g) residualisation is aggressive-by-default.

## Conventions & cross-cutting discipline

- **Style.** `@dataclass(frozen=True)` for all IR; `tuple` / `tuple[tuple[K,V],
  ...]` for collections (**no `frozendict`, no mappings in IR fields**); closed
  `Union`s + `Literal`/`Enum` for `nwx`-owned sets; `Protocol` only for the
  engine-facing `Lowerable`. No `Any` in public signatures.
- **Tooling that actually gates (fixed in Phase 0).** nox `tests` session runs
  `pytest --cov` (≥ 90 % absolute, the existing floor), `ruff check`, `blue
  --check`, **and `pyright`** (newly wired). `.github/workflows/ci.yml` is
  updated to trigger on the feature branches (currently `main`-only, so no PR
  to `nwx`/`ply` is gated today). A test asserts `import gramform.grammars.nwx`
  transitively imports neither `jax` nor `nitrix`.
- **Parser-conflict gate (fixed).** PLY exposes no `parser.conflicts`. A
  `parser_conflicts()` test helper threads a capturing `logging.Handler` as
  `errorlog` into `yacc.yacc(...)` (a new optional `errorlog=` hook on
  `DynamicGrammar`) and asserts **zero** shift/reduce + reduce/reduce records
  (or parses the generated `parser.out`). The gate runs **per-phase and on the
  integration branch after each merge** — conflicts are a global property of the
  merged grammar.
- **Golden tests by value.** Frozen IR ⇒ `assert processor(formula) ==
  ModelGraph(...)` against hand-written expected IR. No fitting, no data
  (except Phase 2's reference engine).
- **PR / gate model.** One PR per phase; gate = pytest + ruff + blue + pyright +
  0 conflicts + coverage ≥ 90 %.

---

## Phase 0 — Cleanup, substrate adoption, gating infra  *(S–M)*

**Goal.** Remove the pre-`ply` modules; adopt `minimaltest` as the confound-
vocabulary base; make the firewall and the gates real.

**Tasks.**

1. **Delete the old `-ops` stack** (breakage accepted): `src/gramform/grammar.py`
   (top-level), `imops.py`, `tagops.py`, `dfops.py`, the empty `grammar/`
   package, and the matching tests (`test_imops.py`, `test_tagops.py`,
   `test_dfops.py`). Verify `test_grammar.py` targets the new `DynamicGrammar`
   (keep) vs the old `Grammar` (delete). **No compat shims** — downstream
   consumers using the old system (`entense.{instance,align,architecture}`
   import `ConfoundFormulaGrammar`/`DataTagGrammar`; these top-level exports are
   already commented out in `__init__.py`) are being rebuilt.
2. **Keep** `resampler.py`, `error.py` (core depends on them) and
   `grammars/minimaltest/` (the confound-vocabulary base for Phase 6).
3. **Generate** `docs/nwx/covariate-vocabulary.md` *from* `minimaltest`'s
   surfaces (`confound_formula_preprocessor`, `d_/dd_`, `^^`, `v_`, `I_`,
   `AND_/OR_/NOT_`, `:::`) — the Phase-6 source of truth. Note `csf` is **not** a
   shorthand (passthrough).
4. **Clean `__init__.py`**; **prune `pyproject.toml`** `[imops]`/`[dfops]`
   extras and the nox `jax[cpu]` install (gramform is jax-free post-deletion);
   add `grammars/__init__.py` (+ `wilkinson/__init__.py`) for consistent
   packaging.
5. **Wire the gates**: pyright into nox; CI on feature branches; the
   `errorlog` hook on `DynamicGrammar`; the no-jax/no-nitrix import test.

**Acceptance gate.** `nox` green; `import gramform.grammars.nwx` jax/nitrix-free
(test); CI runs on the branch; `parser_conflicts()` helper works.

---

## Phase 1 — `ModelSpec` IR + `spec` interpreter (the slice's surface)  *(L)*

**Goal.** Emit a validated `ModelGraph` for the term algebra + `~` + the Gaussian
GLM `ModelSpec` — **plus the minimal directives and covariate ops the runnable
slice needs** (so Phase 2 has a complete surface to execute).

**Files.** `grammars/nwx/{__init__.py, spec.py, transform.py, directives.py,
covariate.py}`.

- **`spec.py`** — the full IR of spec §5: enums, the closed `TermSource` union
  (`Lookup|Const|PyExpr|CovariateRef|Referent`, matched not `kind`-tagged),
  `FactorSpec`, `TermSpec` (role **structural**, not a field; `.order` computed),
  `SmoothSpec`, `GroupingSpec`, `RandomEffectSpec`, `ErrorSpec`,
  `ResidualiseSpec` (default `Mode.AGGRESSIVE`), `FamilySpec`, `ContrastSpec`
  (weights = tuple-of-pairs), `EstimationSpec` (`estimator`/`se`/`dof`),
  `InferenceSpec` (+`cluster_threshold`), `ResponseSpec` (aux tuple-of-pairs),
  `ModelSpec` (+`partial`), `Carry`, `Edge`, `ModelNode` (+`combine`),
  `ModelGraph` (+graph `inference`), `Diagnostic`/`Severity`, the `Lowerable`
  Protocol. All frozen, hashable; a value-equality test per node.
- **`transform.py`** — `NwxState`/`NwxContext` (typed result slots; prefer a
  typed result union over a wide `operational_level` match) and the core
  term-algebra ops (`VARIABLE→Lookup`, `INTERACTION→TermSpec`, `APPEND→union`,
  `REMOVE→difference`, `NESTED`, `LHS_RHS→ModelSpec`, `PUSH_FRAME→Referent`),
  porting the PoC `to_terms`/dedup/intercept semantics (with its tests) onto
  `TermSpec`. Top-level `|` restricted to single-part in v1 (reserved forms
  gated). Reuse `ppr_add_intercept`/`ppr_associative_flatten`/
  `ppr_common_subexpression`.
- **`directives.py` (minimal)** — `family`, `estimator`, `contrasts`,
  `inference` only (enough for the slice); full directive set in Phase 4.
- **`covariate.py` (minimal)** — `Shorthand` + `Derivative` + `Power` only
  (the 36P-style confound subset the slice exercises); full vocabulary Phase 6.

**Tests.** Golden IR for `y~x+z`, `y~1`, `y~group`, frames; PoC term-set parity
(hand-written `TermSpec` expectations — the formulaic oracle is *not* IR-
comparable, so a `TermSpec`↔`Term` bijection is provided as a test helper, not
assumed); intercept pinned in every position.

**Acceptance gate.** Canonical formulae → expected frozen IR; intercept/dedup
parity with the PoC.

---

## Phase 2 — Runnable vertical slice (formula → results)  *(M–L)*

**Goal.** Prove the contract end-to-end on the single most-used model:
**Gaussian mass-univariate GLM + confounds + t/F contrast + permutation/FDR
inference**, all on already-✅ nitrix kernels.

**Where it lives.** A **reference engine outside `gramform`** (preserving the
firewall) — `examples/nwx_reference_engine/` (or a sibling `nwx-engine`
package) that imports the IR from `gramform` and the numerics from `nitrix`
(or, if nitrix is unavailable in an env, a `numpy`/`statsmodels` fallback path).
It is a *real* runner, not the Phase-7 dry-run dispatcher.

**Tasks.**

1. Engine consumes a one-node `ModelGraph`, materialises the `CovariateProgram`
   (Phase-1 subset) against a covariate dataframe, assembles `X`, and binds the
   imaging array's mass axis.
2. Dispatch: `fixed`-only Gaussian → `glm_fit`; `partial` present → residualise
   then fit (FWL); `ResidualiseSpec(AGGRESSIVE)` → `linalg.residualise`.
3. Contrasts → `t_contrast`/`f_contrast`; `inference=permutation(tfce|cluster|
   voxel)` → `permutation_test`; `correction=fdr/bonferroni` → `fdr_bh`/
   `bonferroni`.
4. A worked end-to-end example (a small vertexwise dataset) producing a stat map
   + corrected p-map from a formula string.

**Tests.** Engine round-trips the slice formulae to numeric results; the
reference engine's outputs match a direct `nitrix`/`statsmodels` call on the same
design (oracle). The contract (IR fields consumed) is asserted.

**Acceptance gate.** `thk ~ dx + sex + noise(meanFD) {{ contrasts: dx=dx (t);
inference=permutation(tfce, n=…) }}` runs formula→corrected-map. **Milestone
M1: nwx is usable.**

---

## Phase 3 — Random effects (bar-in-parens)  *(M)*

**File.** `grammar.py` (`RanefComponent`), `transform_ranef.py`.

- Token `RANEF_UNCORR (||)`; productions `factor : LPAREN ranef RPAREN`,
  `ranef : expression (PARTS_SEPARATOR|RANEF_UNCORR) grouping`. New primitives
  `RANDOM_EFFECT`, `GROUPING`.
- Interpreter → `RandomEffectSpec(structure ∈ {SCALAR,DIAGONAL,UNSTRUCTURED})`;
  `(1|g1/g2)` expands to two specs; `(1|g1:g2)` → `GroupingSpec(INTERACTION)`;
  crossing → multiple specs. Intercept rule: present unless `0`/`-1` on the bar
  LHS.
- **Gate**: `parser_conflicts()` = 0 (per §4.2 analysis); fallback `re(1+x,g)`.
- Backend-awareness warnings for non-scalar structures (nitrix v3 §1.1 R2–R4).

**Tests.** Each surface → expected spec; nesting/crossing; conflict gate;
`structure` correctness; bare `g` lookup vs `(…|g)` grouping.

---

## Phase 4 — Smooths + full directive block + error structures  *(M–L)*

**Files.** `transform_smooth.py`, `directives.py` (full), `grammar.py`
(`DirectiveComponent`).

- Smooths: `SMOOTH_CONSTRUCTORS={s,te,ti,t2}` (special only in call position) →
  `SmoothSpec` (k, penalty_order, by + `by_kind`, cyclic, tensor, fx, bounds);
  `bs` string → `BasisKind`; `bs="re"/"fs"` slope var → `by`. Plain `bs`/`ns`/
  `poly` stay PyExpr data-transforms (not smooths).
- **`DirectiveComponent` — EXCLUSIVE `spec` state** (`('spec','exclusive')`),
  defining its own `;`/`=`/`:`/`,`/`(`/`)`/name tokens so they do not collide
  with default-state `:`/`=`. Productions per the §4.5 scoping rule (node/frame
  arm here; the graph arm's `node_seq` is a placeholder top rule replaced in
  Phase 5). Contrast clause reuses the term-combinator over coefficient names
  (no interaction `:` inside the state).
- Full directive set → `FamilySpec`/`EstimationSpec`/`ErrorSpec`/`InferenceSpec`/
  `ContrastSpec`; `correlation=ar1(time|g)` → `ErrorSpec`. Backend-awareness
  warnings for v3 kernels (families beyond 3, error structures, dof, robust se).

**Tests.** Smooth param mapping incl. `by_kind`, cyclic `bounds`, `bs="re"`
slope→by; directive scoping matrix; exclusive-state isolation of `:`/`=`;
longest-match lexer tests; `s(age,by=dx)` without `dx` → warning.

---

## Phase 5 — Residualisation modes + multi-level graph  *(M–L)*

**File.** `transform_struct.py`, `grammar.py` (`PipelineComponent`).

- Residualisation: `~|` → `ResidualiseSpec(mode=AGGRESSIVE)` by default;
  `{{ residualise=nonaggressive }}` flips mode and **requires** a `signal()` set
  (else hard error). `signal()/noise()/nuisance()` (call-position) route terms
  **structurally** into `ResidualiseSpec.signal/.noise`; `noise()` on a *normal*
  RHS routes into `ModelSpec.partial`.
- Multi-level: token `STAGE_PIPE (>>)`; `node_seq` replaces the Phase-4
  placeholder top rule; build `ModelGraph` with `ModelNode`s (level/group_by/
  combine from directives) and `Edge`s. `Edge.carry` binds an upstream
  `ContrastSpec` name + `{cope,varcope}`. `.`-on-stage-LHS = inbound cope
  (disambiguated from complement by position).

**Tests.** Aggressive vs non-aggressive (+ hard-error on empty signal); role
routing (signal/noise/partial); FWL-vs-residualise kept distinct; `>>` graph +
edges + carry binding; node vs graph directive attachment; FE/ME combine; `.`
both meanings.

---

## Phase 6 — `CovariateProgram` completeness  *(S–M)*

**File.** `covariate.py` (extend `minimaltest`).

- Full closed-union `CovariateOp` set from the `minimaltest` surfaces
  (`Shorthand`, `Derivative`, `Power`, `CompCorSelect` via `{{…}}`, `Indicator`,
  `SetOp`, `Scatter`). Shorthand expansions as a preprocessor (the
  `confound_formula_preprocessor` pattern). Emit-only.

**Tests.** 36P `(dd_(rps+wm+csf+gsr))^^2`, spike `:::`/`OR_`/`I_`, aCompCor `v_`
→ expected `CovariateProgram` + term set.

**Acceptance gate.** README/PoC confound formulae emit faithful programs.

---

## Phase 7 — Validation, errors, contract, BIDS-SM importer  *(M–L)*

**Files.** `validate.py`, `docs/nwx/engine-contract.md`, a BIDS-SM importer.

- `validate(graph) -> tuple[Diagnostic, ...]` per spec §8 (rank/identifiability,
  intercept, RE well-formedness, factor-`by` main-effect, non-aggressive needs
  signal, reserved-name shadowing, directive scoping, multi-level DAG +
  `Edge.carry.contrast` resolves, backend-awareness warnings, `.` disambiguation).
- Located errors via `GrammarErrorHandler` (nwx token categories).
- **Engine contract doc** (the dispatch table from spec §9; FLAME varcope/3-level
  chaining; FE/ME) + a **dry-run dispatcher** under `tests/nwx/` (not just
  `examples/`) that prints the intended `nitrix` call sequence without importing
  nitrix and round-trips spec §11.
- **BIDS Stats Models importer (read-direction)**: `model.json` → `ModelGraph`
  (Nodes→ModelNode, Edges→Edge, Transformations→CovariateProgram where they map,
  Contrasts/DummyContrasts→ContrastSpec/Carry) + a documented mapping table.
  Export is a follow-up.

**Acceptance gate.** §11 examples validate; helpful diagnostics; dry-run
dispatcher + BIDS-SM round-trip green.

---

## Ordering, parallelism, milestones

```
Phase 0 ─▶ Phase 1 ─▶ Phase 2 (runnable slice) ─┬─▶ Phase 3 ─┐
                                                 ├─▶ Phase 4 ─┼─▶ Phase 5 ─▶ Phase 7
                                                 └─▶ Phase 6 ─┘
```

- 0→1→2 sequential (2 proves the contract; everything after extends a *runnable*
  base).
- After 2, Phases 3/4/6 are parallel (disjoint `transform_*` / `covariate`
  files), but the **conflict gate runs on each integration merge** (global
  property). 5 depends on 4 (directive `level`/`combine`) + the residualise
  primitive. 7 last.
- **M1** (end P2): nwx usable formula→results. **M2** (end P4): GAM/GLMM-shaped
  formulae + full directives. **M3** (end P5): residualisation + multi-level —
  the distinctive surface. **M4** (end P7): validated, contract-proven, BIDS-SM
  importable.

## Test infrastructure

- `tests/nwx/`; `parser_conflicts()` (errorlog capture) gate; golden-by-value IR
  fixtures; a `TermSpec`↔`formulaic.Term` bijection helper for PoC parity; a
  string-level lme4/mgcv ↔ nwx fixture table; Phase-2 numeric oracle vs direct
  `nitrix`/`statsmodels`. Existing `test_wilkinson_formulaic.py` stays green
  (the `formulaic` interpreter is untouched).

## Risk register

| ID | Risk | Phase | Mitigation |
|---|---|---|---|
| R1 | bar-in-parens LR conflicts | 3 | analysis says none; errorlog gate; fallback `re(...)` |
| R2 | directive mini-language scope creep | 4 | closed v1 key set; exclusive state; unknown keys WARN |
| R3 | IR features outrun nitrix | 3–5 | backend-awareness WARNINGs cite v3 FR; IR carries forward-compat |
| R4 | dedup/intercept regressions | 1 | port PoC `to_terms` tests onto TermSpec |
| R5 | `.` referent overloaded (complement vs inbound cope) | 5 | resolve by position; test both |
| R6 | reserved-name shadowing (`s`/`signal`/…) | 4 | special only in call position; validate + test `y~s` vs `y~s(x)` |
| R7 | new-token maximal munch (`||`>`|`) | 3 | function-free string tokens; longest-match test |

*(The earlier external-consumer risk is retired: breakage of the old `-ops`
consumers is accepted per the owner — they are being rebuilt.)*

## Cross-references

- Spec: `docs/nwx/spec.md`.
- nitrix kernels: `nitrix/docs/feature-requests/stats-modelling-suite-v3.md`
  (the `lme_fit` structure-dispatch ladder, GAMM surfacing, non-aggressive
  residualisation, dof, families).
- Substrate: `src/gramform/core.py`, `grammars/wilkinson/`,
  `grammars/minimaltest/`, `postprocessors.py`; harvested vocabulary:
  `docs/nwx/covariate-vocabulary.md` (Phase 0).
```
