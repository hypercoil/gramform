# `nwx` — Neuroimaging Wilkinson Extension

**Grammar specification & implementation plan**

> **Status (2026-06-17): design locked, no implementation yet.** Branch `nwx`
> (off `ply`). Builds on the Wilkinson proof-of-concept in
> `src/gramform/grammars/wilkinson/`. The four spine decisions (§1) are locked;
> the two structural problems flagged for this round — directive scoping in
> multi-stage designs (§4.5) and explicit signal/noise separation in
> residualisation (§4.6) — are resolved here. Companion: the `nitrix`
> feature-request `docs/feature-requests/stats-modelling-suite-v3.md`, which
> tracks the numeric kernels `nwx` lowers onto.

---

## Part I — Specification

### §0. Purpose

`nwx` is a domain-specific language for **specifying** statistical models for
neuroimaging — generalized linear / additive / (generalized) linear mixed
models, confound residualisation, and multi-level (run → subject → dataset)
designs — in a compact Wilkinson-formula surface. It is built on the `gramform`
substrate (composable `GrammarComponent`s → frozen `Primitive` AST →
`InterpretersDispatch` over an immutable `ExecutionContext`).

The defining decision: `nwx` does **not** emit a design matrix. It emits a
typed, immutable **`ModelSpec`** intermediate representation (IR). A *model* is
more than a design matrix — it carries a family/link, random effects, penalised
/ smooth terms, an error/correlation structure, a residualisation *mode*, an
estimator, contrasts, and (for neuroimaging) a multi-level dataflow. An external
**engine** lowers the `ModelSpec` onto `nitrix` calls. `gramform`/`nwx` never
imports `nitrix` or `jax`.

This split mirrors two systems `nwx` deliberately learns from:

- **formulaic** — `Formula` (specification) vs `ModelMatrix` (materialisation).
- **BIDS Stats Models** — a declarative JSON spec, a graph of `Nodes`+`Edges`,
  consumed by an executor (fitlins) that dispatches to a numeric backend
  (nilearn / AFNI). The spec is portable across backends; the multi-level
  structure is a node graph with explicit dataflow edges.

`nwx` is the specification layer for both ideas, pushed up one level of
abstraction (from *design matrix* to *model*) and targeted at `nitrix`.

### §1. Concern boundaries (locked)

Three layers; `nwx` owns exactly the middle one.

| Layer | Home | Responsibility |
|---|---|---|
| **A. Syntax** | `gramform.core` (unchanged) | tokens, productions, `Primitive` AST |
| **B. Model specification — `nwx`** | `gramform` (new) | formula → immutable `ModelSpec` + `CovariateProgram`; static validation; carry contrasts + multi-level graph |
| **C. Lowering / execution — the engine** | **NOT `gramform`** (separate consumer) | bind data; choose `nitrix` routines; manage the mass axis (`vmap`); run inference |

**`nwx` DOES**: parse; build `ModelSpec`; statically validate (rank /
identifiability, nesting consistency, intercept handling, directive scoping);
emit a backend-agnostic `CovariateProgram` (the confound / derivative /
spline-as-data column transforms); carry estimands and the stage graph.

**`nwx` DOES NOT**: import `nitrix` or `jax`; allocate arrays; choose solvers
(REML vs OLS vs FLAME); know which axis is voxels / vertices / edges / fixels;
perform any fit or inference.

**Contract**: `nwx` defines the IR dataclasses + `Protocol`s; the engine
consumes them. That is the only coupling.

**Locked decisions (2026-06-17):**

1. **Scope guarantee** = GLM / GAM / (G)LMM + residualisation + the multi-level
   node graph. The IR *reserves* a multivariate response shape (`mvbind`) as a
   future extension but does not promise it in v1. Connectivity
   (cov / glasso / PCA) is out of scope.
2. **Model directives** (family, link, estimator, error structure, contrasts,
   inference) live in a trailing `{{ … }}` block (reuses the legacy `dfops`
   parameter-block idiom), node-scoped per §4.5.
3. **Random effects** use the lme4 bar-in-parens idiom: `(1 + x | g)`,
   `(x || g)`, `(1 | g1/g2)`, `(1 | g1:g2)`.
4. **`nwx` is emit-only** for covariates: it returns a `CovariateProgram`; the
   engine materialises columns. `nwx` stays array-free.

### §2. Design principles

- **Immutability throughout.** Every IR node is a `@dataclass(frozen=True)`;
  collections are `tuple` / `frozendict`. The IR is hashable and
  pytree-friendly so the engine can carry it as a static argument.
- **Rigorous typing.** Closed sets (`Family`, `Link`, residualisation `Mode`,
  `Level`, basis `Kind`, contrast `Test`) are `enum.Enum` / `Literal`; open
  extensibility points (`SpecNode`, `TermSource`, `Lowerable`-from-the-engine)
  are `typing.Protocol`. No `Any` in the public IR surface.
- **Pure functions.** The interpreter returns new `ExecutionContext`s; no
  mutation. Consistent with the existing `gramform` interpreters.
- **ruff.** `line-length = 79`, `E,F,W,I001` (per `pyproject.toml`).
- **Reuse the substrate.** New surface is added as `GrammarComponent`s and
  registered `InterpretersDispatch` operations, never by forking the core.

### §3. Lexical grammar

`nwx` is the existing Wilkinson token set plus a small number of additions.
Existing tokens (unchanged): `APPEND (+)`, `REMOVE (-)`, `INTERACTIONS (*)`,
`INTERACTION_ONLY (:)`, `NESTED (/)`, `NESTED_IN (%in%)`, `POWER (^, **)`,
`LPAREN/RPAREN`, `NAME`, `NAME_LITERAL` (backticks), `DOT (.)`, `EXECUTE
({…})`, `LHS_RHS_SEPARATOR (~)`, `PARTS_SEPARATOR (|)`,
`LHS_RESIDUAL_RHS (~|)`, `LBRACKET/RBRACKET ([ ])`, the parameterised
function-call machinery, and the `ASSIGN/PARAM_SEPARATOR` family.

**New tokens:**

| Token | Lexeme | Role |
|---|---|---|
| `RANEF_UNCORR` | `\|\|` | uncorrelated random effects (inside parens) |
| `STAGE_PIPE` | `>>` | multi-level pipeline between nodes |
| `LHS_RESIDUAL_AGGRESSIVE` | `~\|!` | aggressive residualisation (full nuisance projection) |
| `SPEC_OPEN` / `SPEC_CLOSE` | `{{` / `}}` | directive-block delimiters (open a `spec` lexer state) |

**New lexer state `spec`** (entered on `{{`, exited on `}}`), reusing the state
mechanism the legacy `dfops` grammar used for `{{ … }}`: inside it, `;`
separates directives, `=` is key/value, `:` opens a contrasts clause, `,`
separates list items, `(`/`)` group test annotations. This keeps the directive
mini-language lexically isolated from the term algebra.

**Reserved interpretation (not new tokens).** The following are ordinary
`NAME LPAREN … RPAREN` calls given special meaning by the interpreter registry
(exactly as `s`/`te` are recognised, not lexed specially) — a *bare* `signal`
or `noise` remains a column lookup:

- smooth constructors: `s`, `te`, `ti`, `t2`;
- residualisation role markers: `signal`, `noise` (alias `nuisance`);
- error-structure constructors (inside the directive block): `ar1`, `car1`,
  `cs` (compound symmetry), `varIdent`, `varPower`.

### §4. Syntactic grammar

#### §4.1 Term algebra (existing, retained verbatim)

`+ - * : / %in% ^ ** .` and parenthesised grouping behave exactly as in the
current Wilkinson PoC and produce the same `TermSpec` set (see §5). The
intercept postprocessor (`ppr_add_intercept`) is retained.

#### §4.2 Random effects (bar-in-parens) — collision resolution

lme4 writes random effects as `( effect | group )`; `nwx`'s `|` already means
the multi-part separator at the *blocks* level. The two never coincide because
**lme4 random effects are always parenthesised**, so we scope `|` by bracket:

- A bar **inside parentheses** is the random-effects grouping operator.
- A bar **at the blocks (top) level** is the parts separator (unchanged).

New productions (schematic; precedence places `ranef` below the term algebra so
the left/right of the bar parse as full term expressions):

```
factor      : LPAREN ranef RPAREN
ranef       : re_lhs PARTS_SEPARATOR grouping        # correlated
            | re_lhs RANEF_UNCORR    grouping        # uncorrelated (||)
re_lhs      : expression                             # e.g. 1 + x, 0 + x, x - 1
grouping    : term                                   # g, g1/g2 (nested), g1:g2 (crossed)
```

Semantics (→ `RandomEffectSpec`, §5):

| Surface | Meaning |
|---|---|
| `(1 \| g)` | random intercept per level of `g` |
| `(1 + x \| g)` / `(x \| g)` | correlated random intercept + slope |
| `(0 + x \| g)` / `(x - 1 \| g)` | random slope only |
| `(x \|\| g)` | uncorrelated random effects |
| `(1 \| g1/g2)` | nested: `(1 \| g1) + (1 \| g1:g2)` |
| `(1 \| g1:g2)` | crossed grouping |

**Parser-engineering note (risk R1, §15).** Reusing `PARTS_SEPARATOR` at two
grammatical levels is the one real LR hazard (reduce/reduce around `|` and
`(...)`). Mitigation: a dedicated `ranef` non-terminal reachable only from
`LPAREN … RPAREN`, validated against the generated `parser.out` early in
Phase 2; fall back to a functional `re(1 + x, g)` form only if conflicts prove
intractable.

#### §4.3 Smooth terms (GAM / GAMM)

`s(...)`, `te(...)`, `ti(...)`, `t2(...)` reuse the existing parameterised
function-call productions; no new syntax. The interpreter recognises the four
names and parses parameters into a `SmoothSpec`:

```
s(x, k=6, bs="cr", by=dx)        # 1-D penalised smooth, factor-smooth interaction
te(x, z, k=(5,5), bs="ps")       # anisotropic tensor product
ti(x, z)                         # pure interaction (main effects excluded)
s(g, bs="re")                    # random effect AS a smooth  → GAMM bridge
s(g, x, bs="re")                 # random slope as a smooth
```

`bs="re"` / `bs="fs"` are the syntactic bridge to GAMM: a random-effect smooth
is, numerically, a ridge-penalty block — see the `nitrix` v3 FR item on
surfacing GAMM, which `nwx` depends on.

#### §4.4 Structure: `~`, `|`, frames `[ … ]`, pipeline `>>`

Retained from the PoC: `~` (LHS/RHS), `|` (multi-part blocks),
`[ … ]` push-frame (nested sub-model emitting `_hat` fitted / `_tilde`
residualised referent terms into the parent design). **New**: `>>` connects
frames into a multi-level graph (BIDS-SM `Nodes` + `Edges`).

```
graph    : node_seq directives_opt          # trailing block = GRAPH-level (§4.5)
node_seq : node
         | node_seq STAGE_PIPE node
node     : frame
         | structure directives_opt          # implicit outermost node
frame    : LBRACKET structure directives_opt RBRACKET
```

A `>>` chain produces a `ModelGraph` whose nodes are the stages; lower-stage
estimands (copes/varcopes) flow to the next stage as inputs (the BIDS-SM
`DummyContrasts` pass-up), recorded as `Edge`s.

#### §4.5 Directive block `{{ … }}` and its scoping rule — **Problem A**

The flagged hazard: in a multi-stage design, to which node does a `{{ … }}`
block belong? Resolution — **a directive block is owned by exactly one bracket
scope, structurally, never by proximity**:

> **Scoping rule.** `directives` is the optional *final clause* within a
> bracket-delimited scope. The bracket pairs `[ … ]` are the node boundaries;
> the whole input is an implicit outermost bracket. A `{{ … }}` binds to the
> node delimited by the innermost bracket pair that lexically encloses it. If a
> `{{ … }}` follows the entire top-level `>>` pipeline (outside every frame),
> it is a **graph-level** directive. A `{{ … }}` in any other position is a
> parse error.

Consequences (all unambiguous by construction):

```
y ~ x + s(age) {{ family=gaussian; estimator=reml }}
        └─ binds to the single (implicit outermost) node

[ bold ~| rps + wm + csf ]                       ~ task + (1 + task | subject)
  └─ inner frame: no directives → defaults         {{ estimator=reml;
                                                       correlation=ar1(session|subject) }}
                                                   └─ outermost node

[ cope ~ cond {{ level=subject; group_by=subject }} ]
  >> [ . ~ 1 {{ level=dataset; estimator=flame }} ]
       └─ each {{…}} is inside its own frame → unambiguously that stage's node

[ cope ~ cond {{ level=subject }} ] >> [ . ~ 1 {{ level=dataset }} ]
        {{ inference=permutation(tfce, n=5000) }}
        └─ after the whole pipeline, outside all frames → GRAPH-level
```

Because *node-level* directives live **inside** a frame's brackets and
*graph-level* directives live **after** the pipeline (outside all brackets),
the two are distinguished by bracket position alone — no lookahead heuristic,
no precedence trick. Level and `group_by` are ordinary directive keys (this
also sidesteps the `subject:`-label-vs-`:`-interaction collision); a future
`[subject: …]` label sugar is possible but deferred pending lexer
disambiguation.

**Directive grammar (the `spec` state):**

```
directives : SPEC_OPEN directive (";" directive)* SPEC_CLOSE
directive  : key "=" value                       # family=binomial, k=6, level=subject
           | "contrasts" ":" contrast ("," contrast)*
           | "inference" "=" infer_call
contrast   : NAME "=" expression ("(" test ")")?  # age = age (t),  grp = a - b (F)
test       : "t" | "F"
```

Directive keys recognised in v1: `family`, `link`, `estimator`, `correlation`,
`weights`, `vcov`, `dof`, `level`, `group_by`, `contrasts`, `inference`,
`method`. Unknown keys are a validation warning (forward-compatibility for
not-yet-supported kernels), not a hard error.

#### §4.6 Residualisation with explicit signal/noise — **Problem B**

The flagged need: a grammar means to **separate signal from noise** in
residualisation, so that non-aggressive (signal-preserving) cleaning is
expressible.

Two residualisation modes, distinguished by operator glyph:

| Operator | Mode | Meaning |
|---|---|---|
| `~\|` | **non-aggressive** (default) | regress target on `[signal + noise]` jointly; remove only the noise-*unique* fitted contribution (preserve variance shared between signal and noise) — the ICA-AROMA scheme |
| `~\|!` | **aggressive** | project the target onto the orthogonal complement of the *full* noise subspace (`residualise(Y, X=noise)`) — removes shared variance too |

Role markers make the split explicit anywhere a term appears:

- `noise(T)` (alias `nuisance(T)`) tags terms `T` as nuisance role.
- `signal(T)` tags terms `T` as effect-of-interest role.
- **Defaults**: on a normal model RHS, unwrapped terms are *signal*; on the
  RHS of a residualisation (`~|` / `~|!`), unwrapped terms are *noise* (the RHS
  of a residualisation is, by definition, "what to remove").

```
# Aggressive cleaning (remove the full confound subspace):
bold ~|! rps + wm + csf

# Non-aggressive AROMA cleaning (remove noise-unique variance, keep signal-shared):
bold ~| noise(aroma_ic_01 + aroma_ic_02) + signal(task + drift)

# In-model partialling (statistically the correct non-aggressive partial):
y ~ signal(task) + noise(meanFD + rps)
    # confounds partialled out of the reported task effect, not themselves reported
```

The interpreter lowers these to a `ResidualiseSpec{target, noise, signal,
mode}` (§5). Non-aggressive mode **requires** a non-empty `signal` set (else it
is equivalent to aggressive, and the validator emits a warning). Aggressive
mode ignores `signal(...)` (warning if present). The existing `~|` referent
mechanism (`_tilde` terms feeding a parent design) is retained; `~|!` produces
the same referent shape with `mode=AGGRESSIVE`.

`nitrix.linalg.residualise` covers the aggressive path today; the
non-aggressive partial scheme is a v3 FR item (the AROMA partial regression is
not a pure projection).

### §5. The `ModelSpec` IR

Schematic typed schema (the normative field list; final dataclasses live in
`gramform/grammars/nwx/spec.py`). Frozen, hashable, `tuple`-valued.

```python
class Level(Enum):       RUN; SESSION; SUBJECT; DATASET
class Family(Enum):      GAUSSIAN; BINOMIAL; POISSON; GAMMA; NEGBINOMIAL; ...
class Link(Enum):        IDENTITY; LOG; LOGIT; ...
class Mode(Enum):        AGGRESSIVE; NONAGGRESSIVE; SOFT
class Test(Enum):        T; F
class BasisKind(Enum):   PS; CC; TPRS; CR; GP; MRF; RE; FS; TENSOR

class SpecNode(Protocol):            # extensibility marker
    kind: str

@dataclass(frozen=True)
class FactorSource(Protocol): ...     # Lookup | Literal | PyExpr | Smooth | Referent
@dataclass(frozen=True)
class TermSpec:        factors: tuple[FactorSpec, ...]; order: int
@dataclass(frozen=True)
class FactorSpec:      source: FactorSource; coding: ContrastCoding | None
@dataclass(frozen=True)
class SmoothSpec:      covariates: tuple[FactorSpec, ...]; basis: BasisKind
                       k: int; penalty_order: int; by: FactorSpec | None
                       cyclic: bool; tensor: bool
@dataclass(frozen=True)
class GroupingSpec:    factors: tuple[FactorSpec, ...]; relation: Literal['nested','crossed']
@dataclass(frozen=True)
class RandomEffectSpec: group: GroupingSpec; terms: tuple[TermSpec, ...]
                        correlated: bool
@dataclass(frozen=True)
class ErrorSpec:       correlation: CorrelationSpec | None
                       heteroscedasticity: WeightSpec | None
@dataclass(frozen=True)
class ResidualiseSpec: target: tuple[TermSpec, ...]
                       noise: tuple[TermSpec, ...]
                       signal: tuple[TermSpec, ...]
                       mode: Mode; l2: float
@dataclass(frozen=True)
class FamilySpec:      family: Family; link: Link
@dataclass(frozen=True)
class ContrastSpec:    name: str; weights: frozendict[str, float]; test: Test
@dataclass(frozen=True)
class EstimationSpec:  method: Literal['ols','wls','irls','reml','ml','flame']
                       ridge: float; dof: Literal['residual','satterthwaite','kr'] | None
                       vcov: Literal['standard','hc0','hc1','hc2','hc3','cluster'] | None
@dataclass(frozen=True)
class InferenceSpec:   kind: Literal['parametric','permutation']
                       enhancement: Literal['voxel','tfce','cluster_extent','cluster_mass'] | None
                       n_perm: int | None; correction: Literal['fdr','bonferroni','fwe','rft'] | None

@dataclass(frozen=True)
class ResponseSpec:    terms: tuple[TermSpec, ...]; aux: frozendict[str, FactorSpec]  # trials(n), weights(w)

@dataclass(frozen=True)
class ModelSpec:
    response:    ResponseSpec
    fixed:       tuple[TermSpec, ...]
    random:      tuple[RandomEffectSpec, ...]
    smooth:      tuple[SmoothSpec, ...]
    family:      FamilySpec
    errors:      ErrorSpec | None
    residualise: tuple[ResidualiseSpec, ...]
    estimands:   tuple[ContrastSpec, ...]
    estimation:  EstimationSpec
    inference:   InferenceSpec | None
    covariates:  CovariateProgram

@dataclass(frozen=True)
class ModelNode:  level: Level; group_by: tuple[str, ...]; spec: ModelSpec; name: str
@dataclass(frozen=True)
class Edge:       source: str; dest: str; filter: frozendict[str, str]; carry: Carry
@dataclass(frozen=True)
class ModelGraph: nodes: tuple[ModelNode, ...]; edges: tuple[Edge, ...]
```

A single-node formula lowers to a `ModelGraph` with one `ModelNode`. The engine
dispatches structurally on the populated fields:

- `random` non-empty / `errors` set → `reml_fit` / (G)LMM;
- `smooth` non-empty → `gam_fit` (with random-effect smooths → GAMM);
- otherwise → `glm_fit`;
- a `DATASET` node fed by `SUBJECT` copes → `flame_two_level`.

### §6. The `CovariateProgram` (harvested confound vocabulary)

`nwx` emits — never executes (decision 4) — a `CovariateProgram`: an ordered,
immutable list of column transforms over the *tabular covariate frame*. This is
the salvage of the legacy `dfops` vocabulary, re-expressed as IR:

| Surface | Transform | Notes |
|---|---|---|
| `rps`, `wm`, `csf`, `gsr`, `acc`, `fd`, `dv` | shorthand expansion | fMRIPrep regressor names |
| `d1(x)`, `dd1(x)` | backward difference (exclusive / inclusive) | confound derivatives |
| `x^^2` | inclusive power | quadratic expansion |
| `v_{…}` | cumulative-variance CompCor selection | component count by variance |
| `I_[>θ](x)` | indicator | spike regressors |
| `AND_/OR_/NOT_`, `:::` (scatter) | set / scatter ops | spike-regressor assembly |

The program is engine-materialised against a covariate dataframe (via the
engine's `narwhals` layer). `nwx` validates references and operator arity but
holds no array.

### §7. Device → IR → `nitrix` map (with backend status)

| Device | IR node | `nitrix` target | Status |
|---|---|---|---|
| `+ - * : / ^` term algebra | `TermSpec` | design assembled by engine | ✅ exists (PoC) |
| `(1+x\|g)`, `\|\|`, `/`, `:` | `RandomEffectSpec` | `reml_fit` | ⚠️ **v3**: REML is 2-component only |
| `s/te/ti/t2(...)` | `SmoothSpec` | `bspline/cyclic/tprs/tensor_product_basis` + `gam_fit` | ✅ v1/v2 (bases `ps/cc/tp/te`) |
| `s(g, bs="re"/"fs")` | `SmoothSpec(basis=RE/FS)` | `gam_fit` penalty block | ⚠️ **v3**: GAMM not surfaced |
| `{{ family=…; link=… }}` | `FamilySpec` | `Family` registry | ✅ Gaussian/Binomial/Poisson; ⚠️ **v3**: more families |
| `{{ correlation=ar1(…) }}` | `ErrorSpec` | — | ⚠️ **v3**: AR/CAR/CS absent |
| `{{ vcov=hc3 / cluster }}` | `EstimationSpec.vcov` | — | ⚠️ **v3**: sandwich/cluster SEs |
| `{{ dof=satterthwaite }}` | `EstimationSpec.dof` | — | ⚠️ **v3**: mixed-model dof |
| `~\|` (non-aggressive) | `ResidualiseSpec(NONAGGRESSIVE)` | — | ⚠️ **v3**: partial residualisation |
| `~\|!` (aggressive) | `ResidualiseSpec(AGGRESSIVE)` | `linalg.residualise` | ✅ exists |
| `{{ contrasts: … }}` | `ContrastSpec` | `t_contrast` / `f_contrast` | ✅ exists |
| `{{ inference=permutation(tfce) }}` | `InferenceSpec` | `permutation_test` / `fdr_bh` | ✅ v1/v2 |
| `[A] >> [B]` graph | `ModelGraph` | per-node `glm_fit` → `flame_two_level` | ✅ kernels; engine orchestrates |

### §8. Static validation (in `nwx`, before any data)

- **Identifiability / rank**: duplicate terms (up to a constant scale) rejected
  (the PoC already does this); intercept handling consistent across stages.
- **Random effects**: grouping factor must be categorical-by-declaration;
  nested/crossed relation well-formed; no random slope without its fixed effect
  unless explicitly suppressed.
- **Residualisation**: non-aggressive mode requires a `signal(...)` set;
  aggressive mode warns on `signal(...)`.
- **Directive scoping**: a `{{ … }}` outside a legal final-clause position is a
  parse error with a located message (reuse `GrammarErrorHandler`).
- **Multi-level**: each `>>` stage carries a `level`; `group_by` references
  resolve; edges form a DAG.
- **Backend awareness**: directives naming a kernel not in the current `nitrix`
  surface (per §7 "v3") validate with a warning, not an error, so specs are
  forward-compatible.

### §9. Engine contract (informative — not built here)

The engine (a separate consumer, e.g. in `hypercoil` or a new `nwx-engine`)
receives `(ModelGraph, covariate_df, imaging_data)` and:

1. materialises the `CovariateProgram` into covariate columns;
2. assembles per-node fixed design `X`, random design `Z`, smooth bases;
3. dispatches each node to the right `nitrix` routine (§5/§7), `vmap`-ing over
   the mass axis;
4. propagates lower-stage copes/varcopes along `Edge`s;
5. runs the `InferenceSpec` (permutation / parametric) and correction.

`nwx` guarantees the IR is complete and validated; the engine guarantees the
numerics. Neither imports the other's internals.

### §10. Scope

**In scope (v1 guarantee):** GLM, GAM, GAMM, LMM/GLMM, residualisation
(aggressive / non-aggressive), multi-level node graph; mass-univariate *and*
single-fit (the mass axis is engine data-binding, so one spec serves both).

**Reserved (IR-ready, not promised):** multivariate response (`mvbind`),
distributional / location-scale parts (`sigma ~ …`), a `method=` directive for
non-GLM methods.

**Out of scope:** connectivity estimators (cov / corr / partial / glasso /
PCA — not "Wilkinson-shaped"); array numerics; solver selection; the imaging
modality / mass axis; CLI / file-format parsing (BIDS, `.con`/`.mat`).

### §11. Worked examples

```
# 1. Vertexwise group GAMM, confounds partialled, TFCE permutation inference
thk ~ s(age, k=6, by=dx) + dx + sex + noise(meanFD) + (1 | site)
      {{ family=gaussian; estimator=reml;
         contrasts: age_by_dx = s(age):dx (F), dx_main = dx (t);
         inference=permutation(tfce, n=5000) }}

# 2. Two-stage cleaning then longitudinal LMM with AR(1) errors
[ bold ~| noise(rps + wm + csf) + signal(task) ]
  ~ task + (1 + task | subject)
  {{ estimator=reml; correlation=ar1(session | subject); dof=satterthwaite }}

# 3. Multi-level run → subject → dataset (BIDS-SM shape)
[ cope ~ cond {{ level=subject; group_by=subject }} ]
  >> [ . ~ 1 {{ level=dataset; estimator=flame; contrasts: grp = 1 (t) }} ]
     {{ inference=permutation(cluster_mass, n=5000) }}
```

---

## Part II — Implementation plan

### §12. Module layout (additions under `src/gramform/grammars/nwx/`)

```
grammars/nwx/
  grammar.py        # GrammarComponents: reuse Wilkinson + RanefComponent,
                    #   PipelineComponent, DirectiveComponent (spec state),
                    #   ResidualRoleComponent
  spec.py           # the ModelSpec IR (frozen dataclasses + Protocols, §5)
  covariate.py      # CovariateProgram IR + the harvested dfops vocabulary
  transform.py      # the `spec` interpreter: Primitive AST -> ModelSpec
  directives.py     # directive-block mini-parser + scoping (§4.5)
  validate.py       # static checks (§8)
```

The existing `wilkinson/` components are imported and extended, not copied. A
`spec` interpreter is registered alongside the current `formulaic` interpreter
(the latter stays available for design-matrix-only use).

### §13. Phasing

- **Phase 0 — cruft removal & harvest.** Delete the pre-`ply` modules
  (top-level `grammar.py`, `dfops.py`, `imops.py`, `tagops.py`, empty
  `grammar/` package) after confirming nothing imports them (`__init__`
  already comments them out; `core` still needs `resampler` + `error`).
  **Harvest** the `dfops` confound vocabulary into `covariate.py` *before*
  deletion. Gate: test suite green.
- **Phase 1 — retarget.** `spec.py` IR + `Protocol`s; the `spec` interpreter
  producing `ModelSpec` for the existing term algebra + structure; golden tests
  vs the current PoC output (same designs, new IR).
- **Phase 2 — random effects.** `RanefComponent` (bar-in-parens, `||`,
  nesting/crossing) → `RandomEffectSpec`; validate `parser.out` for conflicts
  (risk R1).
- **Phase 3 — smooths + directives.** `s/te/ti/t2` → `SmoothSpec`;
  `DirectiveComponent` (`spec` state) + the §4.5 scoping rule; family /
  estimation / errors / inference / contrasts directives.
- **Phase 4 — residualisation modes + multi-level.** `~|` / `~|!` +
  `signal()`/`noise()` roles → `ResidualiseSpec`; `PipelineComponent` (`>>`) +
  level/group_by → `ModelGraph`; graph-level vs node-level directive binding.
- **Phase 5 — covariate program.** Wire the harvested vocabulary into
  `CovariateProgram` emission.
- **Phase 6 — validation, errors, contract.** `validate.py`; located error
  messages via `GrammarErrorHandler`; document the engine contract; ship a
  reference engine stub (separate repo) exercising the IR end-to-end.

### §14. Testing & oracles

- **Parse equivalence**: term-algebra parses match the current PoC / formulaic
  (reuse `tests/test_wilkinson_formulaic.py` patterns).
- **IR golden tests**: canonical formulae → expected `ModelSpec` (frozen,
  comparable by value).
- **Cross-language exemplars**: random-effects and smooth surfaces checked
  against lme4 / mgcv formula semantics (string-level, since `nwx` does not
  fit) — pinned fixtures, since web access for live extraction is unavailable
  in CI.
- **Scoping**: a matrix of multi-stage formulae asserting each `{{…}}` binds to
  the intended node / graph; malformed positions raise located errors.
- **Residualisation roles**: signal/noise tagging and mode selection produce
  the expected `ResidualiseSpec`; non-aggressive-without-signal warns.

### §15. Risks & open decisions

- **R1 — bar-in-parens LR conflicts** (§4.2). Validate early; fallback `re(...)`.
- **R2 — directive mini-language scope creep.** Keep v1 keys closed (§4.5);
  unknown keys warn, not error.
- **R3 — backend lag.** Several IR features (general random effects, GAMM
  surfacing, non-aggressive residualisation, error structures, mixed-model dof,
  more families) need `nitrix` work tracked in the v3 FR; the IR is designed to
  *carry* them now and validate-with-warning until the kernels land.
- **Open**: whether the multi-level `>>` graph should also accept a pure-Python
  builder (for programmatic node assembly) in addition to the string surface.

### §16. Cross-references

- `nitrix` feature request (the kernels `nwx` lowers onto):
  `nitrix/docs/feature-requests/stats-modelling-suite-v3.md` (this round) and
  its predecessors `stats-modelling-suite.md` (v1, shipped) /
  `stats-modelling-suite-v2.md` (v2).
- Substrate: `src/gramform/core.py`, `src/gramform/grammars/wilkinson/`,
  `src/gramform/postprocessors.py`.
