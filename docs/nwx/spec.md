# `nwx` — Neuroimaging Wilkinson Extension

**Grammar specification & design**

> **Status (2026-06-17): design locked; revised after a four-lens review**
> (engineering rigour, statistical correctness, community usefulness, design /
> abstraction). Branch `nwx` (off `ply`). Builds on the Wilkinson
> proof-of-concept in `src/gramform/grammars/wilkinson/` and the confound
> vocabulary already ported to `src/gramform/grammars/dataframe/`. The
> implementation plan is `docs/nwx/implementation-plan.md`.
>
> **Review-driven changes in this revision:** residualisation is now
> **aggressive-by-default** (the safe, implementable case), with non-aggressive
> partial regression an explicit opt-in (§4.6); the IR contract types are fully
> defined (no `frozendict`; `Carry`/`Diagnostic` defined; closed unions over
> `Protocol` for `nwx`-owned variants) (§5); solver-dispatch and numeric knobs
> are moved out of the IR into the engine contract (§9); statistical fidelity
> fixes to random effects, smooths, and multi-level FLAME (§4.2, §4.3, §9); a
> runnable vertical slice is sequenced first (see the plan); and a BIDS Stats
> Models importer joins the roadmap (§10).

---

## §0. Purpose

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
imports `nitrix` or `jax` (a post-Phase-0 invariant — §1 — enforced by a CI
import guard).

This split mirrors two systems `nwx` deliberately learns from:

- **formulaic** — `Formula` (specification) vs `ModelMatrix` (materialisation).
- **BIDS Stats Models** — a declarative JSON spec, a graph of `Nodes`+`Edges`,
  consumed by an executor (fitlins) that dispatches to a numeric backend. The
  spec is portable across backends; the multi-level structure is a node graph
  with explicit dataflow edges. (`nwx` adds a read-direction importer for this
  format — §10.)

`nwx` is the specification layer for both ideas, pushed up one level of
abstraction (from *design matrix* to *model*) and targeted at `nitrix`.

## §1. Concern boundaries (locked)

Three layers; `nwx` owns exactly the middle one.

| Layer | Home | Responsibility |
|---|---|---|
| **A. Syntax** | `gramform.core` (unchanged) | tokens, productions, `Primitive` AST |
| **B. Model specification — `nwx`** | `gramform` (new) | formula → immutable `ModelSpec` + `CovariateProgram`; static validation; carry contrasts + multi-level graph |
| **C. Lowering / execution — the engine** | **NOT `gramform`** (separate consumer) | bind data; choose `nitrix` routines; manage the mass axis (`vmap`); run inference |

**`nwx` DOES**: parse; build `ModelSpec`; statically validate (rank /
identifiability, nesting consistency, intercept handling, directive scoping,
backend-awareness); emit a backend-agnostic `CovariateProgram`; carry estimands
and the stage graph.

**`nwx` DOES NOT**: import `nitrix` or `jax`; allocate arrays; choose solvers;
know which axis is voxels / vertices / edges / fixels; perform any fit or
inference.

**The jax/nitrix firewall is a post-Phase-0 invariant.** `gramform` *currently*
imports `jax` (via `imops.py`, deleted in Phase 0). After Phase 0 the `nwx`
import surface pulls in neither `jax` nor `nitrix`; a CI test asserts this
(`import gramform.grammars.nwx` must not transitively import `jax`/`nitrix`).

**Contract**: `nwx` defines the IR dataclasses + the single engine-facing
`Lowerable` `Protocol`; the engine consumes them. That is the only coupling.

**Locked decisions (2026-06-17):**

1. **Scope guarantee** = GLM / GAM / (G)LMM + residualisation + the multi-level
   node graph. The IR *reserves* a multivariate response shape (`mvbind`) and a
   distributional part (`sigma ~ …`) as future extensions but does not promise
   them in v1. Connectivity (cov / glasso / PCA) is out of scope (§10).
2. **Model directives** (family, link, estimator, error structure, contrasts,
   inference) live in a trailing `{{ … }}` block, node-scoped (§4.5), parsed in
   a dedicated **exclusive** lexer state (the `{{…}}` push/pop-state pattern is
   demonstrated in `grammars/dataframe/`).
3. **Random effects** use the lme4 bar-in-parens idiom: `(1 + x | g)`,
   `(x || g)`, `(1 | g1/g2)`, `(1 | g1:g2)`.
4. **`nwx` is emit-only** for covariates: it returns a `CovariateProgram`; the
   engine materialises columns.
5. **Residualisation is aggressive-by-default** (§4.6): `~|` projects onto the
   orthogonal complement of the nuisance set; non-aggressive partial regression
   is an explicit opt-in.

### §2. Design principles

- **Immutability throughout.** Every IR node is `@dataclass(frozen=True)`;
  collections are `tuple`. **No mappings in IR fields** — key/value data uses
  `tuple[tuple[K, V], ...]` (hashable, pytree-clean, value-comparable for golden
  tests). There is no `frozendict`.
- **Typed open/closed cut.** Closed sets defined by `nwx` are `Enum` / `Literal`
  / closed `Union`s; `match` over a union gives exhaustiveness under the pyright
  gate. `Protocol` is reserved for the **one genuinely open, engine-owned seam**
  (`Lowerable`). `TermSource`, `CovariateOp`, etc. are *closed unions*, not
  Protocols (the engine consumes them; nobody adds a variant without an engine
  that understands it).
- **Pure functions.** The interpreter returns new `ExecutionContext`s; no
  mutation.
- **Tooling.** ruff (`E,F,W,I001`, line-length 79) **and** `blue --check` (both
  in `noxfile`); `pyright` (wired into the nox `tests` session by Phase 0);
  coverage ≥ 90 % absolute (the existing nox floor).
- **Reuse the substrate.** New surface = new `GrammarComponent`s + registered
  `InterpretersDispatch` operations, never a fork of the core.

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
| `SPEC_OPEN` / `SPEC_CLOSE` | `{{` / `}}` | directive-block delimiters (open the `spec` state) |

`~|` is the (now aggressive-default) residualisation operator; **no separate
aggressive glyph is added** — non-aggressive is a directive opt-in (§4.6), per
the review's "prefer a self-documenting directive over a cryptic glyph".

**Lexer ordering.** New multi-character tokens must out-rank their prefixes by
longest match. `||` must out-rank `|`; these are function-free string tokens, so
PLY orders them by descending regex length automatically — a lexer test asserts
`(x||g)` lexes `RANEF_UNCORR` and `a|b` lexes `PARTS_SEPARATOR`. (`>>` shares no
prefix with any default-state token; inside the `spec` state, comparison tokens
needed by `correlation=`/contrasts must be defined so as not to collide with
`>>`, which does not appear in that state.)

**New lexer state `spec` — EXCLUSIVE** (entered on `{{`, exited on `}}`), built
on `core`'s `push_state_and_return`/`pop_state_and_return` (the pattern
`grammars/dataframe/` uses for its `param` block — note `dataframe`'s state
is *inclusive*; `nwx`'s must be **exclusive** so the directive mini-language's
`:` (contrasts clause) and `=` (key/value) do **not** collide with the default
state's `INTERACTION_ONLY (:)` and `ASSIGN (=)`). The `spec` state defines its
own minimal token set (`;`, `=`, `:`, `,`, `(`, `)`, names, numbers, strings).

**Reserved interpretation (not new tokens).** Ordinary `NAME LPAREN … RPAREN`
calls given meaning by the interpreter registry (as `s`/`te` are; a *bare*
`signal`/`noise` remains a column lookup — these names are special **only in
call position**, a validated rule, §8):

- smooth constructors: `s`, `te`, `ti`, `t2`;
- residualisation role markers: `signal`, `noise` (alias `nuisance`);
- response-aux markers: `trials`, `weights`, `offset`;
- error-structure constructors (inside the directive block): `ar1`, `car1`,
  `cs`, `varIdent`, `varPower`.

### §4. Syntactic grammar

#### §4.1 Term algebra (existing, retained)

`+ - * : / %in% ^ ** .` and parenthesised grouping behave as in the Wilkinson
PoC and produce the same `TermSpec` set. **Intercept**: the `ppr_add_intercept`
postprocessor adds an intercept to each model node's fixed terms unless
suppressed by `0`/`-1`; intercept handling is pinned by test in every IR
position (fixed terms, each frame, each graph node, residualise target) — the
PoC's commented-out `init_hook` intercept path is removed in favour of the
single postprocessor.

#### §4.2 Random effects (bar-in-parens) — collision resolution

A bar **inside parentheses** is the random-effects grouping operator; a bar at
the **blocks (top) level** is the parts separator. The two never coincide
because lme4 random effects are always parenthesised.

```
factor   : LPAREN ranef RPAREN
ranef    : expression PARTS_SEPARATOR grouping     # correlated / scalar
         | expression RANEF_UNCORR    grouping     # uncorrelated (diagonal)
grouping : term                                    # g, g1/g2, g1:g2
```

Semantics → `RandomEffectSpec` (§5):

| Surface | Meaning | IR |
|---|---|---|
| `(1 \| g)` | random intercept | `structure='scalar'`, terms=(1,) |
| `(1 + x \| g)` / `(x \| g)` | correlated intercept + slope | `structure='unstructured'` |
| `(0 + x \| g)` / `(x - 1 \| g)` | slope only | terms=(x,), no intercept |
| `(1 + x \|\| g)` | uncorrelated | `structure='diagonal'` |
| `(1 \| g1/g2)` | nested → `(1\|g1) + (1\|g1:g2)` | **two** `RandomEffectSpec`s |
| `(1 \| g1:g2)` | the **interaction grouping factor** (one variance component over cells) | `GroupingSpec(relation='interaction')` |

**Statistical fidelity fixes (review):** `(1|g1:g2)` is a *single interaction
grouping factor* (one component), **not** "crossed"; genuine crossing
(`(1|g1)+(1|g2)`) is represented by *multiple* `RandomEffectSpec`s, never by one
spec — so `GroupingSpec.relation ∈ {'single','interaction'}` only. The
covariance structure is carried as `structure ∈ {'scalar','diagonal',
'unstructured'}` (not a bare `correlated: bool`), matching the nitrix v3 FR
`lme_fit(structure=…)` dispatch ladder. Intercept inclusion: present iff neither
`0` nor `-1` is on the bar LHS. A random slope without its fixed effect is
**legal** (lme4 permits it) — informational note, not a warning.

**Parser-engineering note (R1).** `(expr)` and `(expr | g)` differ only by
whether a bar follows `expr`; LALR shifts the bar to enter `ranef`, and the
parts-bar production lives at the `blocks` level (`blocks : blocks
PARTS_SEPARATOR block`), which is **not reachable** from inside `LPAREN
expression RPAREN` — so there is no reduce/reduce. The Phase-3 PR gate asserts
**0 conflicts** via a captured `errorlog` (PLY exposes no `parser.conflicts`
attribute — the gate parses the conflict log / `parser.out`); fallback is a
functional `re(1+x, g)` form.

#### §4.3 Smooth terms (GAM / GAMM)

`s/te/ti/t2(...)` reuse the parameterised function-call productions; the
interpreter recognises the four names (special only in call position) and builds
a `SmoothSpec`:

```
s(x, k=6, bs="cr", by=dx)        # 1-D penalised smooth; by → factor-smooth / varying-coef
te(x, z, k=(5,5), bs="ps")       # anisotropic tensor product
ti(x, z)                         # pure interaction (main effects excluded)
s(g, bs="re")                    # random intercept AS a smooth  → GAMM bridge
s(g, x, bs="re")                 # random slope of x by g (slope var → `by`)
```

**Statistical fidelity fixes (review):**

- The GAMM identity is **λ = φ/σ_b²** (not `1/σ_b²`) — exact only after the
  dispersion φ; stated correctly here.
- `by=` has two distinct meanings the IR records: **factor** `by` (a separate
  centered smooth per level — identifiability requires the factor main effect in
  the model; the validator warns if absent) vs **continuous** `by`
  (varying-coefficient `f(x)·z`, not centered). `SmoothSpec.by_kind` carries
  which.
- `SmoothSpec.bounds` carries the period for cyclic smooths (a partial-cycle
  sample must not be auto-periodised to the data range) and the knot range
  generally; `SmoothSpec.fx` carries the unpenalised (fixed-df) case.
- For `bs="re"/"fs"`, the second positional argument is the **slope variable**
  (mapped to `by`), not a smoothed covariate — matching the nitrix v3 FR
  `re_basis(g, by=x)`.

**Backend note:** plain `s/te/ti` lower onto shipped nitrix bases (`ps/cc/tp/te`);
`s(x, by=f)` factor smooths and `bs="re"/"fs"` GAMM bridges depend on the nitrix
v3 FR (§3.1, §2) and validate-with-warning until those land (§8).

#### §4.4 Structure: `~`, `|`, frames `[ … ]`, pipeline `>>`

Retained: `~` (LHS/RHS), `[ … ]` push-frame (nested sub-model emitting `_hat`
fitted / `_tilde` residualised referent terms into the parent design). **Top-
level `|`** (parts separator) is, in v1, **restricted**: a multi-part RHS is a
validation error unless it is a recognised reserved form (the distributional
`sigma ~ …` part and `mvbind` are reserved, §10) — so Phase-1's "tuple of
blocks" has a defined v1 target (single-part) and the reserved forms are gated.
**`>>`** connects frames into a multi-level graph (BIDS-SM `Nodes`+`Edges`).

```
graph    : node_seq directives_opt          # trailing block = GRAPH-level (§4.5)
node_seq : node | node_seq STAGE_PIPE node
node     : structure directives_opt          # implicit outermost node
frame    : LBRACKET structure directives_opt RBRACKET
```

#### §4.5 Directive block `{{ … }}` and its scoping rule — Problem A

> **Scoping rule.** `directives` is the optional *final clause* of a
> bracket-delimited scope. The bracket pairs `[ … ]` are the node boundaries;
> the whole input is an implicit outermost bracket. A `{{ … }}` binds to the
> node delimited by the innermost bracket pair that lexically encloses it. A
> `{{ … }}` after the entire `>>` pipeline (outside every frame) is a
> **graph-level** directive. A `{{ … }}` in any other position is a parse error.

```
y ~ x + s(age) {{ family=gaussian; estimator=reml }}      # outermost node
[ cope ~ cond {{ level=subject }} ] >> [ . ~ 1 {{ level=dataset; estimator=flame }} ]
        {{ inference=permutation(tfce, n=5000) }}          # graph-level
```

Node-level directives live *inside* a frame; graph-level live *after* the
pipeline — distinguished by bracket position alone, no lookahead.

**Directive grammar (the exclusive `spec` state):**

```
directives : SPEC_OPEN directive (";" directive)* SPEC_CLOSE
directive  : key "=" value
           | "contrasts" ":" contrast ("," contrast)*
contrast   : NAME "=" contrast_expr ("(" test ")")?     # age = age (t), grp = a - b (F)
test       : "t" | "F"
```

Contrast expressions are linear combinations of coefficient names (`a - b`,
`2*a + b`) parsed by reusing the term-algebra combinator; interaction `:` is
**not** exposed inside the exclusive `spec` state, so `a:b` is not a contrast
form (interaction contrasts are named by their assembled coefficient label).
Recognised v1 keys: `family`, `link`, `estimator`, `correlation`, `weights`,
`se`, `dof`, `level`, `group_by`, `combine`, `contrasts`, `inference`. **Only
keys whose nitrix kernel exists are honoured in v1**; keys for not-yet-shipped
kernels validate-with-warning (§8), and unknown keys warn (forward-compat).

#### §4.6 Residualisation — aggressive default + explicit signal/noise — Problem B

Residualisation is **aggressive by default** (the safe, currently-implementable
case). Mode is selected by a directive, not a glyph:

| Surface | Mode | Meaning |
|---|---|---|
| `y ~\| n` | **aggressive** (default) | project `y` onto the orthogonal complement of the full nuisance subspace `n` (`residualise(Y, X=n)`) |
| `y ~\| signal(s) + noise(n) {{ residualise=nonaggressive }}` | non-aggressive (opt-in) | regress `y` on `[s + n]` jointly; remove only the noise-*unique* fitted contribution (preserve variance shared with `s`) — the partial-regression scheme used by ICA-AROMA's non-aggressive mode (via `fsl_regfilt`) |

Role markers `signal()` / `noise()` (alias `nuisance()`) are **structural**, not
a per-term tag: the interpreter *routes* terms into the correct IR tuple. On a
residualisation RHS, unwrapped terms are noise; `signal(...)` marks variance to
preserve. **Non-aggressive requires a non-empty signal set** — a bare `~|` with
`residualise=nonaggressive` and no `signal()` is a **hard error** (not a silent
collapse to aggressive). Aggressive mode ignores `signal()` (warning).

**In-model partialling is a distinct construct, not residualisation.** On a
*normal* model RHS, `noise(...)` marks nuisance covariates that are partialled
out of the reported estimands but not themselves reported; these route to
`ModelSpec.partial`, separate from `fixed`. The reported coefficient of a signal
term under in-model partialling is the Frisch–Waugh–Lovell partial — which
coincides with the residualised coefficient but **not** with the non-aggressive
*cleaned-data product*; the two are kept distinct in the IR (`ModelSpec.partial`
vs `ResidualiseSpec`).

`~|` (aggressive) lowers onto the shipped `nitrix.linalg.residualise`;
non-aggressive lowers onto the nitrix v3 FR §5.1 `partial_residualise` and
validates-with-warning until it lands. The `_tilde` referent mechanism (a
residualise frame feeding a parent design) is retained for both modes.

### §5. The `ModelSpec` IR

Frozen, hashable, value-comparable. Closed unions for `nwx`-owned variants; no
mappings (tuple-of-pairs instead).

```python
class Level(Enum):    RUN; SESSION; SUBJECT; DATASET
class Family(Enum):   GAUSSIAN; BINOMIAL; POISSON           # + reserved: GAMMA; NEGBINOMIAL; TWEEDIE; BETA
class Link(Enum):     IDENTITY; LOG; LOGIT                  # + reserved: PROBIT; INVERSE; SQRT
class Mode(Enum):     AGGRESSIVE; NONAGGRESSIVE             # SOFT reserved (nitrix v3 §5.2), not v1
class Test(Enum):     T; F
class Relation(Enum): SINGLE; INTERACTION
class Structure(Enum):SCALAR; DIAGONAL; UNSTRUCTURED
class Combine(Enum):  FIXED; MIXED
class BasisKind(Enum):PS; CC; TPRS; TENSOR                  # shipped; reserved: CR; GP; MRF; RE; FS
class Severity(Enum): ERROR; WARNING

# --- term sources: a CLOSED union (match-exhaustive), not a Protocol ---
@dataclass(frozen=True) class Lookup:      name: str
@dataclass(frozen=True) class Const:       value: float
@dataclass(frozen=True) class PyExpr:      code: str
@dataclass(frozen=True) class CovariateRef:op_index: int            # into CovariateProgram
@dataclass(frozen=True) class Referent:    stage: str; kind: str    # "_hat" | "_tilde" | inbound cope
TermSource = Lookup | Const | PyExpr | CovariateRef | Referent

@dataclass(frozen=True) class FactorSpec: source: TermSource; coding: ContrastCoding | None = None
@dataclass(frozen=True) class TermSpec:   factors: tuple[FactorSpec, ...]   # role is STRUCTURAL, not a field
                                          # .order is a computed property
@dataclass(frozen=True)
class SmoothSpec:   covariates: tuple[FactorSpec, ...]; basis: BasisKind; k: int
                    penalty_order: int = 2; by: FactorSpec | None = None
                    by_kind: Literal['factor','continuous'] | None = None
                    cyclic: bool = False; tensor: bool = False; fx: bool = False
                    bounds: tuple[float, float] | None = None
@dataclass(frozen=True) class GroupingSpec: factors: tuple[FactorSpec, ...]; relation: Relation
@dataclass(frozen=True)
class RandomEffectSpec: group: GroupingSpec; terms: tuple[TermSpec, ...]; structure: Structure
@dataclass(frozen=True) class CorrelationSpec: kind: Literal['ar1','car1','cs']; index: FactorSpec; group: FactorSpec
@dataclass(frozen=True) class WeightSpec: kind: Literal['varIdent','varPower']; arg: FactorSpec
@dataclass(frozen=True) class ErrorSpec: correlation: CorrelationSpec | None = None; heteroscedasticity: WeightSpec | None = None
@dataclass(frozen=True)
class ResidualiseSpec: target: tuple[TermSpec, ...]; noise: tuple[TermSpec, ...]
                       signal: tuple[TermSpec, ...] = (); mode: Mode = Mode.AGGRESSIVE
@dataclass(frozen=True) class FamilySpec: family: Family = Family.GAUSSIAN; link: Link = Link.IDENTITY
@dataclass(frozen=True) class ContrastSpec: name: str; weights: tuple[tuple[str, float], ...]; test: Test
@dataclass(frozen=True)
class EstimationSpec:   estimator: Literal['ols','wls','irls','reml','ml','flame'] = 'ols'
                        se: Literal['model','robust','cluster'] = 'model'
                        robust_variant: Literal['hc0','hc1','hc2','hc3'] | None = None
                        cluster_by: str | None = None
                        dof: Literal['residual','satterthwaite','kr'] | None = None
@dataclass(frozen=True)
class InferenceSpec:    kind: Literal['parametric','permutation']
                        enhancement: Literal['voxel','tfce','cluster_extent','cluster_mass'] | None = None
                        cluster_threshold: float | None = None
                        n_perm: int | None = None
                        correction: Literal['fdr','bonferroni','fwe','rft'] | None = None
@dataclass(frozen=True) class ResponseSpec: terms: tuple[TermSpec, ...]; aux: tuple[tuple[str, FactorSpec], ...] = ()  # trials/weights/offset

@dataclass(frozen=True)
class ModelSpec:
    response:    ResponseSpec
    fixed:       tuple[TermSpec, ...]
    partial:     tuple[TermSpec, ...] = ()              # in-model nuisance (noise() on a normal RHS)
    random:      tuple[RandomEffectSpec, ...] = ()
    smooth:      tuple[SmoothSpec, ...] = ()
    family:      FamilySpec = FamilySpec()
    errors:      ErrorSpec | None = None
    residualise: tuple[ResidualiseSpec, ...] = ()
    estimands:   tuple[ContrastSpec, ...] = ()
    estimation:  EstimationSpec = EstimationSpec()
    inference:   InferenceSpec | None = None            # node-level
    covariates:  CovariateProgram = ()

# --- multi-level ---
@dataclass(frozen=True) class Carry: contrast: str; quantities: Literal['cope_varcope','estimates']   # binds to an upstream ContrastSpec.name
@dataclass(frozen=True) class Edge:  source: str; dest: str; filter: tuple[tuple[str, str], ...]; carry: Carry
@dataclass(frozen=True) class ModelNode: name: str; level: Level; group_by: tuple[str, ...]; combine: Combine; spec: ModelSpec
@dataclass(frozen=True)
class ModelGraph:  nodes: tuple[ModelNode, ...]; edges: tuple[Edge, ...] = ()
                   inference: InferenceSpec | None = None      # graph-level (distinct from per-node)

# --- diagnostics (validation output) ---
@dataclass(frozen=True) class Diagnostic: severity: Severity; code: str; message: str; where: str

# --- the ONE open seam, engine-owned ---
class Lowerable(Protocol): ...        # implemented engine-side; nwx never implements it
```

A single-node formula lowers to a one-node `ModelGraph`. The
*populated-fields → solver* dispatch is the **engine's** responsibility and is
specified in §9, not here (the IR describes the model, never which `nitrix`
function fires).

### §6. The `CovariateProgram` (harvested confound vocabulary)

`nwx` emits — never executes (decision 4) — a `CovariateProgram = tuple[
CovariateOp, ...]` (`CovariateOp` a **closed union**). The source of truth is
`grammars/dataframe/` (the `ply`-engine port of the confound vocabulary),
**not** the pre-`ply` `dfops.py`.

**Wilkinson precedence (governing rule).** `nwx` is a Wilkinson extension first:
where a glyph is expressible in *both* the Wilkinson term algebra and the
confound vocabulary, **the Wilkinson meaning wins** (`^` = crossing-power not
numeric power; `-` = term removal not range; `||` = uncorrelated random effects
not set-union). The confound layer contributes only its non-colliding operators
(`^^`, `d_`/`dd_`, `v_`, `n_`, `I_`, `AND_`/`OR_`/`NOT_`, `:::`, …); colliding
needs are redirected (numeric power → `^^`; ranges → parameter block; set-union
→ `OR_`). Full table: `docs/nwx/covariate-vocabulary.md`. Operators:

| Surface (dataframe) | `CovariateOp` | Notes |
|---|---|---|
| `rps`, `wm`, `gsr`, `acc`, `fd`, `dv`, `wcc`, `ccc` | `Shorthand` | preprocessor expansion (`csf` is **not** a shorthand — passthrough column) |
| `d_`, `dd_` | `Derivative(order, inclusive)` | backward difference |
| `^^` | `Power(order, inclusive)` | inclusive power |
| `v_` | `CompCorSelect(criterion, value)` | cumulative-variance selection (params via `{{…}}`) |
| `I_` + bracketed comparison | `Indicator(comparison, threshold)` | spike regressors |
| `AND_` / `OR_` / `NOT_` | `SetOp(kind)` | set reductions |
| `:::` | `Scatter()` | per-timepoint scatter |

`nwx` validates references + operator arity; it holds no array.

### §7. Device → IR → `nitrix` map (with backend status)

Aligned to the revised nitrix v3 FR (the `lme_fit` structure-dispatch ladder).

| Device | IR node | `nitrix` target | Status |
|---|---|---|---|
| `+ - * : / ^` term algebra | `TermSpec` | engine-assembled design | ✅ |
| `(1\|g)` | `RandomEffectSpec(SCALAR)` | `reml_fit` (R1, shipped) | ✅ kernel |
| `(1+x\|g)`, `(x\|\|g)` | `RandomEffectSpec(UNSTRUCTURED/DIAGONAL)` | `lme_fit` (R2) | ⚠️ v3 §1.1 |
| `(1\|g1/g2)`, crossed | multiple `RandomEffectSpec` | `lme_fit` (R3/R4) | ⚠️ v3 §1.1 |
| `s/te/ti` (plain) | `SmoothSpec` | `*_basis` + `gam_fit` | ✅ (ps/cc/tp/te) |
| `s(x, by=f)` factor smooth | `SmoothSpec(by_kind='factor')` | factor-smooth basis | ⚠️ v3 §3.1 |
| `s(g, bs="re"/"fs")` | `SmoothSpec(basis=RE/FS)` | `re_basis`/`gam_fit` | ⚠️ v3 §2 (GAMM) |
| `{{ family=… }}` (Gaussian/Binomial/Poisson) | `FamilySpec` | `Family` | ✅ |
| `{{ family=gamma/nb/… }}` | `FamilySpec` | — | ⚠️ v3 §4 |
| `{{ correlation=ar1(…) }}` | `ErrorSpec` | — | ⚠️ v3 §1.4 |
| `{{ se=robust/cluster }}` | `EstimationSpec` | — | ⚠️ v3 §6.2 |
| `{{ dof=satterthwaite/kr }}` (mixed only) | `EstimationSpec.dof` | — | ⚠️ v3 §1.3 |
| `~\|` (aggressive) | `ResidualiseSpec(AGGRESSIVE)` | `linalg.residualise` | ✅ |
| `~\|` + `nonaggressive` | `ResidualiseSpec(NONAGGRESSIVE)` | `partial_residualise` | ⚠️ v3 §5.1 |
| `{{ contrasts: … }}` — GLM | `ContrastSpec` | `t_contrast`/`f_contrast` | ✅ |
| `{{ contrasts: … }}` — LME | `ContrastSpec` | LME contrast + dof | ⚠️ v3 §1.3 |
| `{{ inference=permutation(tfce/cluster) }}` | `InferenceSpec` | `permutation_test`/`fdr_bh` | ✅ |
| `{{ inference=parametric(rft) }}` | `InferenceSpec(correction='rft')` | — | ⚠️ v3 §7 |
| `[A] >> [B]` graph | `ModelGraph` | per-node + `flame_two_level` | ✅ kernels; engine orchestrates |

### §8. Static validation (in `nwx`, before any data)

- Rank / identifiability (duplicate terms; port the PoC structural-singularity
  guard); intercept consistency across all nodes/frames.
- Random effects: grouping factor categorical-by-declaration; nesting/crossing
  well-formed; `structure` consistent with the bar form.
- Smooths: factor `by=` requires the parametric main effect (warn if absent).
- Residualisation: non-aggressive requires a non-empty `signal` set (**error**);
  aggressive + `signal()` (warning).
- Reserved-name shadowing: `s`/`te`/`signal`/`noise`/`ar1`/… are special **only
  in call position**; a bare `signal` is a lookup, a column literally named `s`
  is legal — tested both ways.
- Directive scoping violations (parse-time, located via `GrammarErrorHandler`).
- Multi-level: each stage has a `level` and `combine`; `group_by` references
  resolve; edges form a DAG; each `Edge.carry.contrast` names a real upstream
  `ContrastSpec`.
- **Backend awareness**: directives/IR features whose nitrix kernel is not yet
  shipped (per §7 "⚠️ v3") emit a `WARNING` citing the FR item, so specs stay
  forward-compatible.
- `.` disambiguation: `VARIABLE_COMPLEMENT` means "complement set" on a node's
  RHS but "inbound cope" on a downstream stage LHS — the interpreter resolves by
  position; tested both ways.

### §9. Engine contract (informative — not built in `gramform`)

The engine (a separate consumer) receives `(ModelGraph, covariate_df,
imaging_data)` and:

1. materialises the `CovariateProgram` into covariate columns;
2. assembles per-node fixed design `X`, random design `Z_g` (per grouping term),
   smooth bases, and the `partial` nuisance block;
3. **dispatches structurally** to the cheapest exact `nitrix` routine — this
   table lives here, not in the IR:

   | Populated | Engine route |
   |---|---|
   | only `fixed` | `glm_fit` (family) + `t/f_contrast` |
   | `partial` present | residualise/partial then `glm_fit` (FWL coefficient) |
   | `random` present | `lme_fit` structure-dispatch (R1 `reml_fit` for `scalar`; R2–R4 otherwise) — nitrix v3 §1.1 |
   | `smooth` present | `gam_fit` (+ `re`/`fs` blocks → GAMM) |
   | `DATASET` node fed by copes | `flame_two_level` |

4. **propagates `{cope, varcope}` along `Edge`s.** FLAME requires the *known*
   within-level variance: the engine binds the upstream `ContrastSpec`'s varcope
   (its `c'(XᵀX)⁻¹c·φ`) to `flame_two_level(var_within=…)`. A three-level
   RUN→SUBJECT→DATASET design is **two chained two-level FLAME fits** with
   varcope propagation through the middle (nitrix ships only `flame_two_level`).
   `ModelNode.combine` selects fixed-effects (precision-weighted, no between
   variance — typical run→subject) vs mixed-effects (estimate σ_b² —
   subject→dataset);
5. runs the node-level and graph-level `InferenceSpec`.

`nwx` guarantees a complete, validated IR; the engine guarantees the numerics
and the no-regression dispatch (nitrix v3 §0.1).

### §10. Scope

**In scope (v1 guarantee):** GLM, GAM, GAMM, LMM/GLMM, residualisation
(aggressive default; non-aggressive opt-in), multi-level node graph; mass-
univariate *and* single-fit (the mass axis is engine data-binding).

**Roadmap:** a **read-direction BIDS Stats Models importer** (`model.json` →
`ModelGraph`; the IR is already node+edge shaped) plus a documented BIDS-SM ↔
`nwx` mapping table; export is a follow-up.

**Reserved (IR-ready, not promised):** multivariate response (`mvbind`),
distributional / location-scale parts (`sigma ~ …`), soft/shrunk residualisation
(`Mode.SOFT`), the exotic smooth bases (`cr/gp/mrf`), robust SEs and mixed-model
dof (all gated by nitrix v3).

**Out of scope:** connectivity / predictive (CBPM) / multivariate-decomposition
methods (CCA/PLS/MDMR) — these are *estimator-shaped*, not *formula-shaped*: a
Wilkinson surface adds no leverage over a direct API, so they belong elsewhere
in the ecosystem. Also out: array numerics, solver selection, the imaging mass
axis, and CLI / non-BIDS file-format parsing.

### §11. Worked examples

**Simplest first (the models most users write):**

```
y ~ 1                                    # one-sample group map (intercept)
y ~ group {{ contrasts: grp = group (t) }}   # two-sample
```

**A small Rosetta table (R ↔ nwx):**

| Intent | R (lme4/mgcv) | nwx |
|---|---|---|
| random intercept | `y ~ x + (1\|g)` | `y ~ x + (1\|g)` |
| correlated slope | `y ~ x + (1+x\|g)` | `y ~ x + (1+x\|g)` |
| smooth of age | `y ~ s(age, k=6)` | `y ~ s(age, k=6)` |
| factor-smooth | `y ~ s(age, by=dx) + dx` | `y ~ s(age, by=dx) + dx` |
| binomial GLM | `glm(y ~ x, binomial)` | `y ~ x {{ family=binomial }}` |

**Showcase (forward-looking; `by=`/RE/AR1 gated by nitrix v3):**

```
# 1. Vertexwise group GAMM (s(by=) and (1|site) depend on v3 kernels)
thk ~ s(age, k=6, by=dx) + dx + sex + noise(meanFD) + (1 | site)
      {{ family=gaussian; estimator=reml;
         contrasts: age_by_dx = s(age):dx (F), dx_main = dx (t);
         inference=permutation(tfce, n=5000) }}

# 2. Aggressive confound cleaning then longitudinal LMM with AR(1)
[ bold ~| rps + wm + csf ] ~ task + (1 + task | subject)
  {{ estimator=reml; correlation=ar1(session | subject); dof=satterthwaite }}

# 2b. Non-aggressive (AROMA) cleaning — explicit opt-in, requires signal()
bold ~| noise(aroma_ic_01 + aroma_ic_02) + signal(task + drift)
     {{ residualise=nonaggressive }}

# 3. Multi-level run→subject→dataset (two chained FLAME fits)
[ cope ~ cond {{ level=subject; group_by=subject; combine=fixed }} ]
  >> [ . ~ 1 {{ level=dataset; combine=mixed; estimator=flame; contrasts: grp = 1 (t) }} ]
     {{ inference=permutation(cluster_mass, n=5000) }}
```

**Validator-catches (onboarding):**

```
3:x + 2:x                 # ERROR: structural singularity (term scaled twice)
bold ~| n {{ residualise=nonaggressive }}   # ERROR: non-aggressive needs signal()
y ~ s(age, by=dx)         # WARNING: factor by= without the dx main effect
```

### §12. Module layout

```
grammars/nwx/
  grammar.py       # RanefComponent, DirectiveComponent (exclusive spec state),
                   #   PipelineComponent, ResidualComponent  (+ reuse wilkinson, dataframe)
  spec.py          # the ModelSpec IR (frozen dataclasses, closed unions, Diagnostic/Carry)
  covariate.py     # CovariateProgram (closed-union CovariateOp; extends dataframe)
  transform.py          # core term-algebra AST→IR ops (Phase 1)
  transform_ranef.py    # random-effects ops          (Phase 3)
  transform_smooth.py   # smooth ops                  (Phase 4)
  transform_struct.py   # residualise / pipeline ops  (Phase 5)
  directives.py    # directive mini-parser → Family/Estimation/Error/Inference/Contrast
  validate.py      # static checks (§8) → tuple[Diagnostic, ...]
```

The interpreter is split by feature family (each module registers into the
shared `InterpretersDispatch` group) so Phases 3/4/5 touch disjoint files. A new
`spec` interpreter is registered alongside the existing `formulaic` one.

### §13. Cross-references

- Implementation plan: `docs/nwx/implementation-plan.md`.
- nitrix kernels this lowers onto:
  `nitrix/docs/feature-requests/stats-modelling-suite-v3.md` (the `lme_fit`
  structure-dispatch ladder, GAMM surfacing, non-aggressive residualisation,
  dof, families) and predecessors v1/v2.
- Substrate: `src/gramform/core.py`, `src/gramform/grammars/wilkinson/`,
  `src/gramform/grammars/dataframe/`, `src/gramform/postprocessors.py`.
