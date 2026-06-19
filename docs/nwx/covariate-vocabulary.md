# nwx covariate vocabulary — harvested reference

> **Source of truth for nwx Phase 5/6** (the `CovariateProgram`). Transcribed
> from the **`grammars/dataframe/`** grammar (the `ply` engine's port of the
> confound / dataframe vocabulary), which is the authority — **not** the deleted
> pre-`ply` `dfops.py`. Surfaces are grounded in
> `src/gramform/grammars/dataframe/grammar.py`.

## Governing rule: Wilkinson conventions take precedence

`nwx` is a **Wilkinson extension first**. Where an operator glyph is expressible
in *both* the Wilkinson term algebra and the confound vocabulary, **the
Wilkinson meaning wins** — a user who knows R/Wilkinson must never be surprised.
The confound vocabulary contributes only its **non-colliding** operators; any
confound need that would reuse a Wilkinson glyph is redirected to a distinct
surface.

### Wilkinson-reserved glyphs (confound meaning redirected)

| Glyph | nwx (Wilkinson) meaning | Legacy confound use | Redirect for the confound need |
|---|---|---|---|
| `+` | term union (`APPEND`) | column concatenation | none needed — semantics coincide |
| `-` | term removal (`REMOVE`) | numeric range (`1-5`) | ranges via a parameter block / enumeration, never infix `-` |
| `^` | **crossing-power** (`(a+b)^2` → interactions to order 2) | numeric power of a regressor | numeric power via `^^` (below) or a transform fn (`poly`/`{x**2}`) |
| `:` | interaction | — (`:::` scatter is a *distinct* glyph) | n/a |
| `*` `/` `**` `%in%` | crossing / nesting | unused in confound vocab | n/a |
| `\|\|` | **uncorrelated random effects** (inside parens) | boolean set-union | set-union via the `OR_` reduce form, never infix `\|\|` |
| `\|` | parts separator / random-effects bar | — | n/a |

So `(rps + wm)^^2` (inclusive numeric power — confound idiom) is fine because
`^^` is **not** a Wilkinson glyph, but `x^2` means *Wilkinson crossing-power*
(here just `x`), **not** `x` squared.

## Confound-specific operators (no Wilkinson collision — retained)

| Surface | Primitive (dataframe) | Meaning |
|---|---|---|
| `^^` | `POWER_INCLUSIVE` → numeric `POWER(1..n)` | inclusive numeric power (keeps original) |
| `d_` | `BACKDIFF` | backward difference (exclusive) |
| `dd_` | `BACKDIFF_INCLUSIVE` → `BACKDIFF(0..n)` | inclusive backward difference |
| `n_` | `FIRST_N` | take first N components |
| `v_` | `CUMUL_VAR` | cumulative-variance selection (aCompCor by variance) |
| `I_` + bracketed comparison | `INDICATOR` (`= != <> ~= < <= > >=`) | indicator / spike regressors |
| `&&` | `INTERSECTION` | set intersection (mask combination) |
| `!` | `NEGATION` | logical negation |
| `AND_(...)` | `INTERSECTION_REDUCE` | reduce-intersection over a group |
| `OR_(...)` | `UNION_REDUCE` | reduce-union over a group (the set-union surface) |
| `NOT_(...)` | `NEGATION_SURFACE` → `INDICATOR(NEGATION(...))` | negated indicator |
| `:::` | `SCATTER` | per-timepoint scatter (one spike regressor per flagged frame) |

(`&&` for intersection is retained for now since Wilkinson has no `&&`; if a
future Wilkinson convention claims it, it likewise defers — and a reduce form
`AND_` is always available.)

## Shorthand expansions (preprocessor)

From `confound_formula_preprocessor()`. Applied as text before parsing.

| Shorthand | Expansion |
|---|---|
| `wm` | `white_matter` |
| `gsr`, `gs` | `global_signal` |
| `rps` | `trans_x + trans_y + trans_z + rot_x + rot_y + rot_z` |
| `fd` | `framewise_displacement` |
| `dv` | `std_dvars` |
| `acc` | `a_comp_cor` |
| `wcc` | `w_comp_cor` |
| `ccc` | `c_comp_cor` |

**Note:** `csf` is **not** a shorthand — it is a passthrough column name (a
common slip; the legacy README implied otherwise). nwx treats an unrecognised
name as a plain `Lookup`, never an error.

## Parameter blocks

- `{{ key=value; ... }}` — a parameter block in a dedicated lexer state
  (`begin_param`/`end_param` via `push_state_and_return`/`pop_state_and_return`;
  `;` = `ARG_SEP`, `=` = `KV_SEP`). nwx reuses this state mechanism but makes its
  directive state **exclusive** (spec §3/§4.5) to isolate `:`/`=` from the term
  algebra. Numeric **ranges** (the redirected confound `-` use) live here.
- `[ ... ]` — bracketed parameter grouping.

## Mapping to the nwx `CovariateProgram`

Phase 5/6 lowers these to the closed-union `CovariateOp` (spec §6): `Shorthand`,
`Derivative` (`d_`/`dd_`), `Power` (`^^`, numeric), `CompCorSelect` (`v_`),
`Indicator` (`I_`), `SetOp` (`AND_`/`OR_`/`NOT_`), `Scatter` (`:::`). nwx emits
the program; the engine materialises columns (spec decision 4).
