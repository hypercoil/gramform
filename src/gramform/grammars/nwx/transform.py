# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
``nwx`` term-algebra interpreter (Phase 1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Maps the Wilkinson AST (built by the existing :class:`WilkinsonGrammar`) onto
the typed :mod:`gramform.grammars.nwx.spec` IR -- a ``ModelGraph`` -- rather
than ``formulaic`` objects. A new ``spec`` interpreter is registered alongside
the PoC's ``formulaic`` one; this module imports neither ``jax`` nor
``nitrix`` (the firewall test asserts it).

The term algebra (``+ - * : / ^`` and grouping), the ``~`` / ``~|`` structure,
and bracketed frames ``[ ... ]`` reuse the Wilkinson grammar verbatim; only the
*interpretation* differs. The PoC's ``to_terms`` dedup / structural-singularity
guard is ported onto ``TermSpec``, and an nwx-aware intercept postprocessor
(:func:`nwx_add_intercept`) descends past the directive wrappers. A trailing
``{{ ... }}`` directive block is captured by the grammar as a single
``DIRECTIVE_BLOCK`` token (its text parsed by :mod:`directives`) and attaches by
*position*: ``PROGRAM_DIRECTIVES`` (outermost node) or ``FRAME_DIRECTIVES``
(inside a frame). No separate lexer state is needed -- the block is opaque to
the term lexer, so the directive ``:`` / ``=`` never collide with the term
algebra's.

Results flow through a single typed slot (``NwxState.eval``) as a
``FactorSpec | TermSpec | tuple[TermSpec, ...] | _Block`` union -- no
``operational_level`` bookkeeping. Frame sub-models accumulate in
``NwxState.deps`` and become extra graph nodes; the directive set is stashed in
``NwxState.directives`` for the finaliser.
"""

import ast as _ast
import dataclasses
import warnings
from dataclasses import dataclass
from typing import Callable, Iterable, Type

from gramform.core import (
    ExecutionContext,
    InterpretersDispatch,
    Primitive,
    TransformProcessor,
    TypedState,
    withCacheSubcontext,
)
from gramform.grammars.nwx.directives import DirectiveSet, parse_directives
from gramform.grammars.nwx.grammar import NwxGrammar
from gramform.grammars.nwx.spec import (
    Combine,
    Const,
    FactorSpec,
    Level,
    Lookup,
    Mode,
    ModelGraph,
    ModelNode,
    ModelSpec,
    PyExpr,
    RandomEffectSpec,
    Referent,
    ResidualiseSpec,
    ResponseSpec,
    SmoothSpec,
    TermSource,
    TermSpec,
)
from gramform.grammars.wilkinson.transform import add_intercept_to_formula
from gramform.postprocessors import (
    ppr_associative_flatten,
    ppr_common_subexpression,
)

INTERPRETERS: InterpretersDispatch = InterpretersDispatch()

#: The canonical intercept / unit term and the zero (suppress-intercept) term.
ONE: TermSpec = TermSpec(factors=(FactorSpec(Const(1.0)),))
ZERO: TermSpec = TermSpec(factors=(FactorSpec(Const(0.0)),))


class NwxError(ValueError):
    """A structural error in an ``nwx`` formula (e.g. a singularity)."""


# ---------------------------------------------------------------------------
# interpreter plumbing: a single typed result slot + accumulators
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Block:
    """An interpreted formula body, before model assembly. ``lhs`` is the
    response (``~``) or the residualisation target (``~|``); ``rhs`` is the
    design (or the noise set, when ``residualise``); ``partial`` holds in-model
    nuisance terms routed from ``noise()`` on a normal RHS."""

    lhs: tuple[TermSpec, ...]
    rhs: tuple[TermSpec, ...]
    residualise: bool = False
    partial: tuple[TermSpec, ...] = ()
    random: tuple[RandomEffectSpec, ...] = ()
    smooth: tuple[SmoothSpec, ...] = ()
    signal: tuple[TermSpec, ...] = ()  # signal() set on a `~|` RHS


class NwxState(TypedState):
    """Typed result slot (``eval``) plus parse-wide accumulators. ``deps``
    collects frame sub-models (emitted as extra graph nodes); ``partial``
    accumulates ``noise()`` in-model nuisance terms for the enclosing block;
    ``random`` accumulates ``(...|g)`` random-effect specs and ``smooth`` the
    ``s()``/``te()``/... smooth specs for the enclosing block. On a ``~|``
    residualisation RHS, ``in_residualise`` is set so ``signal()``/``noise()``
    route into ``signal`` / ``residualise_noise`` instead of ``partial``;
    ``directives`` holds the parsed ``{{ ... }}`` block for the finaliser."""

    deps: tuple[ModelNode, ...] = ()
    partial: tuple[TermSpec, ...] = ()
    random: tuple[RandomEffectSpec, ...] = ()
    smooth: tuple[SmoothSpec, ...] = ()
    signal: tuple[TermSpec, ...] = ()
    residualise_noise: tuple[TermSpec, ...] = ()
    in_residualise: bool = False
    directives: DirectiveSet | None = None


class NwxContext(ExecutionContext, withCacheSubcontext):
    """Execution context for the ``spec`` interpreter."""

    __state__: Type[NwxState] = NwxState


# ---------------------------------------------------------------------------
# term-algebra helpers (ported from the PoC `to_term`/`to_terms`)
# ---------------------------------------------------------------------------


def _factor_key(factor: FactorSpec) -> tuple[str, ...]:
    """A stable sort key for a non-constant factor's variable signature."""
    source = factor.source
    match source:
        case Lookup(name=name):
            return ('1lookup', name)
        case PyExpr(code=code):
            return ('2pyexpr', code)
        case Referent(stage=stage, kind=kind):
            return ('3referent', stage, kind)
        case Const(value=value):
            return ('0const', repr(value))
        case _:
            return ('4other', repr(source))


def to_term(factors: Iterable[FactorSpec]) -> TermSpec:
    """Build a term from factors, collapsing literal factors into a single
    constant (their product, placed first) and deduplicating variable factors
    -- a Wilkinson term is a *set* of factors, so ``x:x`` is ``x`` (PoC
    parity)."""
    factors = tuple(factors)
    consts = [f for f in factors if isinstance(f.source, Const)]
    # Deduplicate and canonicalise factor order (interaction is commutative,
    # so `dog:rat` and `rat:dog` are the same term).
    variables = sorted(
        dict.fromkeys(f for f in factors if not isinstance(f.source, Const)),
        key=_factor_key,
    )
    if consts:
        product = 1.0
        for f in consts:
            assert isinstance(f.source, Const)
            product *= f.source.value
        return TermSpec((FactorSpec(Const(product)), *variables))
    return TermSpec(tuple(variables))


def to_terms(terms: Iterable[TermSpec]) -> tuple[TermSpec, ...]:
    """Deduplicate a term set, preserving order, and reject structural
    singularities (the same variable signature scaled by two constants), per
    the PoC. Identical terms collapse first; only a *rescaled* duplicate is an
    error. A bare constant other than the unit intercept is dropped (warning).
    """
    ordered = list(
        dict.fromkeys(terms)
    )  # collapse identical terms, keep order
    scales: dict[tuple[FactorSpec, ...], float] = {}
    keys: dict[tuple[FactorSpec, ...], list[TermSpec]] = {}
    to_remove: list[TermSpec] = []
    for term in ordered:
        consts = [
            f.source.value for f in term.factors if isinstance(f.source, Const)
        ]
        variables = tuple(
            sorted(
                (f for f in term.factors if not isinstance(f.source, Const)),
                key=_factor_key,
            )
        )
        scaled = scales.get(variables)
        if consts:
            scale = 1.0
            for c in consts:
                scale *= c
            if not variables and scale != 1:
                warnings.warn(
                    f'Constant term {term} was interpreted as a scale of '
                    f'{scale}, but only an intercept constant (scale of 1) is '
                    'allowed. This term has been removed from the formula '
                    'automatically.'
                )
                instances = [term]
            elif scaled is not None:
                raise NwxError(
                    f'Attempting to scale {TermSpec(variables)} by {scale}, '
                    f'but it has already been scaled by {scaled}'
                )
            else:
                scales[variables] = scale
                instances = keys.get(variables, [])
            to_remove.extend(instances)
        keys[variables] = keys.get(variables, []) + [term]
    return tuple(t for t in ordered if t not in to_remove)


def _coerce(value: object) -> list[TermSpec]:
    """Normalise an interpreter result into a list of terms."""
    if isinstance(value, FactorSpec):
        return [to_term((value,))]
    if isinstance(value, TermSpec):
        return [value]
    if isinstance(value, tuple):
        return list(value)
    raise NwxError(f'cannot coerce {type(value).__name__} to terms')


def _factor_seqs(value: object) -> dict[tuple[FactorSpec, ...], None]:
    """Normalise a result into a set of factor sequences (for products)."""
    seqs: dict[tuple[FactorSpec, ...], None] = {}
    if isinstance(value, FactorSpec):
        seqs[(value,)] = None
    elif isinstance(value, TermSpec):
        seqs[tuple(value.factors)] = None
    elif isinstance(value, tuple):
        for term in value:
            seqs[tuple(term.factors)] = None
    else:
        raise NwxError(f'cannot coerce {type(value).__name__} to factors')
    return seqs


def remove_terms(orig: object, remove: object) -> tuple[TermSpec, ...]:
    """Remove ``remove`` terms from ``orig``. Removing the zero term re-adds
    the intercept; identical terms are compared by value. Coercion is done
    without the singularity pass so a bare zero survives to drive intercept
    re-insertion (PoC parity)."""
    kept = list(dict.fromkeys(_coerce(orig)))
    removed = set(_coerce(remove))
    if ZERO in removed:
        removed.discard(ZERO)
        if ONE not in kept:
            kept.append(ONE)
    return tuple(t for t in kept if t not in removed)


def standardise_code(code: str) -> str:
    """Normalise inline Python (the ``{...}`` execute form)."""
    return _ast.unparse(_ast.parse(code, mode='eval')).replace('\n', ' ')


def _render(source: TermSource) -> str:
    match source:
        case Lookup(name=name):
            return name
        case Const(value=value):
            return str(int(value)) if value.is_integer() else str(value)
        case PyExpr(code=code):
            return code
        case Referent(stage=stage, kind=kind):
            return f'{stage}{kind}'
        case _:
            return repr(source)


# ---------------------------------------------------------------------------
# term-algebra operations
# ---------------------------------------------------------------------------


def VARIABLE_impl(node: Primitive, context: NwxContext) -> NwxContext:
    name = node.get_parameters()
    return context.with_result(FactorSpec(Lookup(name)))


def NUMERIC_LITERAL_impl(node: Primitive, context: NwxContext) -> NwxContext:
    literal = node.get_parameters()
    return context.with_result(FactorSpec(Const(float(literal.value))))


def EXECUTE_impl(node: Primitive, context: NwxContext) -> NwxContext:
    code = standardise_code(node.get_parameters())
    return context.with_result(FactorSpec(PyExpr(code)))


def VARIABLE_COMPLEMENT_impl(
    node: Primitive,
    context: NwxContext,
) -> NwxContext:
    # `.` on a RHS is the complement set; resolved by the validator/engine
    # later. Carried here as a plain lookup of the reserved name.
    return context.with_result((TermSpec((FactorSpec(Lookup('.')),)),))


def APPEND_impl(node: Primitive, context: NwxContext) -> NwxContext:
    all_terms: dict[TermSpec, None] = {}
    for child in node.parameters:
        context = child(context)
        result = context.get_result()
        if getattr(child, 'name', None) == 'UNARY_NEGATION':
            kept = remove_terms(tuple(all_terms), result)
            all_terms = dict.fromkeys(kept)
            continue
        # Coerce without the singularity pass so a bare zero survives to drive
        # intercept suppression; the structural-singularity guard runs on the
        # accumulated set below (PoC parity).
        update = _coerce(result)
        for term in update:
            all_terms[term] = None
        if ZERO in update:
            all_terms.pop(ZERO, None)
            all_terms.pop(ONE, None)
    return context.with_result(to_terms(tuple(all_terms)))


def REMOVE_impl(node: Primitive, context: NwxContext) -> NwxContext:
    expr, removal = node.parameters
    context = expr(context)
    orig = context.get_result()
    context = removal(context)
    rem = context.get_result()
    return context.with_result(remove_terms(orig, rem))


def UNARY_NEGATION_impl(node: Primitive, context: NwxContext) -> NwxContext:
    child = node.get_parameters()
    context = child(context)
    return context.with_result(to_terms(_coerce(context.get_result())))


def INTERACTION_impl(node: Primitive, context: NwxContext) -> NwxContext:
    children = node.parameters
    context = children[0](context)
    seqs = _factor_seqs(context.get_result())
    for nxt in children[1:]:
        context = nxt(context)
        new = _factor_seqs(context.get_result())
        seqs = {a + b: None for a in seqs for b in new}
    terms = [to_term(seq) for seq in seqs]
    return context.with_result(to_terms(terms))


def NESTED_impl(node: Primitive, context: NwxContext) -> NwxContext:
    left, right = node.parameters
    context = left(context)
    left_seqs = _factor_seqs(context.get_result())
    context = right(context)
    right_seqs = _factor_seqs(context.get_result())
    left_reduced: tuple[FactorSpec, ...] = sum(left_seqs.keys(), ())
    terms = [to_term(a) for a in left_seqs]
    terms += [to_term(left_reduced + b) for b in right_seqs]
    return context.with_result(to_terms(terms))


#: Nuisance markers (special only in call position): on a normal RHS they mark
#: in-model FWL partials; on a ``~|`` residualisation RHS they mark the noise
#: set. ``signal()`` marks variance to preserve (residualisation RHS only).
_PARTIAL_MARKERS = frozenset({'noise', 'nuisance'})
_SIGNAL_MARKERS = frozenset({'signal'})

#: Extra ``NAME(...)`` call handlers contributed by feature-family modules
#: (smooths register ``s``/``te``/``ti``/``t2`` here at import). A handler
#: takes the ``NAMED_FUNCTION`` node + context and returns a context whose
#: result is the call's term contribution (``()`` when it routes elsewhere).
NAMED_FUNCTION_HANDLERS: dict[
    str, Callable[[Primitive, 'NwxContext'], 'NwxContext']
] = {}


def NAMED_FUNCTION_impl(node: Primitive, context: NwxContext) -> NwxContext:
    name = node.parameters[0]
    if name in _PARTIAL_MARKERS:
        # `noise(...)` routes structurally and contributes no fixed term. On a
        # residualisation (`~|`) RHS it joins the noise set; on a normal RHS it
        # marks an in-model FWL partial (`ModelSpec.partial`).
        expr = node.parameters[1]
        context = expr(context)
        terms = to_terms(_coerce(context.get_result()))
        if context.state.in_residualise:
            context = context.update_state(
                residualise_noise=context.state.residualise_noise + terms
            )
        else:
            context = context.update_state(
                partial=context.state.partial + terms
            )
        return context.with_result(())
    if name in _SIGNAL_MARKERS:
        # `signal(...)` marks variance to preserve under non-aggressive
        # residualisation; it is meaningful only on a `~|` RHS.
        if not context.state.in_residualise:
            raise NwxError(
                'signal(...) is only valid on a residualisation (~|) RHS'
            )
        expr = node.parameters[1]
        context = expr(context)
        terms = to_terms(_coerce(context.get_result()))
        context = context.update_state(signal=context.state.signal + terms)
        return context.with_result(())
    handler = NAMED_FUNCTION_HANDLERS.get(name)
    if handler is not None:
        return handler(node, context)
    # Data-transform functions (plain `bs`/`ns`/`poly`) arrive later.
    raise NotImplementedError(
        f'function-call terms ({name}(...)) are not supported yet'
    )


# ---------------------------------------------------------------------------
# structure: ~, ~|, frames
# ---------------------------------------------------------------------------


def LHS_RHS_STRUCTURE_impl(node: Primitive, context: NwxContext) -> NwxContext:
    lhs_expr, rhs_expr = node.parameters
    context = lhs_expr(context)
    lhs = to_terms(_coerce(context.get_result()))
    context = rhs_expr(context)
    rhs = to_terms(_coerce(context.get_result()))
    # Consume any noise() partials, (...|g) random effects, and s()/te()/...
    # smooths accumulated while evaluating this RHS, so they bind to this block
    # and do not leak to an enclosing one.
    partial = context.state.partial
    random = context.state.random
    smooth = context.state.smooth
    context = context.update_state(partial=(), random=(), smooth=())
    return context.with_result(
        _Block(
            lhs=lhs,
            rhs=rhs,
            partial=partial,
            random=random,
            smooth=smooth,
        )
    )


def RESIDUAL_STRUCTURE_impl(
    node: Primitive,
    context: NwxContext,
) -> NwxContext:
    target_expr, noise_expr = node.parameters
    context = target_expr(context)
    target = to_terms(_coerce(context.get_result()))
    # In residualisation context, signal()/noise() route to the signal/noise
    # sets; unwrapped terms (incl. the de-meaning intercept) are noise.
    context = context.update_state(in_residualise=True)
    context = noise_expr(context)
    unwrapped = to_terms(_coerce(context.get_result()))
    signal = context.state.signal
    extra_noise = context.state.residualise_noise
    context = context.update_state(
        in_residualise=False, signal=(), residualise_noise=()
    )
    noise = to_terms(unwrapped + extra_noise)
    return context.with_result(
        _Block(lhs=target, rhs=noise, residualise=True, signal=signal)
    )


def SUBPARTS_STRUCTURE_impl(
    node: Primitive,
    context: NwxContext,
) -> NwxContext:
    # Top-level `|` (multi-part RHS) is restricted to a single part in v1; the
    # reserved distributional / mvbind forms are gated (spec §4.4).
    raise NwxError(
        'top-level `|` (multi-part RHS) is restricted to a single part in v1'
    )


def _push_frame(
    inner: Primitive,
    context: NwxContext,
    directives: DirectiveSet | None = None,
) -> NwxContext:
    """Evaluate a bracketed sub-model into a graph sub-node and inject its
    fitted (``_hat``) / residualised (``_tilde``) referent into the parent
    design. ``directives`` (a frame-level ``{{ ... }}``) configure the
    sub-node's spec + level/group_by/combine."""
    context = inner(context)
    block = context.get_result()
    if not isinstance(block, _Block):
        # A bare bracketed expression `[expr]` with no `~`: pass its terms up.
        return context.with_result(block)
    deps = context.state.deps
    stage = f'frame{len(deps)}'
    kind = '_tilde' if block.residualise else '_hat'
    spec = _block_to_modelspec(block)
    level, group_by, combine = Level.DATASET, (), Combine.FIXED
    if directives is not None:
        spec = _apply_directives(spec, directives)
        _validate_residualise(spec.residualise)
        level = directives.level or level
        group_by = directives.group_by
        combine = directives.combine or combine
    subnode = ModelNode(
        name=stage,
        level=level,
        group_by=group_by,
        combine=combine,
        spec=spec,
    )
    referents = tuple(
        TermSpec((FactorSpec(Referent(stage=stage, kind=kind)),))
        for _ in block.lhs
    )
    context = context.update_state(deps=deps + (subnode,))
    return context.with_result(to_terms(referents))


def PUSH_FRAME_impl(node: Primitive, context: NwxContext) -> NwxContext:
    return _push_frame(node.get_parameters(), context)


def FRAME_DIRECTIVES_impl(node: Primitive, context: NwxContext) -> NwxContext:
    inner, block = node.parameters
    return _push_frame(inner, context, parse_directives(_block_text(block)))


def PROGRAM_DIRECTIVES_impl(
    node: Primitive,
    context: NwxContext,
) -> NwxContext:
    # A trailing `{{ ... }}` on the whole formula binds to the implicit
    # outermost node: stash its directives for the finaliser, then evaluate the
    # formula body.
    formula, block = node.parameters
    context = context.update_state(
        directives=parse_directives(_block_text(block))
    )
    return formula(context)


def _block_text(token: str) -> str:
    """Strip the ``{{`` / ``}}`` delimiters off a ``DIRECTIVE_BLOCK`` token."""
    return token[2:-2]


# ---------------------------------------------------------------------------
# model assembly
# ---------------------------------------------------------------------------


def _block_to_modelspec(block: _Block) -> ModelSpec:
    response = ResponseSpec(terms=block.lhs)
    if block.residualise:
        return ModelSpec(
            response=response,
            fixed=(),
            residualise=(
                ResidualiseSpec(
                    target=block.lhs,
                    noise=block.rhs,
                    signal=block.signal,
                    mode=Mode.AGGRESSIVE,
                ),
            ),
        )
    return ModelSpec(
        response=response,
        fixed=block.rhs,
        partial=block.partial,
        random=block.random,
        smooth=block.smooth,
    )


def _apply_directives(
    spec: ModelSpec,
    directives: DirectiveSet | None,
) -> ModelSpec:
    if directives is None:
        return spec
    updates: dict[str, object] = {}
    if directives.family is not None:
        updates['family'] = directives.family
    if directives.estimation is not None:
        updates['estimation'] = directives.estimation
    if directives.errors is not None:
        updates['errors'] = directives.errors
    if directives.estimands:
        updates['estimands'] = directives.estimands
    if directives.inference is not None:
        updates['inference'] = directives.inference
    if directives.residualise is not None:
        if spec.residualise:
            updates['residualise'] = tuple(
                dataclasses.replace(r, mode=directives.residualise)
                for r in spec.residualise
            )
        else:
            warnings.warn(
                'a `residualise=` directive was given but the formula has no '
                '`~|` residualisation; the directive is ignored'
            )
    return dataclasses.replace(spec, **updates) if updates else spec


def _validate_residualise(specs: tuple[ResidualiseSpec, ...]) -> None:
    """Enforce the residualisation-mode rules (spec §4.6 / §8): non-aggressive
    needs a non-empty ``signal`` set (hard error); aggressive ignores any
    ``signal`` set (warning)."""
    for r in specs:
        if r.mode is Mode.NONAGGRESSIVE and not r.signal:
            raise NwxError(
                'residualise=nonaggressive requires a signal() set '
                '(a bare `~|` cannot preserve shared variance); spec §4.6'
            )
        if r.mode is Mode.AGGRESSIVE and r.signal:
            warnings.warn(
                'aggressive residualisation ignores the signal() set; add '
                '`{{ residualise=nonaggressive }}` to preserve shared variance'
            )


def finalise_hook(context: NwxContext) -> NwxContext:
    result = context.get_result()
    directives = context.state.directives
    deps = context.state.deps
    if isinstance(result, _Block):
        spec = _block_to_modelspec(result)
    else:
        spec = ModelSpec(
            response=ResponseSpec(terms=()),
            fixed=to_terms(_coerce(result)),
            partial=context.state.partial,
            random=context.state.random,
            smooth=context.state.smooth,
        )
    spec = _apply_directives(spec, directives)
    _validate_residualise(spec.residualise)
    for dep in deps:
        _validate_residualise(dep.spec.residualise)
    root = ModelNode(
        name='root',
        level=directives.level
        if directives and directives.level
        else (Level.DATASET),
        group_by=directives.group_by if directives else (),
        combine=directives.combine
        if directives and directives.combine
        else (Combine.FIXED),
        spec=spec,
    )
    graph = ModelGraph(nodes=(root, *deps), edges=())
    return context.with_result(graph)


# ---------------------------------------------------------------------------
# interpreter registration + processor
# ---------------------------------------------------------------------------

INTERPRETERS.register_interpreter('spec')
INTERPRETERS.register_group('build', ('spec',))
INTERPRETERS.register_operation('build', 'VARIABLE', VARIABLE_impl)
INTERPRETERS.register_operation(
    'build', 'NUMERIC_LITERAL', NUMERIC_LITERAL_impl
)
INTERPRETERS.register_operation('build', 'EXECUTE', EXECUTE_impl)
INTERPRETERS.register_operation(
    'build', 'VARIABLE_COMPLEMENT', VARIABLE_COMPLEMENT_impl
)
INTERPRETERS.register_operation('build', 'APPEND', APPEND_impl)
INTERPRETERS.register_operation('build', 'REMOVE', REMOVE_impl)
INTERPRETERS.register_operation('build', 'UNARY_NEGATION', UNARY_NEGATION_impl)
INTERPRETERS.register_operation('build', 'INTERACTION', INTERACTION_impl)
INTERPRETERS.register_operation('build', 'NESTED', NESTED_impl)
INTERPRETERS.register_operation('build', 'NAMED_FUNCTION', NAMED_FUNCTION_impl)
INTERPRETERS.register_operation(
    'build', 'LHS_RHS_STRUCTURE', LHS_RHS_STRUCTURE_impl
)
INTERPRETERS.register_operation(
    'build', 'RESIDUAL_STRUCTURE', RESIDUAL_STRUCTURE_impl
)
INTERPRETERS.register_operation(
    'build', 'SUBPARTS_STRUCTURE', SUBPARTS_STRUCTURE_impl
)
INTERPRETERS.register_operation('build', 'PUSH_FRAME', PUSH_FRAME_impl)
INTERPRETERS.register_operation(
    'build', 'FRAME_DIRECTIVES', FRAME_DIRECTIVES_impl
)
INTERPRETERS.register_operation(
    'build', 'PROGRAM_DIRECTIVES', PROGRAM_DIRECTIVES_impl
)

# Register the disjoint feature-family interpreters into the shared ``spec``
# group (spec §12). Imported here, after the dispatch + helpers above are
# defined, so the module-level registrations resolve their imports from this
# module -- an intentional bottom import for its side effect.
import gramform.grammars.nwx.transform_ranef  # noqa: E402, F401
import gramform.grammars.nwx.transform_smooth  # noqa: E402, F401

# ---------------------------------------------------------------------------
# intercept postprocessor (nwx-aware)
# ---------------------------------------------------------------------------


def _walk_intercept(tree: object, context: NwxContext) -> object:
    """Recurse, re-adding the implicit intercept to every nested formula (a
    ``PUSH_FRAME`` / ``FRAME_DIRECTIVES`` sub-model), preserving the opaque
    ``DIRECTIVE_BLOCK`` string operands."""
    if not isinstance(tree, Primitive) or tree.is_terminal:
        return tree
    if tree.name == 'PUSH_FRAME':
        return tree.bind(_top_formula(tree.get_parameters(), context))
    if tree.name == 'FRAME_DIRECTIVES':
        formula, block = tree.parameters
        return tree.bind(_top_formula(formula, context), block)
    return tree.bind(
        *[_walk_intercept(child, context) for child in tree.parameters]
    )


def _top_formula(tree: object, context: NwxContext) -> object:
    """Add the intercept to a formula's RHS, then recurse into its factors."""
    if isinstance(tree, Primitive):
        tree = add_intercept_to_formula(tree, context)[0]
    return _walk_intercept(tree, context)


def nwx_add_intercept(
    tree: Primitive,
    context: NwxContext,
) -> tuple[Primitive, NwxContext]:
    """The nwx counterpart of Wilkinson's ``ppr_add_intercept`` postprocessor.

    The directive wrappers (``PROGRAM_DIRECTIVES`` / ``FRAME_DIRECTIVES``)
    carry an opaque ``{{ ... }}`` string operand that must NOT be wrapped in an
    intercept; this descends past them to the inner formula(e)."""
    if tree.name == 'PROGRAM_DIRECTIVES':
        formula, block = tree.parameters
        return tree.bind(_top_formula(formula, context), block), context
    result = _top_formula(tree, context)
    assert isinstance(result, Primitive)
    return result, context


class NwxProcessor:
    """Formula string -> validated :class:`ModelGraph` IR."""

    def __init__(self) -> None:
        self._inner = TransformProcessor(
            grammar=NwxGrammar(),
            preprocessors=(),
            postprocessors=(
                nwx_add_intercept,
                ppr_associative_flatten,
                ppr_common_subexpression,
            ),
            interpreters=INTERPRETERS,
            context_class=NwxContext,
            default_interpreter='spec',
        )
        self._inner.register_finalisation('spec', finalise_hook)

    def __call__(self, formula: str) -> ModelGraph:
        return self._inner.transform(formula)


def get_processor() -> NwxProcessor:
    return NwxProcessor()
