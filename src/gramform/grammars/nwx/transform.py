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
guard and the single ``ppr_add_intercept`` postprocessor are ported onto
``TermSpec``. A trailing ``{{ ... }}`` directive block (the minimal Phase-1 key
set) is split off textually and parsed by :mod:`directives`; the integrated
exclusive-state lexer lands in Phase 4.

Results flow through a single typed slot (``NwxState.eval``) as a
``FactorSpec | TermSpec | tuple[TermSpec, ...] | _Block`` union -- no
``operational_level`` bookkeeping. Frame sub-models accumulate in
``NwxState.deps`` and become extra graph nodes; the directive set is stashed in
``NwxState.directives`` for the finaliser.
"""

import ast as _ast
import dataclasses
import re
import warnings
from dataclasses import dataclass
from typing import Iterable, Type

from gramform.core import (
    ExecutionContext,
    InterpretersDispatch,
    Primitive,
    TransformProcessor,
    TypedState,
    withCacheSubcontext,
)
from gramform.grammars.nwx.directives import DirectiveSet, parse_directives
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
    Referent,
    ResidualiseSpec,
    ResponseSpec,
    TermSource,
    TermSpec,
)
from gramform.grammars.wilkinson.grammar import WilkinsonGrammar
from gramform.grammars.wilkinson.transform import ppr_add_intercept
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


class NwxState(TypedState):
    """Typed result slot (``eval``) plus parse-wide accumulators. ``deps``
    collects frame sub-models (emitted as extra graph nodes); ``partial``
    accumulates ``noise()`` in-model nuisance terms for the enclosing block;
    ``directives`` holds the parsed ``{{ ... }}`` block for the finaliser."""

    deps: tuple[ModelNode, ...] = ()
    partial: tuple[TermSpec, ...] = ()
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


#: In-model nuisance markers (special only in call position). ``signal()`` and
#: residualisation-context ``noise()`` routing arrive in Phase 5.
_PARTIAL_MARKERS = frozenset({'noise', 'nuisance'})


def NAMED_FUNCTION_impl(node: Primitive, context: NwxContext) -> NwxContext:
    name = node.parameters[0]
    if name in _PARTIAL_MARKERS:
        # `noise(...)` on a normal RHS marks in-model nuisance: the wrapped
        # terms are partialled out of the reported estimands but not reported
        # (FWL), so they route structurally to `ModelSpec.partial`, not
        # `fixed`. The call contributes no fixed terms.
        expr = node.parameters[1]
        context = expr(context)
        terms = to_terms(_coerce(context.get_result()))
        context = context.update_state(partial=context.state.partial + terms)
        return context.with_result(())
    # Smooths (`s`/`te`/...) and data-transform functions arrive in Phase 4.
    raise NotImplementedError(
        f'function-call terms ({name}(...)) are not supported until Phase 4'
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
    # Consume any noise() partials accumulated while evaluating this RHS, so
    # they bind to this block and do not leak to an enclosing one.
    partial = context.state.partial
    context = context.update_state(partial=())
    return context.with_result(_Block(lhs=lhs, rhs=rhs, partial=partial))


def RESIDUAL_STRUCTURE_impl(
    node: Primitive,
    context: NwxContext,
) -> NwxContext:
    target_expr, noise_expr = node.parameters
    context = target_expr(context)
    target = to_terms(_coerce(context.get_result()))
    context = noise_expr(context)
    noise = to_terms(_coerce(context.get_result()))
    return context.with_result(_Block(lhs=target, rhs=noise, residualise=True))


def SUBPARTS_STRUCTURE_impl(
    node: Primitive,
    context: NwxContext,
) -> NwxContext:
    # Top-level `|` (multi-part RHS) is restricted to a single part in v1; the
    # reserved distributional / mvbind forms are gated (spec §4.4).
    raise NwxError(
        'top-level `|` (multi-part RHS) is restricted to a single part in v1'
    )


def PUSH_FRAME_impl(node: Primitive, context: NwxContext) -> NwxContext:
    inner = node.get_parameters()
    context = inner(context)
    block = context.get_result()
    if not isinstance(block, _Block):
        # A bare bracketed expression `[expr]` with no `~`: pass its terms up.
        return context.with_result(block)
    deps = context.state.deps
    stage = f'frame{len(deps)}'
    kind = '_tilde' if block.residualise else '_hat'
    subnode = ModelNode(
        name=stage,
        level=Level.DATASET,
        group_by=(),
        combine=Combine.FIXED,
        spec=_block_to_modelspec(block),
    )
    referents = tuple(
        TermSpec((FactorSpec(Referent(stage=stage, kind=kind)),))
        for _ in block.lhs
    )
    context = context.update_state(deps=deps + (subnode,))
    return context.with_result(to_terms(referents))


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
                    mode=Mode.AGGRESSIVE,
                ),
            ),
        )
    return ModelSpec(response=response, fixed=block.rhs, partial=block.partial)


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
    if directives.estimands:
        updates['estimands'] = directives.estimands
    if directives.inference is not None:
        updates['inference'] = directives.inference
    return dataclasses.replace(spec, **updates) if updates else spec


def init_hook(
    ast: Primitive,
    context: NwxContext,
    directives: DirectiveSet | None = None,
) -> tuple[Primitive, NwxContext]:
    return ast, context.update_state(directives=directives)


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
        )
    spec = _apply_directives(spec, directives)
    root = ModelNode(
        name='root',
        level=Level.DATASET,
        group_by=(),
        combine=Combine.FIXED,
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


_DIRECTIVE_RE = re.compile(r'\{\{(.*?)\}\}\s*$', re.DOTALL)


def _split_directives(formula: str) -> tuple[str, DirectiveSet | None]:
    """Split a trailing ``{{ ... }}`` directive block off the formula body.

    Phase 1 supports a single trailing (node- or graph-level) block; full
    bracket-scoped directive placement lands in Phase 4/5. Single-brace
    ``{python}`` execute forms in the body are untouched.
    """
    match = _DIRECTIVE_RE.search(formula)
    if match is None:
        return formula, None
    return formula[: match.start()], parse_directives(match.group(1))


class NwxProcessor:
    """Formula string -> validated :class:`ModelGraph` IR."""

    def __init__(self) -> None:
        self._inner = TransformProcessor(
            grammar=WilkinsonGrammar(),
            preprocessors=(),
            postprocessors=(
                ppr_add_intercept,
                ppr_associative_flatten,
                ppr_common_subexpression,
            ),
            interpreters=INTERPRETERS,
            context_class=NwxContext,
            default_interpreter='spec',
        )
        self._inner.register_initialisation('spec', init_hook)
        self._inner.register_finalisation('spec', finalise_hook)

    def __call__(self, formula: str) -> ModelGraph:
        body, directives = _split_directives(formula)
        return self._inner.transform(body, directives=directives)


def get_processor() -> NwxProcessor:
    return NwxProcessor()
