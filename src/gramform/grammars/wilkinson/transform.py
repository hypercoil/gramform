# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Wilkinson Transforms
~~~~~~~~~~~~~~~~~~~~
Transformations for converting Wilkinson notation AST to formulaic.Formula.

This is a proof of concept---formulaic's parser already supports Wilkinson
notation---but we use it as a composable component in an extended Wilkinson
grammar.
"""

import ast
import warnings
from functools import reduce, singledispatch
from typing import Any, Dict, Iterable, Tuple, Type

import formulaic
from formulaic.parser.types import Factor, Term
from formulaic.utils.structured import Structured
from pydantic import Field

from gramform.core import (
    ExecutionContext,
    InterpretersDispatch,
    Primitive,
    TransformProcessor,
    TypedState,
    withCacheSubcontext,
)
from gramform.grammars.wilkinson.grammar import (
    APPEND,
    LHS_RHS_STRUCTURE,
    NUMERIC_LITERAL,
    PUSH_FRAME,
    REMOVE,
    RESIDUAL_STRUCTURE,
    SUBPARTS_STRUCTURE,
    UNARY_NEGATION,
    OperationalLevel,
    WilkinsonGrammar,
    lift_literal,
)
from gramform.postprocessors import (
    ppr_associative_flatten,
    ppr_common_subexpression,
)

INTERPRETERS: InterpretersDispatch = InterpretersDispatch()
ZERO: Term = Term(
    factors=[Factor('0', eval_method=Factor.EvalMethod.LITERAL)],
)
ONE: Term = Term(
    factors=[Factor('1', eval_method=Factor.EvalMethod.LITERAL)],
)


class InvalidPromotion(ValueError):
    """Exception raised when a promotion is invalid."""

    pass


@singledispatch
def to_term(arg: Iterable[Factor]) -> Term:
    variables = [f for f in arg if f.eval_method != Factor.EvalMethod.LITERAL]
    # Collapse literals into a single factor
    literals = [f for f in arg if f.eval_method == Factor.EvalMethod.LITERAL]
    if literals:
        literal = reduce(
            lambda x, y: x * y,
            [
                float(f.expr) if '.' in f.expr else int(f.expr)
                for f in literals
            ],
            1,
        )
        literals = [
            Factor(str(literal), eval_method=Factor.EvalMethod.LITERAL)
        ]
    return Term(factors=(literals + variables))


@to_term.register
def _(arg: Factor) -> Term:
    return Term(factors=[arg])


@to_term.register
def _(arg: Term) -> Term:
    return arg


@to_term.register
def _(arg: dict) -> Term:
    raise InvalidPromotion(
        f'Cannot promote a sequence of Terms to a Term: {arg}'
    )


@to_term.register
def _(arg: Any) -> Term:
    raise InvalidPromotion(
        f'Cannot promote an object of type {type(arg)} to a Term: {arg}'
    )


@singledispatch
def to_terms(arg: Any) -> Dict[Term, None]:
    raise InvalidPromotion(
        f'Cannot promote an object of type {type(arg)} to a sequence of '
        f'Terms: {arg}'
    )


@to_terms.register
def _(arg: dict) -> Dict[Term, None]:
    # Ensure no terms are duplicated, or duplicated up to a constant factor
    # so we disallow structural singularities.
    scales = {}
    keys = {}
    to_remove = []
    for t in arg:
        literals = [
            f for f in t.factors if f.eval_method == Factor.EvalMethod.LITERAL
        ]
        variables = [
            f for f in t.factors if f.eval_method != Factor.EvalMethod.LITERAL
        ]
        variables = tuple(sorted(variables, key=lambda x: x.expr))
        scaled = scales.get(variables, None)
        # TODO
        # I think we're hitting this block more frequently than we need to.
        # We can worry about optimising this later. In practice the parse
        # time is very small compared to steps like model matrix
        # generation.
        # # print(f"variables: {variables}")
        # # print(f"literals: {literals}")
        # # print(f"scaled: {scaled}")
        if literals:
            scale = reduce(
                lambda x, y: x * y,
                [
                    float(f.expr) if '.' in f.expr else int(f.expr)
                    for f in literals
                ],
            )
            if not variables and scale != 1:
                # intercept term: we should only allow a scale of 1
                warnings.warn(
                    f'Constant term {t} was interpreted as a scale of '
                    f'{scale}, but only an intercept constant (scale of 1) '
                    f'is allowed. This term has been removed from the '
                    'formula automatically.'
                )
                instances_to_remove = [t]
            elif scaled is not None:
                raise InvalidPromotion(
                    f'Attempting to scale term {Term(variables)} by {scale}, '
                    f'but {Term(variables)} has already been scaled by '
                    f'{scaled}'
                )
            else:
                scales[variables] = scale
                instances_to_remove = keys.get(variables, [])
            for instance in instances_to_remove:
                # Replace the instance with the scaled version if
                # both are present
                to_remove.append(instance)
                # arg[instance] = Term(
                #     factors=[
                #         Factor(
                #             str(scale),
                #             eval_method=Factor.EvalMethod.LITERAL,
                #         )
                #     ] + arg[instance].factors
                # )
        keys[variables] = keys.get(variables, []) + [t]
    for instance in to_remove:
        del arg[instance]
    return {t: None for t in arg}


@to_terms.register
def _(arg: Term) -> Dict[Term, None]:
    return {arg: None}


@to_terms.register
def _(arg: Factor) -> Dict[Term, None]:
    return {Term(factors=[arg]): None}


@to_terms.register
def _(arg: formulaic.SimpleFormula) -> Dict[Term, None]:
    # Technically this is an invalid promotion, because we're not
    # returning a sequence of Terms. But the only place this comes up,
    # we want to keep the SimpleFormula as is.
    return arg


@to_terms.register
def _(arg: Structured) -> Dict[Term, None]:
    # Technically this is an invalid promotion, because we're not
    # returning a sequence of Terms. But the only place this comes up,
    # we want to keep the Structured as is.
    return arg


class WilkinsonState(TypedState):
    factor: Factor | None = None
    term: Term | None = None
    terms: Dict[Term, None] = Field(default_factory=dict)
    block: Dict[Term, None] | formulaic.Formula = Field(default_factory=dict)
    operational_level: OperationalLevel = Field(
        default=OperationalLevel.FACTOR
    )
    dependencies: list[formulaic.Formula] | None = None

    def evict(self) -> 'WilkinsonState':
        return self.model_validate(
            self.model_dump(include=self.operational_level)
        )


class WilkinsonContext(
    ExecutionContext,
    withCacheSubcontext,
):
    __state__: Type[WilkinsonState] = WilkinsonState

    def set_operational_level(
        self,
        level: OperationalLevel,
    ) -> 'WilkinsonContext':
        return self.update_state(operational_level=level)

    def with_result(self, result: Any) -> 'WilkinsonContext':
        match self.state.operational_level:
            case OperationalLevel.FACTOR:
                return self.update_state(factor=result)
            case OperationalLevel.TERM:
                return self.update_state(term=result)
            case OperationalLevel.TERMS:
                return self.update_state(terms=result)
            case OperationalLevel.BLOCK:
                return self.update_state(block=result)
            case OperationalLevel.NONE:
                return self.update_state(eval=result)

    def get_result(self) -> Any:
        match self.state.operational_level:
            case OperationalLevel.FACTOR:
                return self.state.factor
            case OperationalLevel.TERM:
                return self.state.term
            case OperationalLevel.TERMS:
                return self.state.terms
            case OperationalLevel.BLOCK:
                return self.state.block
            case OperationalLevel.NONE:
                return self.state.eval


def standardise_code(code: str) -> str:
    """Standardise code by removing whitespace and newlines."""
    return ast.unparse(ast.parse(code, mode='eval')).replace('\n', ' ')


def remove_terms(
    orig: Term | Factor | Iterable[Term | Factor],
    remove: Term | Factor | Iterable[Term | Factor],
) -> Dict[Term, None]:
    orig = to_terms(orig)
    remove = to_terms(remove)
    if ZERO in remove:
        del remove[ZERO]
        orig[ONE] = None  # Interpret removal of ZERO as addition of ONE
    result = {e: None for e in orig if e not in remove}
    return result


def build_factor_seqs(
    candidates: Iterable[Term] | Factor | Term,
) -> Dict[Tuple[Factor], None]:
    seqs = {}
    if isinstance(candidates, Iterable):
        seqs.update(dict.fromkeys(tuple(e.factors) for e in candidates))
    elif isinstance(candidates, Term):
        candidates = tuple(candidates.factors)
        seqs[candidates] = None
    elif isinstance(candidates, Factor):
        seqs[(candidates,)] = None
    else:
        raise ValueError(f'Unexpected child result: {candidates}')
    return seqs


def build_simple_formula(
    result: Any,
    context: WilkinsonContext,
) -> Tuple[Any, bool]:
    if context.state.operational_level == OperationalLevel.FACTOR:
        result = [Term(factors=[result])]
    elif context.state.operational_level == OperationalLevel.TERM:
        result = [result]
    elif context.state.operational_level == OperationalLevel.BLOCK:
        # If it's a formula block, that implies complex structure that cannot
        # be represented as a simple formula.
        result = result._structure
        return result.get('root', result), False
    result = sorted(list(result), key=lambda t: len(t.factors))
    return formulaic.SimpleFormula(result), True


def _add_dependencies(
    context: WilkinsonContext,
    result: Any,
) -> WilkinsonContext:
    if context.state.dependencies is not None:
        result = Structured(
            result,
            deps=context.state.dependencies,
        )
        context = context.update_state(dependencies=None)
    return context, result


def _referent_terms(dependencies: Structured, suffix: str) -> Dict[Term, None]:
    return {
        Term(
            Factor(
                f'{f}{suffix}',
                eval_method=Factor.EvalMethod.LOOKUP,
            )
            for f in e.factors
        ): None
        for e in dependencies
    }


def VARIABLE_impl(
    node: Primitive,
    context: WilkinsonContext,
) -> WilkinsonContext:
    """Handle variable nodes by creating a Factor for lookup."""
    name = node.get_parameters()
    factor = Factor(name, eval_method=Factor.EvalMethod.LOOKUP)
    return context.set_operational_level(OperationalLevel.FACTOR).with_result(
        factor
    )


def NUMERIC_LITERAL_impl(
    node: Primitive,
    context: WilkinsonContext,
) -> WilkinsonContext:
    """Handle literal nodes by creating a Factor for literal values."""
    lit = node.get_parameters()
    factor = Factor(str(lit.value), eval_method=Factor.EvalMethod.LITERAL)
    return context.set_operational_level(OperationalLevel.FACTOR).with_result(
        factor
    )


def EXECUTE_impl(
    node: Primitive,
    context: WilkinsonContext,
) -> WilkinsonContext:
    """Handle execution of Python code."""
    code = node.get_parameters()
    code = standardise_code(code)
    factor = Factor(code, eval_method=Factor.EvalMethod.PYTHON)
    return context.set_operational_level(OperationalLevel.FACTOR).with_result(
        factor
    )


def VARIABLE_COMPLEMENT_impl(
    _: Primitive,
    context: WilkinsonContext,
) -> WilkinsonContext:
    """Handle variable complement nodes by creating a Factor for lookup."""
    term = to_term(Factor('.', eval_method=Factor.EvalMethod.LOOKUP))
    return context.set_operational_level(OperationalLevel.TERM).with_result(
        term
    )


def UNARY_NEGATION_impl(
    node: Primitive,
    context: WilkinsonContext,
) -> WilkinsonContext:
    """Handle unary negation nodes by creating a Factor for lookup."""
    child = node.get_parameters()
    context = child(context)
    result = to_terms(context.get_result())
    return context.set_operational_level(OperationalLevel.TERMS).with_result(
        result
    )


def APPEND_impl(
    node: Primitive,
    context: WilkinsonContext,
) -> WilkinsonContext:
    """Handle concatenation (union) of terms."""
    all_terms = {}

    for child in node.get_parameters():
        context = child(context)
        if child.name == UNARY_NEGATION.name:
            all_terms = remove_terms(
                all_terms,
                context.get_result(),
            )
            continue
        result = context.get_result()
        update = to_terms(result)
        all_terms.update(update)
        if ZERO in update:
            del all_terms[ZERO]
            if ONE in all_terms:
                del all_terms[ONE]

    return context.set_operational_level(OperationalLevel.TERMS).with_result(
        to_terms(all_terms)
    )


def REMOVE_impl(
    node: Primitive,
    context: WilkinsonContext,
) -> WilkinsonContext:
    """Handle removal of terms."""
    expr, remove = node.get_parameters()
    context = expr(context)
    orig = context.get_result()
    context = remove(context)
    remove = context.get_result()
    result = remove_terms(orig, remove)
    return context.set_operational_level(OperationalLevel.TERMS).with_result(
        result
    )


def INTERACTION_impl(
    node: Primitive,
    context: WilkinsonContext,
) -> WilkinsonContext:
    """Handle interaction (Cartesian product) of factors."""

    children = node.get_parameters()
    first, remaining = children[0], children[1:]
    context = first(context)
    factor_seqs = build_factor_seqs(context.get_result())

    for next in remaining:
        context = next(context)
        new_seqs = build_factor_seqs(context.get_result())
        factor_seqs = {a + b: None for a in factor_seqs for b in new_seqs}

    factor_seqs = {to_term(seq): None for seq in factor_seqs}
    return context.set_operational_level(OperationalLevel.TERMS).with_result(
        to_terms(factor_seqs)
    )


def NESTED_impl(
    node: Primitive,
    context: WilkinsonContext,
) -> WilkinsonContext:
    """Handle nested effects (hierarchical structure)."""
    left, right = node.get_parameters()

    context = left(context)
    left_result = context.get_result()
    context = right(context)
    right_result = context.get_result()

    left_factors = build_factor_seqs(left_result)
    right_factors = build_factor_seqs(right_result)

    left_reduced = sum(left_factors.keys(), ())

    factor_seqs = {
        **{to_term(a): None for a in left_factors},
        **{to_term(left_reduced + b): None for b in right_factors},
    }
    return context.set_operational_level(OperationalLevel.TERMS).with_result(
        to_terms(factor_seqs)
    )


def POWER_impl(
    node: Primitive,
    context: WilkinsonContext,
) -> WilkinsonContext:
    """
    Power operations are currently lowered to APPEND and INTERACTION.

    This is here in case we find a more efficient way to handle power
    operations as a primitive.
    """
    raise NotImplementedError('Power operations are not yet supported')


def PARAMETER_impl(
    node: Primitive,
    context: WilkinsonContext,
) -> WilkinsonContext:
    """Handle parameters."""
    value = node.get_parameters()
    return context.set_operational_level(OperationalLevel.NONE).with_result(
        f'{value(context).get_result()}'
    )


def NAMED_PARAMETER_impl(
    node: Primitive,
    context: WilkinsonContext,
) -> WilkinsonContext:
    """Handle parameters."""
    name, value = node.get_parameters()
    return context.set_operational_level(OperationalLevel.NONE).with_result(
        f'{name}={value(context).get_result()}'
    )


def FUNCTION_PARAMETERS_impl(
    node: Primitive,
    context: WilkinsonContext,
) -> WilkinsonContext:
    """Handle parameters."""
    parameters = [
        param(context).get_result() for param in node.get_parameters()
    ]
    dummy_call = f'f({", ".join(parameters)})'
    parameters = f', {standardise_code(dummy_call)[2:-1]}'
    return context.set_operational_level(OperationalLevel.NONE).with_result(
        parameters
    )


def NAMED_FUNCTION_impl(
    node: Primitive,
    context: WilkinsonContext,
) -> WilkinsonContext:
    """Handle named function calls."""
    name, expr, *args, level = node.get_parameters()
    argstr = args[0](context).get_result() if args else ''
    context = expr(context)
    result = to_terms(context.get_result())
    match level:
        case OperationalLevel.FACTOR:
            result = to_terms(
                {
                    to_term(
                        [
                            Factor(
                                f'{name}({f}{argstr})',
                                eval_method=Factor.EvalMethod.PYTHON,
                            )
                            for f in e.factors
                        ]
                    ): None
                    for e in result
                }
            )
        case OperationalLevel.TERM:
            result = to_terms(
                {
                    to_term(
                        [
                            Factor(
                                (
                                    f'{name}('
                                    f'{" * ".join(f.expr for f in e.factors)}'
                                    f'{argstr})'
                                ),
                                eval_method=Factor.EvalMethod.PYTHON,
                            )
                        ]
                    ): None
                    for e in result
                }
            )
    return context.set_operational_level(OperationalLevel.TERMS).with_result(
        result
    )


def LHS_RHS_STRUCTURE_impl(
    node: Primitive,
    context: WilkinsonContext,
) -> WilkinsonContext:
    """Assign subparts of the formula to the LHS and RHS."""
    lhs_expr, rhs_expr = node.get_parameters()
    context = lhs_expr(context)
    lhs, _ = build_simple_formula(context.get_result(), context)
    context, lhs = _add_dependencies(context, lhs)
    context = rhs_expr(context)
    rhs, _ = build_simple_formula(context.get_result(), context)
    context, rhs = _add_dependencies(context, rhs)
    result = Structured(
        lhs=lhs,
        rhs=rhs,
    )
    return context.set_operational_level(OperationalLevel.BLOCK).with_result(
        result
    )


def RESIDUAL_STRUCTURE_impl(
    node: Primitive,
    context: WilkinsonContext,
) -> WilkinsonContext:
    """Assign subparts of the formula to the LHS and RHS."""
    residualise_expr, wrt_expr = node.get_parameters()
    context = residualise_expr(context)
    residualise, _ = build_simple_formula(context.get_result(), context)
    context, residualise = _add_dependencies(context, residualise)
    context = wrt_expr(context)
    wrt, _ = build_simple_formula(context.get_result(), context)
    context, wrt = _add_dependencies(context, wrt)
    result = Structured(
        residualise=residualise,
        wrt=wrt,
    )
    return context.set_operational_level(OperationalLevel.BLOCK).with_result(
        result
    )


def SUBPARTS_STRUCTURE_impl(
    node: Primitive,
    context: WilkinsonContext,
) -> WilkinsonContext:
    """Assign subparts of the formula to the LHS and RHS."""
    subparts = []
    for child in node.get_parameters():
        context = child(context)
        result = context.get_result()
        result = to_terms(result)
        result, _ = build_simple_formula(result, context)
        context, result = _add_dependencies(context, result)
        subparts.append(result)
    result = Structured(tuple(subparts))
    return context.set_operational_level(OperationalLevel.BLOCK).with_result(
        result
    )


def PUSH_FRAME_impl(
    node: Primitive,
    context: WilkinsonContext,
) -> WilkinsonContext:
    """Push a frame onto the stack."""
    inner = node.get_parameters()
    subcontext = context.push()
    subcontext = inner(subcontext)
    dependencies = subcontext.get_result()
    if not isinstance(dependencies, Structured):
        return context.with_result(dependencies)
    if 'lhs' in dependencies:
        result = _referent_terms(dependencies['lhs'], '_hat')
    elif 'residualise' in dependencies:
        result = _referent_terms(dependencies['residualise'], '_tilde')
    else:
        raise ValueError(f'Unknown dependencies: {dependencies}')
    if subcontext.state.dependencies is not None:
        dependencies = Structured(
            dependencies,
            deps=subcontext.state.dependencies,
        )
    if context.state.dependencies is not None:
        dependencies = [
            *context.state.dependencies,
            dependencies,
        ]
    else:
        dependencies = [dependencies]
    context = context.set_operational_level(
        OperationalLevel.TERMS
    ).update_state(
        terms=to_terms(result),
        dependencies=tuple(dependencies),
    )
    return context


def init_hook(
    ast: Primitive,
    context: WilkinsonContext,
    # include_intercept: bool = True,
) -> Tuple[Primitive, WilkinsonContext]:
    # if include_intercept:
    #     intercept = Literal.create(1, int)
    #     if ast.name == APPEND.name:
    #         ast = APPEND.bind(
    #             *ast.get_parameters(),
    #             intercept,
    #         )
    #     else:
    #         ast = APPEND.bind(
    #             ast,
    #             intercept,
    #         )
    return ast, context.set_operational_level(OperationalLevel.TERMS)


def finalise_hook(
    context: WilkinsonContext,
) -> WilkinsonContext:
    result = context.get_result()
    result, is_simple_formula = build_simple_formula(result, context)
    context, result = _add_dependencies(context, result)
    if is_simple_formula:
        return context.set_operational_level(
            OperationalLevel.BLOCK
        ).with_result(result)
    else:
        return context.set_operational_level(
            OperationalLevel.BLOCK
        ).with_result(formulaic.Formula(result))


def add_intercept_to_formula(
    tree: Primitive,
    context: ExecutionContext,
) -> Tuple[Primitive, ExecutionContext]:
    if tree.name in (
        'LHS_RHS_STRUCTURE',
        'RESIDUAL_STRUCTURE',
    ):
        op = (
            LHS_RHS_STRUCTURE
            if tree.name == 'LHS_RHS_STRUCTURE'
            else RESIDUAL_STRUCTURE
        )
        left, right = tree.parameters
        right = add_intercept_to_formula(right, context)[0]
        return op.bind(
            left,
            right,
        ), context
    elif tree.name == 'SUBPARTS_STRUCTURE':
        return SUBPARTS_STRUCTURE.bind(
            *[
                add_intercept_to_formula(child, context)[0]
                for child in tree.parameters
            ],
        ), context
    elif tree.name in (
        'APPEND',
        'REMOVE',
    ):
        op = APPEND if tree.name == 'APPEND' else REMOVE
        left, *others = tree.parameters
        left = add_intercept_to_formula(left, context)[0]
        return op.bind(
            left,
            *others,
        ), context
    return APPEND.bind(
        lift_literal(int, NUMERIC_LITERAL)(1),
        tree,
    ), context


def ppr_add_intercept(
    tree: Primitive,
    context: ExecutionContext,
) -> Tuple[Primitive, ExecutionContext]:
    def _walk(
        tree: Primitive,
        context: ExecutionContext,
    ) -> Tuple[Primitive, ExecutionContext]:
        if not isinstance(tree, Primitive) or tree.is_terminal:
            return tree, context
        if tree.name == 'PUSH_FRAME':
            return PUSH_FRAME.bind(
                ppr_add_intercept(
                    tree.get_parameters(),
                    context,
                )[0],
            ), context
        return tree.bind(
            *[_walk(child, context)[0] for child in tree.parameters],
        ), context

    tree, context = add_intercept_to_formula(tree, context)
    return _walk(tree, context)


# Register interpreters
INTERPRETERS.register_interpreter('formulaic')
INTERPRETERS.register_group('build', ['formulaic'])
INTERPRETERS.register_operation('build', 'VARIABLE', VARIABLE_impl)
INTERPRETERS.register_operation(
    'build', 'NUMERIC_LITERAL', NUMERIC_LITERAL_impl
)
INTERPRETERS.register_operation('build', 'EXECUTE', EXECUTE_impl)
INTERPRETERS.register_operation(
    'build', 'VARIABLE_COMPLEMENT', VARIABLE_COMPLEMENT_impl
)
INTERPRETERS.register_operation('build', 'UNARY_NEGATION', UNARY_NEGATION_impl)
INTERPRETERS.register_operation('build', 'APPEND', APPEND_impl)
INTERPRETERS.register_operation('build', 'REMOVE', REMOVE_impl)
INTERPRETERS.register_operation('build', 'INTERACTION', INTERACTION_impl)
INTERPRETERS.register_operation('build', 'NESTED', NESTED_impl)
INTERPRETERS.register_operation('build', 'POWER', POWER_impl)
INTERPRETERS.register_operation('build', 'NAMED_FUNCTION', NAMED_FUNCTION_impl)
INTERPRETERS.register_operation('build', 'PARAMETER', PARAMETER_impl)
INTERPRETERS.register_operation(
    'build', 'NAMED_PARAMETER', NAMED_PARAMETER_impl
)
INTERPRETERS.register_operation(
    'build', 'FUNCTION_PARAMETERS', FUNCTION_PARAMETERS_impl
)
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


def get_processor():
    processor = TransformProcessor(
        grammar=WilkinsonGrammar(),
        preprocessors=(),
        postprocessors=(
            ppr_add_intercept,
            ppr_associative_flatten,
            ppr_common_subexpression,
        ),
        interpreters=INTERPRETERS,
        context_class=WilkinsonContext,
        default_interpreter='formulaic',
    )
    processor.register_initialisation('formulaic', init_hook)
    processor.register_finalisation('formulaic', finalise_hook)
    return processor
