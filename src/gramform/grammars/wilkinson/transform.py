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
import dataclasses
from enum import Enum
from typing import Any, Dict, Iterable, Tuple, Type

import formulaic
from formulaic.parser.types import Factor, Term
from pydantic import Field

from gramform.core import (
    CacheSubcontextMixin,
    ExecutionContext,
    InterpretersDispatch,
    Primitive,
    TypedState,
    TransformProcessor,
)
from gramform.grammars.wilkinson.grammar import (
    OperationalLevel,
    WilkinsonGrammar,
    UNARY_NEGATION,
)
from gramform.postprocessors import (
    ppr_associative_flatten,
    ppr_common_subexpression,
)

INTERPRETERS = InterpretersDispatch()
ZERO = Term(factors=[Factor("0", eval_method="literal")])
ONE = Term(factors=[Factor("1", eval_method="literal")])


class WilkinsonState(TypedState):
    factor: Factor | None = None
    term: Term | None = None
    terms: Dict[Term, None] | formulaic.Formula = Field(default_factory=dict)
    operational_level: OperationalLevel = Field(
        default=OperationalLevel.FACTOR
    )

    def evict(self) -> 'WilkinsonState':
        return self.model_validate(
            self.model_dump(include=self.operational_level)
        )


class WilkinsonContext(
    ExecutionContext,
    CacheSubcontextMixin,
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
            case OperationalLevel.NONE:
                return self.state.eval


def VARIABLE_impl(
    node: Primitive,
    context: WilkinsonContext,
) -> WilkinsonContext:
    """Handle variable nodes by creating a Factor for lookup."""
    name = node.get_parameters()
    factor = Factor(name, eval_method="lookup")
    return context.set_operational_level(
        OperationalLevel.FACTOR
    ).with_result(factor)


def NUMERIC_LITERAL_impl(
    node: Primitive,
    context: WilkinsonContext,
) -> WilkinsonContext:
    """Handle literal nodes by creating a Factor for literal values."""
    lit = node.get_parameters()
    factor = Factor(str(lit.value), eval_method="literal")
    return context.set_operational_level(
        OperationalLevel.FACTOR
    ).with_result(factor)


def standardise_code(code: str) -> str:
    """Standardise code by removing whitespace and newlines."""
    return ast.unparse(ast.parse(code, mode='eval')).replace('\n', ' ')


def EXECUTE_impl(
    node: Primitive,
    context: WilkinsonContext,
) -> WilkinsonContext:
    """Handle execution of Python code."""
    code = node.get_parameters()
    code = standardise_code(code)
    factor = Factor(code, eval_method="python")
    return context.set_operational_level(
        OperationalLevel.FACTOR
    ).with_result(factor)


def VARIABLE_COMPLEMENT_impl(
    _: Primitive,
    context: WilkinsonContext,
) -> WilkinsonContext:
    """Handle variable complement nodes by creating a Factor for lookup."""
    term = Term(factors=[Factor('.', eval_method="lookup")])
    return context.set_operational_level(
        OperationalLevel.TERM
    ).with_result(term)


def UNARY_NEGATION_impl(
    node: Primitive,
    context: WilkinsonContext,
) -> WilkinsonContext:
    """Handle unary negation nodes by creating a Factor for lookup."""
    child = node.get_parameters()
    result = child(context).get_result()
    if isinstance(result, Factor):
        result = {Term(factors=[result]): None}
    elif isinstance(result, Term):
        result = {result: None}
    elif isinstance(result, Iterable):
        result = {e: None for e in result}
    return context.set_operational_level(
        OperationalLevel.TERMS
    ).with_result(result)


def APPEND_impl(
    node: Primitive,
    context: WilkinsonContext,
) -> WilkinsonContext:
    """Handle concatenation (union) of terms."""
    all_terms = {}

    for child in node.get_parameters():
        if child.name == UNARY_NEGATION.name:
            all_terms = remove_terms(
                all_terms,
                child(context).get_result(),
            )
            continue
        result = child(context).get_result()
        if isinstance(result, Iterable):
            update = dict.fromkeys(result)
        elif isinstance(result, Term):
            update = {result: None}
        elif isinstance(result, Factor):
            update = {Term(factors=[result]): None}
        else:
            raise ValueError(f"Unexpected child result: {result}")
        all_terms.update(update)
        if ZERO in update:
            del all_terms[ZERO]
            if ONE in all_terms:
                del all_terms[ONE]

    return context.set_operational_level(
        OperationalLevel.TERMS
    ).with_result(all_terms)


def remove_terms(
    orig: Term | Factor | Iterable[Term | Factor],
    remove: Term | Factor | Iterable[Term | Factor],
) -> Dict[Term, None]:
    if isinstance(orig, Factor):
        orig = {Term(factors=[orig]): None}
    elif isinstance(orig, Term):
        orig = {orig: None}
    if isinstance(remove, Factor):
        remove = {Term(factors=[remove]): None}
    elif isinstance(remove, Term):
        remove = {remove: None}
    else:
        remove = {e: None for e in remove}
    if ZERO in remove:
        del remove[ZERO]
        orig[ONE] = None
    result = {e: None for e in orig if e not in remove}
    return result


def REMOVE_impl(
    node: Primitive,
    context: WilkinsonContext,
) -> WilkinsonContext:
    """Handle removal of terms."""
    expr, remove = node.get_parameters()
    orig = expr(context).get_result()
    remove = remove(context).get_result()
    result = remove_terms(orig, remove)
    return context.set_operational_level(
        OperationalLevel.TERMS
    ).with_result(result)


def build_factor_seqs(
    candidates: Iterable[Term] | Factor | Term
) -> Dict[Tuple[Factor], None]:
    seqs = {}
    if isinstance(candidates, Iterable):
        seqs.update(
            dict.fromkeys(
                tuple(e.factors)
                for e in candidates
            )
        )
    elif isinstance(candidates, Term):
        candidates = tuple(candidates.factors)
        seqs[candidates] = None
    elif isinstance(candidates, Factor):
        seqs[(candidates,)] = None
    else:
        raise ValueError(f"Unexpected child result: {candidates}")
    return seqs


def INTERACTION_impl(
    node: Primitive,
    context: WilkinsonContext,
) -> WilkinsonContext:
    """Handle interaction (Cartesian product) of factors."""

    children = node.get_parameters()
    first, remaining = children[0], children[1:]
    result = first(context).get_result()
    factor_seqs = build_factor_seqs(result)

    for next in remaining:
        result = next(context).get_result()
        new_seqs = build_factor_seqs(result)
        factor_seqs = {
            a + b: None
            for a in factor_seqs
            for b in new_seqs
        }

    factor_seqs = {Term(factors=seq): None for seq in factor_seqs}
    return context.set_operational_level(
        OperationalLevel.TERMS
    ).with_result(factor_seqs)


def NESTED_impl(
    node: Primitive,
    context: WilkinsonContext,
) -> WilkinsonContext:
    """Handle nested effects (hierarchical structure)."""
    left, right = node.get_parameters()

    left_result = left(context).get_result()
    right_result = right(context).get_result()

    left_factors = build_factor_seqs(left_result)
    right_factors = build_factor_seqs(right_result)

    left_reduced = sum(left_factors.keys(), ())

    factor_seqs = {
        **{Term(factors=a): None for a in left_factors},
        **{Term(factors=left_reduced + b): None for b in right_factors},
    }
    return context.set_operational_level(
        OperationalLevel.TERMS
    ).with_result(factor_seqs)


def POWER_impl(
    node: Primitive,
    context: WilkinsonContext,
) -> WilkinsonContext:
    """
    Power operations are currently lowered to APPEND and INTERACTION.

    This is here in case we find a more efficient way to handle power
    operations as a primitive.
    """
    raise NotImplementedError("Power operations are not yet supported")


def PARAMETER_impl(
    node: Primitive,
    context: WilkinsonContext,
) -> WilkinsonContext:
    """Handle parameters."""
    value = node.get_parameters()
    return context.set_operational_level(OperationalLevel.NONE).with_result(
        f"{value(context).get_result()}"
    )


def NAMED_PARAMETER_impl(
    node: Primitive,
    context: WilkinsonContext,
) -> WilkinsonContext:
    """Handle parameters."""
    name, value = node.get_parameters()
    return context.set_operational_level(OperationalLevel.NONE).with_result(
        f"{name}={value(context).get_result()}"
    )


def FUNCTION_PARAMETERS_impl(
    node: Primitive,
    context: WilkinsonContext,
) -> WilkinsonContext:
    """Handle parameters."""
    parameters = [
        param(context).get_result()
        for param in node.get_parameters()
    ]
    dummy_call = f'f({", ".join(parameters)})'
    parameters = f", {standardise_code(dummy_call)[2:-1]}"
    return context.set_operational_level(OperationalLevel.NONE).with_result(
        parameters
    )


def NAMED_FUNCTION_impl(
    node: Primitive,
    context: WilkinsonContext,
) -> WilkinsonContext:
    """Handle named function calls."""
    name, expr, *args, level = node.get_parameters()
    argstr = args[0](context).get_result() if args else ""
    result = expr(context).get_result()
    if isinstance(result, Factor):
        result = {Term(factors=[result]): None}
    elif isinstance(result, Term):
        result = {result: None}
    elif isinstance(result, Iterable):
        result = {e: None for e in result}
    match level:
        case OperationalLevel.FACTOR:
            result = {
                Term(
                    factors=[
                        Factor(f"{name}({f}{argstr})", eval_method="python")
                        for f in e.factors
                    ],
                ): None
                for e in result
            }
        case OperationalLevel.TERM:
            result = {
                Term(
                    factors=[
                        Factor(
                            (
                                f"{name}("
                                f"{' * '.join(f.expr for f in e.factors)}"
                                f"{argstr})"
                            ),
                            eval_method="python",
                        ),
                    ],
                ): None
                for e in result
            }
    return context.set_operational_level(
        OperationalLevel.TERMS
    ).with_result(result)


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
    return ast, context.set_operational_level(
        OperationalLevel.TERMS
    )


def finalise_hook(
    context: WilkinsonContext,
) -> WilkinsonContext:
    result = context.get_result()
    if context.state.operational_level == OperationalLevel.FACTOR:
        result = [Term(factors=[result])]
    elif context.state.operational_level == OperationalLevel.TERM:
        result = [result]
    result = sorted(list(result), key=lambda t: len(t.factors))
    return context.set_operational_level(
        OperationalLevel.TERMS
    ).with_result(formulaic.Formula(result))


def add_intercept_preprocessor(expr: str) -> str:
    return f"1 + {expr}"


# Register interpreters
INTERPRETERS.register_interpreter('formulaic')
INTERPRETERS.register_operation('__all__', 'VARIABLE', VARIABLE_impl)
INTERPRETERS.register_operation('__all__', 'NUMERIC_LITERAL', NUMERIC_LITERAL_impl)
INTERPRETERS.register_operation('__all__', 'EXECUTE', EXECUTE_impl)
INTERPRETERS.register_operation('__all__', 'VARIABLE_COMPLEMENT', VARIABLE_COMPLEMENT_impl)
INTERPRETERS.register_operation('__all__', 'UNARY_NEGATION', UNARY_NEGATION_impl)
INTERPRETERS.register_operation('__all__', 'APPEND', APPEND_impl)
INTERPRETERS.register_operation('__all__', 'REMOVE', REMOVE_impl)
INTERPRETERS.register_operation('__all__', 'INTERACTION', INTERACTION_impl)
INTERPRETERS.register_operation('__all__', 'NESTED', NESTED_impl)
INTERPRETERS.register_operation('__all__', 'POWER', POWER_impl)
INTERPRETERS.register_operation('__all__', 'NAMED_FUNCTION', NAMED_FUNCTION_impl)
INTERPRETERS.register_operation('__all__', 'PARAMETER', PARAMETER_impl)
INTERPRETERS.register_operation('__all__', 'NAMED_PARAMETER', NAMED_PARAMETER_impl)
INTERPRETERS.register_operation('__all__', 'FUNCTION_PARAMETERS', FUNCTION_PARAMETERS_impl)


def get_processor():
    processor = TransformProcessor(
        grammar=WilkinsonGrammar(),
        preprocessors=(add_intercept_preprocessor,),
        postprocessors=(
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
