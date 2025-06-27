# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Wilkinson Transforms
~~~~~~~~~~~~~~~~~~~
Transformations for converting Wilkinson notation AST to formulaic.Formula.
"""
import dataclasses
from enum import Enum
from typing import Any, Dict, Iterable, List, Tuple, Type

import formulaic
from formulaic.parser.types import Factor, Term
from pydantic import Field

from gramform.core import (
    CacheSubcontextMixin,
    ExecutionContext,
    InterpretersDispatch,
    Literal,
    Primitive,
    TypedState,
    TransformProcessor,
)
from gramform.grammars.wilkinson.grammar import WilkinsonGrammar, CONCATENATE
from gramform.postprocessors import (
    ppr_associative_flatten,
    ppr_common_subexpression,
)

INTERPRETERS = InterpretersDispatch()


class OperationalLevel(Enum):
    FACTOR = 'factor'
    TERM = 'term'
    TERMS = 'terms'


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

    def get_result(self) -> Any:
        match self.state.operational_level:
            case OperationalLevel.FACTOR:
                return self.state.factor
            case OperationalLevel.TERM:
                return self.state.term
            case OperationalLevel.TERMS:
                return self.state.terms


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


def LITERAL_impl(
    node: Primitive,
    context: WilkinsonContext,
) -> WilkinsonContext:
    """Handle literal nodes by creating a Factor for literal values."""
    factor = Factor(str(node.value), eval_method="literal")
    return context.set_operational_level(
        OperationalLevel.FACTOR
    ).with_result(factor)


def CONCATENATE_impl(
    node: Primitive,
    context: WilkinsonContext,
) -> WilkinsonContext:
    """Handle concatenation (union) of terms."""
    all_terms = {}

    for child in node.get_parameters():
        result = child(context).get_result()
        if isinstance(result, Iterable):
            all_terms.update(dict.fromkeys(result))
        elif isinstance(result, Term):
            all_terms[result] = None
        elif isinstance(result, Factor):
            all_terms[Term(factors=[result])] = None
        else:
            raise ValueError(f"Unexpected child result: {result}")

    return context.set_operational_level(
        OperationalLevel.TERMS
    ).with_result(all_terms)


def REMOVAL_impl(
    node: Primitive,
    context: WilkinsonContext,
) -> WilkinsonContext:
    """Handle removal of terms."""
    expr, remove = node.get_parameters()
    orig = expr(context).get_result()
    if isinstance(orig, Term):
        orig = {orig: None}
    elif isinstance(orig, Factor):
        orig = {Term(factors=[orig]): None}
    remove = remove(context).get_result()
    if isinstance(remove, Factor):
        remove = {Term(factors=[remove]): None}
    elif isinstance(remove, Term):
        remove = {remove: None}
    else:
        remove = {e: None for e in remove}
    result = {e: None for e in orig if e not in remove}
    return context.set_operational_level(
        OperationalLevel.TERMS
    ).with_result(result)


def INTERACTION_impl(
    node: Primitive,
    context: WilkinsonContext,
) -> WilkinsonContext:
    """Handle interaction (Cartesian product) of factors."""

    def build_factor_seqs(
        candidates: Iterable[Term] | Factor | Term
    ) -> Dict[List[Factor], None]:
        seqs = {}
        if isinstance(candidates, Iterable):
            seqs.update(
                dict.fromkeys(
                    tuple(e.factors)
                    # if isinstance(e, Term)
                    # else [e]
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
    return context


def POWER_impl(
    node: Primitive,
    context: WilkinsonContext,
) -> WilkinsonContext:
    """Handle power operations (polynomial terms)."""
    return context


def NAMED_FUNCTION_impl(
    node: Primitive,
    context: WilkinsonContext,
) -> WilkinsonContext:
    """Handle named function calls."""
    return context


def init_hook(
    ast: Primitive,
    context: WilkinsonContext,
    # include_intercept: bool = True,
) -> Tuple[Primitive, WilkinsonContext]:
    # if include_intercept:
    #     intercept = Literal.create(1, int)
    #     if ast.name == CONCATENATE.name:
    #         ast = CONCATENATE.bind(
    #             *ast.get_parameters(),
    #             intercept,
    #         )
    #     else:
    #         ast = CONCATENATE.bind(
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
INTERPRETERS.register_operation('__all__', 'LITERAL', LITERAL_impl)
INTERPRETERS.register_operation('__all__', 'CONCATENATE', CONCATENATE_impl)
INTERPRETERS.register_operation('__all__', 'REMOVAL', REMOVAL_impl)
INTERPRETERS.register_operation('__all__', 'INTERACTION', INTERACTION_impl)
INTERPRETERS.register_operation('__all__', 'NESTED', NESTED_impl)
INTERPRETERS.register_operation('__all__', 'POWER', POWER_impl)
INTERPRETERS.register_operation('__all__', 'NAMED_FUNCTION', NAMED_FUNCTION_impl)


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


def test_formula_equivalence(wilkinson_expr: str, formulaic_expr: str = None):
    """Test that our Wilkinson parser produces the same result as formulaic."""
    if formulaic_expr is None:
        formulaic_expr = wilkinson_expr

    # Our Wilkinson parser
    processor = get_processor()

    our_result = processor(wilkinson_expr)

    # Formulaic's parser (ground truth)
    try:
        formulaic_result = formulaic.Formula(formulaic_expr)
    except Exception as e:
        print(f"Formulaic failed to parse '{formulaic_expr}': {e}")
        return False

    # Compare results
    our_terms = list(our_result)
    formulaic_terms = list(formulaic_result)
    match = formulaic.Formula(our_terms) == formulaic.Formula(formulaic_terms)
    if not match:
        breakpoint()

    print(f"Wilkinson: '{wilkinson_expr}' -> {our_terms}")
    print(f"Formulaic: '{formulaic_expr}' -> {formulaic_terms}")
    print(f"Match: {match}")
    print()

    return match


def main():
    """Test the Wilkinson formula processor against formulaic ground truth."""
    print("Testing Wilkinson parser against formulaic ground truth:")
    print("=" * 60)

    # Test cases with expected formulaic equivalents
    test_cases = [
        ("x + y", "x + y"),
        ("x:y", "x:y"),
        ("x^2", "x^2"),
        ("dog + cat", "dog + cat"),
        ("rat*dog", "rat*dog"),
        ("cat:dog", "cat:dog"),
        ("(rat*dog + cat:dog)^2", "(rat*dog + cat:dog)^2"),
        ("dog + cat + (rat*dog + cat:dog)^2", "dog + cat + (rat*dog + cat:dog)^2"),
        ("x + y - x - 1", "x + y - x - 1"),
    ]

    all_passed = True
    for wilkinson_expr, formulaic_expr in test_cases:
        if not test_formula_equivalence(wilkinson_expr, formulaic_expr):
            all_passed = False

    print(f"Overall result: {'PASS' if all_passed else 'FAIL'}")


if __name__ == '__main__':
    main()
