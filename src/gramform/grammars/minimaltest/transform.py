# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Minimal grammar for testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Transformations for the minimal test grammar.
"""
import dataclasses
import operator
from functools import reduce
from itertools import chain
from typing import Any, Iterable, List, Type

import narwhals as nw
import numpy as np
from narwhals.typing import IntoFrameT
from pydantic import field_validator

from gramform.core import (
    ExecutionContext,
    InterpretersDispatch,
    Primitive,
    TransformProcessor,
    Tuple,
    TypedState,
    withCacheSubcontext,
)
from gramform.grammars.minimaltest.grammar import (
    MinimalGrammar,
    confound_formula_preprocessor,
)
from gramform.postprocessors import (
    ppr_associative_flatten,
    ppr_common_subexpression,
)

INTERPRETERS = InterpretersDispatch()


class DataFrameState(TypedState):
    data: Any = None
    select: List[str] = dataclasses.field(default_factory=list)

    @field_validator('data')
    def check_dataframe(cls, v):
        if v is None:
            return v
        try:
            nw.from_native(v)  # Try to wrap with narwhals
            return v
        except Exception:
            raise ValueError("Expected a narwhals-compatible DataFrame")


class DataFrameContext(
    ExecutionContext,
    withCacheSubcontext,
):
    __state__: Type[DataFrameState] = DataFrameState

    def with_data(self, data: IntoFrameT) -> 'DataFrameContext':
        return self.update_state(data=data)

    def with_selection(self, select: list[str]) -> 'DataFrameContext':
        return self.update_state(select=select)

    def with_eval(self, eval: Any) -> 'DataFrameContext':
        return self.update_state(eval=eval)

    def get_data(self) -> IntoFrameT:
        return self.state.data

    def get_selection(self) -> list[str]:
        return self.state.select

    def get_eval(self) -> Any:
        return self.state.eval


def VARIABLE_impl(node, context):
    name, state = node.get_parameters(), context.state
    if name not in state.data:
        raise ValueError(f"Variable {name} not found in data")
    return context.with_selection(state.select + [name])


def RANGE_impl(node, context):
    expr_start, expr_end = node.get_parameters()
    start, context = expr_start(context).pop()
    end, context = expr_end(context).pop()
    return context.with_result(range(start.eval, end.eval + 1))


def ENUM_impl(node, context):
    eval = []
    for child in node.get_parameters():
        result, context = child(context).pop()
        new_eval = result.eval
        if not isinstance(new_eval, Iterable):
            new_eval = (new_eval,)
        eval.extend(new_eval)
    context = context.with_result(chain(eval))
    return context


def CONCATENATE_impl(node, context):
    state, context = context.pop('select')
    selection = state.select
    for child in node.parameters:
        new_state, context = child(context).pop('select')
        selection.extend(new_state.select)
    return context.with_selection(selection)


def POWER_impl(node, context):
    argument, power = node.get_parameters()
    context = argument(context)
    state, context = context.pop('data', 'select')
    data, selection = state.data, state.select
    context = power(context)
    state, context = context.pop('eval', 'select')
    pow_order, pow_cols = state.eval, state.select
    if pow_cols:
        raise ValueError("Power operation does not support column selection")
    new_selection = []
    values = [nw.col(e) for e in selection]
    if not isinstance(pow_order, Iterable):
        pow_order = (pow_order,)
    for pow in pow_order:
        if pow == 1:
            new_selection.extend(selection)
            continue
        new_columns = [f'{e}_power{pow}' for e in selection]
        data = data.with_columns([
            (arg ** pow).alias(col)
            for col, arg in zip(new_columns, values)
        ])
        new_selection.extend(new_columns)
    context = context.update_state(data=data, select=new_selection)
    return context


def BACKDIFF_impl(node, context):
    argument, order = node.get_parameters()
    state, context = argument(context).pop('data', 'select')
    data, selection = state.data, state.select
    state, context = order(context).pop('eval', 'select')
    order, order_cols = state.eval, state.select
    if order_cols:
        raise ValueError(
            "Backdiff operation does not support column selection"
        )
    new_selection, result = [], {}
    values = [nw.col(e) for e in selection]
    if not isinstance(order, Iterable):
        order = (order,)
    required_orders = set(order)
    max_order = max(tuple(order))
    for ord in range(1, max_order + 1):
        values = [e.diff() for e in values]
        if ord in required_orders:
            result[ord] = values
    for ord in order:
        if ord == 0:
            new_selection.extend(selection)
            continue
        new_columns = [f'{c}_derivative{ord}' for c in selection]
        data = data.with_columns([
            e.over(order_by='index').alias(col)
            for e, col in zip(result[ord], new_columns)
        ])
        new_selection.extend(new_columns)
    context = context.update_state(data=data, select=new_selection)
    return context


def BINOP_impl(node, context, op: callable, col_infix: str):
    left_expr, right_expr = node.get_parameters()
    state, context = left_expr(context).pop('select', 'eval')
    left_selection, left_eval = state.select, state.eval
    state, context = right_expr(context).pop('select', 'eval')
    right_selection, right_eval = state.select, state.eval
    data = context.get_data()
    if left_selection is not None:
        left_args = [nw.col(e) for e in left_selection]
        left_cols = left_selection
    else:
        left_args = [left_eval]
        left_cols = [f'{left_eval}']
    if right_selection is not None:
        right_args = [nw.col(e) for e in right_selection]
        right_cols = right_selection
    else:
        right_args = [right_eval]
        right_cols = [f'{right_eval}']
    result = [
        op(left_arg, right_arg)
        for left_arg, right_arg in zip(left_args, right_args)
    ]
    new_columns = [
        f"{lcol}_{col_infix}_{rcol}"
        for lcol in left_cols
        for rcol in right_cols
    ]
    data = data.with_columns([
        val.alias(col)
        for val, col in zip(result, new_columns)
    ])
    # if data.isnull().any().any():
    #     raise ValueError("Result of binary operation contains NaN")
    return context.update_state(data=data, select=new_columns)


def CONDITION_EQUAL_impl(node, context):
    return BINOP_impl(node, context, operator.eq, 'eq')


def CONDITION_NOT_EQUAL_impl(node, context):
    return BINOP_impl(node, context, operator.ne, 'ne')


def CONDITION_GREATER_impl(node, context):
    return BINOP_impl(node, context, operator.gt, 'gt')


def CONDITION_LESS_impl(node, context):
    return BINOP_impl(node, context, operator.lt, 'lt')


def CONDITION_GREATER_EQUAL_impl(node, context):
    return BINOP_impl(node, context, operator.ge, 'ge')


def CONDITION_LESS_EQUAL_impl(node, context):
    return BINOP_impl(node, context, operator.le, 'le')


def INTERSECTION_impl(node, context):
    return BINOP_impl(node, context, np.logical_and, 'and')


def UNION_impl(node, context):
    return BINOP_impl(node, context, np.logical_or, 'or')


def NEGATION_impl(node, context):
    argument = node.get_parameters()
    state, context = argument(context).pop('select')
    selection = state.select
    data = context.get_data()
    result = [~(nw.col(e)) for e in selection]
    col_names = [f'not_{c}' for c in selection]
    data = data.with_columns([
        val.alias(col)
        for val, col in zip(result, col_names)
    ])
    return context.update_state(data=data, select=col_names)


def UNION_REDUCE_impl(node, context):
    argument = node.get_parameters()
    state, context = argument(context).pop('select')
    selection = state.select
    data = context.get_data()
    result = [reduce(operator.or_, (nw.col(e) for e in selection))]
    col_names = [f"any_{'_or_'.join(selection)}"]
    data = data.with_columns([
        val.alias(col)
        for val, col in zip(result, col_names)
    ])
    return context.update_state(data=data, select=col_names)


def INTERSECTION_REDUCE_impl(node, context):
    argument = node.get_parameters()
    state, context = argument(context).pop('select')
    selection = state.select
    data = context.get_data()
    result = [reduce(operator.and_, (nw.col(e) for e in selection))]
    col_names = [f"all_{'_and_'.join(selection)}"]
    data = data.with_columns([
        val.alias(col)
        for val, col in zip(result, col_names)
    ])
    return context.update_state(data=data, select=col_names)


def INDICATOR_impl(node, context):
    """
    This currently is an identity operation, but it might in the future be
    used to materialize boolean expressions into the DataFrame.
    """
    expr = node.get_parameters()
    return expr(context)


def init_hook(
    ast: Primitive,
    context: DataFrameContext,
    data: IntoFrameT,
) -> Tuple[Primitive, DataFrameContext]:
    try:
        data = nw.from_native(data)
    except TypeError:
        raise ValueError(f"Invalid input type: {type(data)}")
    if 'index' not in data:
        data = data.with_row_index()
    context = context.update_state(data=data)
    return ast, context


def finalise_hook(
    context: DataFrameContext,
) -> DataFrameContext:
    result = context.get_result()
    if result is None:
        state, context = context.pop('data', 'select')
        data, selection = state.data, state.select
        result = data.select(selection)
    return context.with_result(nw.to_native(result))


INTERPRETERS.register_interpreter('nw')
INTERPRETERS.register_operation('__all__', 'CONCATENATE', CONCATENATE_impl)
INTERPRETERS.register_operation('__all__', 'POWER', POWER_impl)
INTERPRETERS.register_operation('__all__', 'BACKDIFF', BACKDIFF_impl)
INTERPRETERS.register_operation('__all__', 'VARIABLE', VARIABLE_impl)
INTERPRETERS.register_operation('__all__', 'RANGE', RANGE_impl)
INTERPRETERS.register_operation('__all__', 'ENUM', ENUM_impl)
INTERPRETERS.register_operation('__all__', 'INDICATOR', INDICATOR_impl)
INTERPRETERS.register_operation('__all__', 'CONDITION_EQUAL', CONDITION_EQUAL_impl)
INTERPRETERS.register_operation('__all__', 'CONDITION_NOT_EQUAL', CONDITION_NOT_EQUAL_impl)
INTERPRETERS.register_operation('__all__', 'CONDITION_GREATER', CONDITION_GREATER_impl)
INTERPRETERS.register_operation('__all__', 'CONDITION_LESS', CONDITION_LESS_impl)
INTERPRETERS.register_operation('__all__', 'CONDITION_GREATER_EQUAL', CONDITION_GREATER_EQUAL_impl)
INTERPRETERS.register_operation('__all__', 'CONDITION_LESS_EQUAL', CONDITION_LESS_EQUAL_impl)
INTERPRETERS.register_operation('__all__', 'UNION', UNION_impl)
INTERPRETERS.register_operation('__all__', 'INTERSECTION', INTERSECTION_impl)
INTERPRETERS.register_operation('__all__', 'NEGATION', NEGATION_impl)
INTERPRETERS.register_operation('__all__', 'UNION_REDUCE', UNION_REDUCE_impl)
INTERPRETERS.register_operation('__all__', 'INTERSECTION_REDUCE', INTERSECTION_REDUCE_impl)


def get_processor():
    processor = TransformProcessor(
        grammar=MinimalGrammar(),
        preprocessors=(confound_formula_preprocessor(),),
        postprocessors=(
            ppr_associative_flatten,
            ppr_common_subexpression,
        ),
        interpreters=INTERPRETERS,
        context_class=DataFrameContext,
        default_interpreter='nw',
    )
    processor.register_initialisation('nw', init_hook)
    processor.register_finalisation('nw', finalise_hook)
    return processor
