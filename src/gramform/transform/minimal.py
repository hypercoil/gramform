# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
DataFrames
~~~~~~~~~~
Transformations for DataFrame operations.
"""
import dataclasses
import operator
import os
from itertools import chain
from typing import Any, Mapping, Iterable, Literal, Sequence

import numpy as np
try:
    import pandas as pd
except ImportError:
    pass
try:
    import polars as pl
except ImportError:
    pass
import wadler_lindig as wl

from gramform.core import (
    ExecutionContext,
    InterpretersDispatch,
    Processor,
)
from gramform.grammar.minimal import (
    MinimalGrammar,
    confound_formula_preprocessor,
)
from gramform.postprocessors import (
    ppr_associative_flatten,
    ppr_common_subexpression,
)

INTERPRETERS = InterpretersDispatch()


def VARIABLE_impl(node, context):
    name, = node.parameters
    data, selection = context.read('data', 'select')
    if name not in data:
        raise ValueError(f"Variable {name} not found in data")
    selection.append(name)
    context = context.write('select', selection)
    return context


def LITERAL_impl(node, context):
    value, dtype = node.parameters
    context = context.write('eval', value)
    return context


def RANGE_impl(node, context):
    start, end = node.parameters
    (start,), context = start(context).pop('eval')
    (end,), context = end(context).pop('eval')
    context = context.write('eval', range(start, end + 1))
    return context


def ENUM_impl(node, context):
    eval = []
    for child in node.parameters:
        (new_eval,), context = child(context).pop('eval')
        if not isinstance(new_eval, Iterable):
            new_eval = (new_eval,)
        eval.extend(new_eval)
    context = context.write(eval=chain(eval))
    return context


def CONCATENATE_impl(node, context):
    (selection,), context = context.pop('select')
    for child in node.parameters:
        (new_selection,), context = child(context).pop('select')
        selection.extend(new_selection)
    context = context.write(select=selection)
    return context


def POWER_impl(node, context):
    argument, power = node.parameters
    context = argument(context)
    (data, selection), context = context.pop('data', 'select')
    context = power(context)
    (pow_order, pow_cols), context = context.pop('eval', 'select')
    if pow_cols:
        raise ValueError("Power operation does not support column selection")
    new_selection = []
    arg = data[selection]
    if not isinstance(pow_order, Iterable):
        pow_order = (pow_order,)
    for pow in pow_order:
        if pow == 1:
            new_selection.extend(selection)
            continue
        new_columns = [f'{e}_power{pow}' for e in selection]
        data[new_columns] = (arg ** pow).to_numpy()
        new_selection.extend(new_columns)
    context = context.write(data=data, select=new_selection)
    return context


def BACKDIFF_impl(node, context):
    argument, order = node.parameters
    context = argument(context)
    (data, selection), context = context.pop('data', 'select')
    context = order(context)
    (order, order_cols), context = context.pop('eval', 'select')
    if order_cols:
        raise ValueError("Backdiff operation does not support column selection")
    new_selection, result = [], {}
    arg = data[selection]
    if not isinstance(order, Iterable):
        order = (order,)
    required_orders = set(order)
    max_order = max(tuple(order))
    for ord in range(max_order + 1):
        arg = arg.diff()
        if ord in required_orders:
            result[ord] = arg
    for ord in order:
        if ord == 0:
            new_selection.extend(selection)
            continue
        new_columns = [f'{c}_derivative{ord}' for c in selection]
        data[new_columns] = result[ord].to_numpy()
        new_selection.extend(new_columns)
    context = context.write(data=data, select=new_selection)
    return context


def BINOP_impl(node, context, op: callable, col_infix: str):
    left, right = node.parameters
    context = left(context)
    (left_selection, left_eval), context = context.pop('select', 'eval')
    context = right(context)
    (right_selection, right_eval), context = context.pop('select', 'eval')
    (data,) = context.read('data')
    if left_selection is not None:
        left_arg = data[left_selection]
        left_cols = left_selection
    else:
        left_arg = left_eval
        left_cols = [f'{left_eval}']
    if right_selection is not None:
        right_arg = data[right_selection]
        right_cols = right_selection
    else:
        right_arg = right_eval
        right_cols = [f'{right_eval}']
    result = op(left_arg.to_numpy(), right_arg.to_numpy())
    new_columns = [
        f"{lcol}_{col_infix}_{rcol}"
        for lcol in left_cols
        for rcol in right_cols
    ]
    data[new_columns] = pd.DataFrame(result, index=data.index)
    if data.isnull().any().any():
        raise ValueError("Result of binary operation contains NaN")
    return context.write(data=data, select=new_columns)


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
    argument, = node.parameters
    context = argument(context)
    (selection, eval), context = context.pop('select', 'eval')
    (data,) = context.read('data')
    if selection is not None:
        result = ~(data[selection])
        col_names = [f'not_{c}' for c in selection]
    else:
        result = ~eval
        col_names = [f'not_{c}' for c in eval]
    data[col_names] = pd.DataFrame(result, index=data.index)
    return context.write(data=data, select=col_names)


def UNION_REDUCE_impl(node, context):
    argument, = node.parameters
    context = argument(context)
    (selection, eval), context = context.pop('select', 'eval')
    (data,) = context.read('data')
    if selection is not None:
        result = data[selection].any(axis=1)
        col_names = [f"any_{'_or_'.join(selection)}"]
    else:
        result = eval.any(axis=1)
        col_names = [f"any_{'_or_'.join(eval)}"]
    data[col_names] = pd.DataFrame(result, index=data.index)
    return context.write(data=data, select=col_names)


def INTERSECTION_REDUCE_impl(node, context):
    argument, = node.parameters
    context = argument(context)
    (selection, eval), context = context.pop('select', 'eval')
    (data,) = context.read('data')
    if selection is not None:
        result = data[selection].all(axis=1)
        col_names = [f"all_{'_and_'.join(selection)}"]
    else:
        result = eval.all(axis=1)
        col_names = [f"all_{'_and_'.join(eval)}"]
    data[col_names] = pd.DataFrame(result, index=data.index)
    return context.write(data=data, select=col_names)


def INDICATOR_impl(node, context):
    """
    This currently is an identity operation, but it might in the future be
    used to materialize boolean expressions into the DataFrame.
    """
    expr, = node.parameters
    return expr(context)


def EXEC_impl(node, context):
    tree, = node.parameters
    (exec_mode,), context = context.pop('eval')
    (input,), context = context.pop('data')
    if exec_mode == 'df' or isinstance(input, pd.DataFrame):
        data = input
    elif exec_mode == 'file' or os.path.isfile(input):
        data = pd.read_csv(input)
    else:
        raise ValueError(f"Invalid exec mode: {exec_mode}")
    context = context.write(data=data)
    context = tree(context)
    if context.eval is not None:
        result, context = context.pop('eval')
    else:
        (data, selection), context = context.pop('data', 'select')
        result = data[selection]
    return context.write(eval=result)


@dataclasses.dataclass(frozen=True, kw_only=True, repr=False)
class DataFrameContext(ExecutionContext):
    data: pd.DataFrame
    select: list[str] = dataclasses.field(default_factory=list)
    cache_vars: Mapping[str, callable] = dataclasses.field(
        default_factory=lambda: {
            'select': lambda in_context, in_cache: in_cache
        }
    )

    @classmethod
    def eval_head(self) -> str | None:
        return 'exec_mode'


INTERPRETERS.register_interpreter('pd')
INTERPRETERS.register_interpreter('pl')
INTERPRETERS.register_operation('__all__', 'CONCATENATE', CONCATENATE_impl)
INTERPRETERS.register_operation('__all__', 'POWER', POWER_impl)
INTERPRETERS.register_operation('__all__', 'BACKDIFF', BACKDIFF_impl)
INTERPRETERS.register_operation('__all__', 'VARIABLE', VARIABLE_impl)
INTERPRETERS.register_operation('__all__', 'LITERAL', LITERAL_impl)
INTERPRETERS.register_operation('__all__', 'RANGE', RANGE_impl)
INTERPRETERS.register_operation('__all__', 'ENUM', ENUM_impl)
INTERPRETERS.register_operation('__all__', 'EXECUTION_HEAD', EXEC_impl)
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


def main():
    processor = Processor(
        grammar=MinimalGrammar,
        preprocessors=(confound_formula_preprocessor(),),
        postprocessors=(
            ppr_associative_flatten,
            ppr_common_subexpression,
        ),
        interpreters=INTERPRETERS,
        execution_context=DataFrameContext,
        default_interpreter='pd',
    )
    result = processor.process('d_[1]((x+y)^^2 + (x+y)^^2)')
    result = processor(
        'dd_[3]((x+y)^2,4-5 + (x+y)^2,4-5)',
        data=pd.DataFrame(
            {'x': [1, 2, 3], 'y': [4, 5, 6]},
            index=[1, 2, 3],
        ),
    )
    result = processor(
        'NOT_((x=y && x=z) || x>=w) + AND_(I_[x=y] + I_[x=z] + OR_(I_[x=w] + I_[x=v]))',
        data=pd.DataFrame(
            {'x': [1, 2, 3], 'y': [3, 2, 1], 'z': [2, 2, 2], 'w': [0, 2, 3], 'v': [1, 0, 0]},
            index=[1, 2, 3],
        ),
    )
    breakpoint()


if __name__ == '__main__':
    main()
