# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
DataFrames
~~~~~~~~~~
Transformations for DataFrame operations.
"""
import dataclasses
import os
from typing import Any, Mapping, Iterable
try:
    import pandas as pd
except ImportError:
    pass
try:
    import polars as pl
except ImportError:
    pass

from gramform.core import ExecutionContext, InterpretersDispatch, Processor
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
    context = context.write('eval', tuple(range(start, end + 1)))
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
        data[new_columns] = arg ** pow
        new_selection.extend(new_columns)
    context = context.write(data=data, select=new_selection)
    return context


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
    if context.eval:
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
INTERPRETERS.register_operation('__all__', 'VARIABLE', VARIABLE_impl)
INTERPRETERS.register_operation('__all__', 'LITERAL', LITERAL_impl)
INTERPRETERS.register_operation('__all__', 'RANGE', RANGE_impl)
INTERPRETERS.register_operation('__all__', 'EXECUTION_HEAD', EXEC_impl)


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
        '((x+y)^^2 + (x+y)^^2)',
        data=pd.DataFrame(
            {'x': [1, 2, 3], 'y': [4, 5, 6]},
            index=[1, 2, 3],
        ),
    )
    breakpoint()


if __name__ == '__main__':
    main()
