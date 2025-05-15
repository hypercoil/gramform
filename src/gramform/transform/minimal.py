# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
DataFrames
~~~~~~~~~~
Transformations for DataFrame operations.
"""
import dataclasses
from typing import Any, Mapping
try:
    import pandas as pd
except ImportError:
    pass
try:
    import polars as pl
except ImportError:
    pass

from gramform.core import init_interpreters

(
    INTERPRETERS,
    register_interpreter,
    register_operation,
) = init_interpreters()


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
    for child in node.parameters:
        context = child(context)
    # This is actually a no-op, because all children have already been
    # written to the selection buffer by the time we get here.
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
    for pow in pow_order:
        new_selection.append(f'{selection}_power{pow}')
        data[new_selection[-1]] = data[selection] ** pow
    context = context.write(data=data, select=new_selection)
    return context


def EXEC_impl(node, context):
    exec_mode, interpreter, expr = node.parameters
    (input,), context = context.pop('data')
    if exec_mode == 'file':
        data = pd.read_csv(input)
    elif exec_mode == 'df':
        data = input
    else:
        raise ValueError(f"Invalid exec mode: {exec_mode}")
    interpreter = INTERPRETERS[interpreter]
    context = context.write(data=data, interpreter=interpreter)
    tree = node.parameters[0]
    context = tree(context)
    if context.eval:
        result, context = context.pop('eval')
    else:
        (data, selection), context = context.pop('data', 'select')
        result = data[selection]
    return context.write(eval=result)


register_interpreter('pd')
register_interpreter('pl')
register_operation('pd', 'CONCATENATE', CONCATENATE_impl)
register_operation('pl', 'CONCATENATE', CONCATENATE_impl)
register_operation('pd', 'POWER', POWER_impl)
register_operation('pl', 'POWER', POWER_impl)
register_operation('pd', 'VARIABLE', VARIABLE_impl)
register_operation('pl', 'VARIABLE', VARIABLE_impl)
register_operation('pd', 'LITERAL', LITERAL_impl)
register_operation('pl', 'LITERAL', LITERAL_impl)
register_operation('pd', 'RANGE', RANGE_impl)
register_operation('pl', 'RANGE', RANGE_impl)


def main():
    breakpoint()


if __name__ == '__main__':
    main()
