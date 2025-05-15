# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
`gramform`
~~~~~~~~~~
Core components of the `gramform` library for building simple DSLs.
"""
import dataclasses
import re
from typing import Any, Mapping, Tuple, Type

import wadler_lindig as wl


@dataclasses.dataclass(frozen=True)
class Grammar:
    """Subclass this and follow the `ply` tutorial
    (https://www.dabeaz.com/ply/ply.html)
    to create a new grammar."""
    pass


@dataclasses.dataclass(frozen=True)
class Primitive:
    name: str
    parameters: Tuple[Any, ...] = dataclasses.field(
        default_factory=tuple
    )
    # Used for operator flattening when postprocessing the tree.
    is_associative: bool = False
    is_terminal: bool = dataclasses.field(default=False, repr=False)

    def bind(self, *pparams):
        pparams = pparams or ()
        return type(self)(
            name=self.name,
            parameters=tuple(pparams),
            is_associative=self.is_associative,
            is_terminal=self.is_terminal,
        )

    def __repr__(self):
        return wl.pformat(self)

    def __eq__(self, other):
        return self.name == other.name and self.parameters == other.parameters

    def __hash__(self):
        return hash((self.name, self.parameters))

    def __call__(self, context):
        return context.interpreter[self.name](self, context).eval


@dataclasses.dataclass(frozen=True)
class Literal:
    value: Any = None
    dtype: Type | None = None

    def __post_init__(self):
        if self.dtype is None:
            object.__setattr__(self, 'dtype', type(self.value))

    @classmethod
    def create(cls, *pparams):
        value, dtype = pparams
        return cls(value=value, dtype=dtype)

    @property
    def is_terminal(self) -> bool:
        return True

    def __repr__(self):
        return wl.pformat(self)

    def __eq__(self, other):
        return self.value == other.value and self.dtype == other.dtype

    def __hash__(self):
        return hash((self.value, self.dtype))


@dataclasses.dataclass(frozen=True)
class ExecutionContext:
    interpreter: Mapping[str, callable]
    data: Any #pd.DataFrame
    selection: tuple[str, ...] = dataclasses.field(default_factory=tuple)
    eval: Any = None

    def read(self, *pparams):
        return tuple(getattr(self, key) for key in pparams)

    def write(self, *pparams, **params):
        if pparams:
            if len(pparams) != 2:
                raise ValueError(
                    "Positional parameters to context.write must be a key "
                    f"and value: {pparams}"
                )
            key, value = pparams
            params = {
                **{key: value},
                **params,
            }
        return dataclasses.replace(self, **params)

    def pop(self, *pparams):
        return (
            self.read(*pparams),
            dataclasses.replace(self, **{key: None for key in pparams}),
        )


@dataclasses.dataclass(frozen=True)
class Processor:
    grammar: Grammar
    preprocessors: Tuple[Mapping[str, str] | callable, ...]
    postprocessors: Tuple[callable, ...]
    interpreters: Mapping[str, Mapping[str, callable]]

    def __post_init__(self):
        postprocessors = self.postprocessors
        if ppr_execution_head not in postprocessors:
            postprocessors = list(postprocessors) + [ppr_execution_head]
        object.__setattr__(self, 'postprocessors', tuple(postprocessors))

    def _preprocess(self, expr: str) -> str:
        for preprocessor in self.preprocessors:
            if isinstance(preprocessor, Mapping):
                expr = re.sub(
                    rf'\b({"|".join(re.escape(key) for key in preprocessor)})',
                    lambda m: preprocessor[m.group(0)],
                    expr,
                )
            else:
                expr = preprocessor(expr)
        return expr

    def _parse(self, expr: str) -> Primitive:
        parser = self.grammar.__parser__()
        return parser.parse(expr)

    def _postprocess(self, expr: Primitive) -> Primitive:
        for postprocessor in self.postprocessors:
            expr = postprocessor(expr)
        return expr

    def process(self, expr: str) -> Primitive:
        expr = self._preprocess(expr)
        expr = self._parse(expr)
        expr = self._postprocess(expr)
        return expr

    def __call__(self, expr: str, **params) -> Primitive:
        ast = self.process(expr)
        if 'context' in params:
            context = params['context']
        else:
            context = ExecutionContext(**params)
        return ast(context)


EXECUTION_HEAD = Primitive('EXECUTION_HEAD')


def ppr_execution_head(tree):
    if tree.name != 'EXECUTION_HEAD':
        tree = EXECUTION_HEAD.bind(tree)
    return tree


def init_interpreters():
    # TODO: This closure is gonna bite us in the ass if we want to support
    # parallel execution of grammars.
    INTERPRETERS = {}

    def register_interpreter(name):
        INTERPRETERS[name] = {}

    def register_operation(interpreter: str, operation: str, impl: callable):
        INTERPRETERS[interpreter][operation] = impl

    return INTERPRETERS, register_interpreter, register_operation
