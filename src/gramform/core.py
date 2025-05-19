# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
`gramform`
~~~~~~~~~~
Core components of the `gramform` library for building simple DSLs.
"""
import dataclasses
import inspect
import re
from typing import Any, Mapping, Tuple, Type

import ply.lex as lex
import ply.yacc as yacc
import wadler_lindig as wl


@dataclasses.dataclass(frozen=True)
class Grammar:
    """Subclass this and follow the `ply` tutorial
    (https://www.dabeaz.com/ply/ply.html)
    to create a new grammar."""

    @classmethod
    def __lexer__(cls, **params):
        return lex.lex(module=cls, **params)

    @classmethod
    def __parser__(cls, **params):
        lexer = lex.lex(module=cls, **params)
        return yacc.yacc(module=cls, **params)


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
        cache_hit = context.cache.get(self, NotInCache())
        match cache_hit:
            case NotInCache():
                result = context.interpreter[self.name](self, context)
            case NotEvaluated():
                result = context.interpreter[self.name](self, context)
                result = result.write(
                    cache={
                        **context.cache,
                        self: {
                            k: v
                            for k, v in zip(
                                result.cache_vars,
                                result.read(*tuple(result.cache_vars.keys())),
                            )
                        },
                    }
                )
            case _:  # Cache hit
                update = {
                    var: combine(context.read(var)[0], cache_hit[var])
                    for var, combine in context.cache_vars.items()
                }
                result = context.write(**update)
        return result


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

    def __call__(self, context):
        context = context.write(eval=self.value)
        return context


@dataclasses.dataclass(frozen=True)
class NotEvaluated:
    """Sentinel value for unevaluated primitives."""
    pass


@dataclasses.dataclass(frozen=True)
class NotInCache:
    """Sentinel value for primitives not in the cache."""
    pass


@dataclasses.dataclass(frozen=True, kw_only=True)
class TransformationContext:
    cache: Mapping[Primitive, Any] = dataclasses.field(default_factory=dict)

    def prepare_cache(self, primitive: Primitive):
        self.cache[primitive] = NotEvaluated()
        return self

    def get_cache(self, primitive: Primitive):
        return self.cache.get(primitive, NotInCache())


@dataclasses.dataclass(frozen=True, kw_only=True)
class ExecutionContext:
    interpreter: Mapping[str, callable]
    eval: Any = None
    cache: Mapping[Primitive, Any] = dataclasses.field(default_factory=dict)
    cache_vars: Mapping[str, callable] = dataclasses.field(
        default_factory=dict
    )

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
            dataclasses.replace(
                self,
                **{
                    key: (
                        inspect.signature(
                            self.__init__
                        ).parameters[key].default
                        if key in self.__dict__
                        else None
                    )
                    for key in pparams
                },
            ),
        )

    def __repr__(self):
        return wl.pformat(self)

    @classmethod
    def eval_head(self) -> str | None:
        return None


# It's not really frozen when we keep changing the mutable fields, is it?
@dataclasses.dataclass(frozen=True)
class InterpretersDispatch:
    interpreters: Mapping[str, Mapping[str, callable]] = dataclasses.field(
        default_factory=dict
    )
    groups: Mapping[str, Tuple[str, ...]] = dataclasses.field(
        default_factory=dict
    )

    def __post_init__(self):
        self.groups['__all__'] = tuple(self.interpreters.keys())

    def register_interpreter(self, name: str):
        self.interpreters[name] = {}
        self.groups['__all__'] = tuple(self.interpreters.keys())

    def register_operation(
        self,
        interpreter: str,
        operation: str,
        impl: callable,
    ):
        if interpreter in self.groups:
            for _interpreter in self.groups[interpreter]:
                self.interpreters[_interpreter][operation] = impl
        else:
            self.interpreters[interpreter][operation] = impl

    def register_group(self, name: str, interpreters: Tuple[str, ...]):
        self.groups[name] = interpreters

    def __getitem__(self, key: str) -> Mapping[str, callable]:
        return self.interpreters[key]

    def __iter__(self):
        return iter(self.interpreters)

    def __len__(self):
        return len(self.interpreters)

    def __repr__(self):
        return wl.pformat(self)


@dataclasses.dataclass(frozen=True)
class Processor:
    grammar: Type[Grammar]
    preprocessors: Tuple[Mapping[str, str] | callable, ...]
    postprocessors: Tuple[callable, ...]
    interpreters: InterpretersDispatch
    execution_context: Type[ExecutionContext]
    default_interpreter: str | None = None

    def __post_init__(self):
        postprocessors = self.postprocessors
        if ppr_execution_head not in postprocessors:
            postprocessors = list(postprocessors) + [ppr_execution_head]
        object.__setattr__(self, 'postprocessors', tuple(postprocessors))

    def __repr__(self):
        return wl.pformat(self)

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

    def _postprocess(
        self,
        expr: Primitive,
        context: TransformationContext,
    ) -> Tuple[Primitive, TransformationContext]:
        for postprocessor in self.postprocessors:
            expr, context = postprocessor(expr, context)
        return expr, context

    def process(self, expr: str) -> Tuple[Primitive, TransformationContext]:
        expr = self._preprocess(expr)
        expr = self._parse(expr)
        context = TransformationContext()
        expr, context = self._postprocess(expr, context)
        return expr, context

    def __call__(self, expr: str, **params) -> Primitive:
        ast, t_context = self.process(expr)
        if 'context' in params:
            context = params.pop('context')
        else:
            eval_head = self.execution_context.eval_head()
            eval_head = (
                {'eval': params.pop(eval_head, None)}
                if eval_head is not None
                else {}
            )
            interpreter = params.pop(
                'interpreter',
                self.default_interpreter,
            )
            orig_params = tuple(params.keys())
            parameter_names = inspect.signature(
                self.execution_context
            ).parameters
            context_params = {
                e: params.pop(e)
                for e in orig_params
                if e in parameter_names
            }
            context = self.execution_context(
                **context_params,
                **eval_head,
                interpreter=self.interpreters[interpreter],
            )
        context = context.write(**{
            k: v
            for k, v in t_context.__dict__.items()
            if k not in context_params
        })
        result = ast(context, **params)
        if result.eval is not None:
            result = result.eval
        return result


def ppr_execution_head(
    tree: Primitive,
    context: TransformationContext,
) -> Tuple[Primitive, TransformationContext]:
    if tree.name != 'EXECUTION_HEAD':
        tree = EXECUTION_HEAD.bind(tree)
    return tree, context


EXECUTION_HEAD = Primitive('EXECUTION_HEAD')
