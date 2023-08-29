# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Data tags
~~~~~~~~~
Transformations and grammar for operations on data tags.
"""
from dataclasses import dataclass, field
from functools import reduce
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from .grammar import (
    Grammar,
    Grouping,
    GroupingPool,
    LeafInterpreter,
    Literalisation,
    TransformPool,
    TransformPrimitive,
)


@dataclass(frozen=True)
class DataTagGrammar(Grammar):
    groupings: GroupingPool = GroupingPool(
        Grouping(open='(', close=')'),
    )
    transforms: TransformPool = field(
        default_factory=lambda: TransformPool(
            UnionNode(),
            IntersectionNode(),
            ComplementNode(),
            ExclusiveOrNode(),
        )
    )
    whitespace: bool = False
    default_interpreter: Optional[LeafInterpreter] = field(
        default_factory=lambda: TagSelectInterpreter()
    )
    default_root_transform: Optional[TransformPrimitive] = field(
        default_factory=lambda: ReturnSelected()
    )


@dataclass(frozen=True)
class TagSelectInterpreter(LeafInterpreter):
    def __call__(self, leaf: str) -> Callable:
        def select_by_tag(
            tags: Mapping[str, Sequence[str]],
            keys: Sequence[str],
        ) -> Tuple[Mapping[str, Any], Sequence[str]]:
            return tags[leaf], keys

        return select_by_tag


@dataclass(frozen=True)
class ReturnSelected(TransformPrimitive):
    min_arity: int = 1
    max_arity: int = 1
    priority: int = float('inf')
    associative: bool = False
    commutative: bool = False
    literals: Sequence[Literalisation] = ()

    def __call__(self, *pparams, **params) -> Callable:
        f = pparams[0]

        def return_selected(
            arg: Any,
            **datatypes,
        ) -> Mapping[str, Any]:
            keys = set(datatypes.keys())
            return {k: datatypes[k] for k in f(arg, keys)[0]}

        return return_selected


# ---------------------------------- Union --------------------------------- #


@dataclass(frozen=True)
class UnionInfixLiteralisation(Literalisation):
    affix: Literal['prefix', 'suffix', 'infix', 'leaf'] = 'infix'
    regex: str = r'\|'

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params


@dataclass(frozen=True)
class UnionNode(TransformPrimitive):
    min_arity: int = 2
    max_arity: int = float('inf')
    priority: int = 4
    associative: bool = True
    commutative: bool = True
    literals: Sequence[Literalisation] = (UnionInfixLiteralisation(),)

    def ascend(self, *pparams, **params) -> Callable:
        def union(
            arg: Any,
            keys: Sequence[str],
        ) -> Tuple[Mapping[str, Any], Sequence[str]]:
            arg = tuple(set(f(arg, keys)[0]) for f in pparams)
            arg = reduce((lambda x, y: x | y), arg)
            return arg, keys

        return union


# ------------------------------ Intersection ------------------------------ #


@dataclass(frozen=True)
class IntersectionInfixLiteralisation(Literalisation):
    affix: Literal['prefix', 'suffix', 'infix', 'leaf'] = 'infix'
    regex: str = r'\&'

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params


@dataclass(frozen=True)
class IntersectionNode(TransformPrimitive):
    min_arity: int = 2
    max_arity: int = float('inf')
    priority: int = 2
    associative: bool = True
    commutative: bool = True
    literals: Sequence[Literalisation] = (IntersectionInfixLiteralisation(),)

    def ascend(self, *pparams, **params) -> Callable:
        def intersection(
            arg: Any,
            keys: Sequence[str],
        ) -> Tuple[Mapping[str, Any], Sequence[str]]:
            arg = tuple(set(f(arg, keys)[0]) for f in pparams)
            print(arg)
            arg = reduce((lambda x, y: x & y), arg)
            return arg, keys

        return intersection


# ------------------------------- Complement ------------------------------- #


@dataclass(frozen=True)
class ComplementPrefixExclLiteralisation(Literalisation):
    affix: Literal['prefix', 'suffix', 'infix', 'leaf'] = 'prefix'
    regex: str = r'\!'

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params


@dataclass(frozen=True)
class ComplementPrefixTildeLiteralisation(Literalisation):
    affix: Literal['prefix', 'suffix', 'infix', 'leaf'] = 'prefix'
    regex: str = r'\~'

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params


@dataclass(frozen=True)
class ComplementNode(TransformPrimitive):
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 1
    literals: Sequence[Literalisation] = (
        ComplementPrefixExclLiteralisation(),
        ComplementPrefixTildeLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:
        f = pparams[0]

        def complement(
            arg: Any,
            keys: Sequence[str],
        ) -> Tuple[Mapping[str, Any], Sequence[str]]:
            return set(keys) - set(f(arg, keys)[0]), keys

        return complement


# ------------------------------ Exclusive Or ------------------------------ #


@dataclass(frozen=True)
class ExclusiveOrInfixLiteralisation(Literalisation):
    affix: Literal['prefix', 'suffix', 'infix', 'leaf'] = 'infix'
    regex: str = r'\^'

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params


@dataclass(frozen=True)
class ExclusiveOrNode(TransformPrimitive):
    min_arity: int = 2
    max_arity: int = float('inf')
    priority: int = 3
    associative: bool = True
    commutative: bool = True
    literals: Sequence[Literalisation] = (ExclusiveOrInfixLiteralisation(),)

    def ascend(self, *pparams, **params) -> Callable:
        def xor(
            arg: Any,
            keys: Sequence[str],
        ) -> Tuple[Mapping[str, Any], Sequence[str]]:
            arg = tuple(set(f(arg, keys)[0]) for f in pparams)
            arg = reduce((lambda x, y: x ^ y), arg)
            return arg, keys

        return xor
