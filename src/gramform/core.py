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
from collections import namedtuple
from functools import lru_cache
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
)
from enum import Enum, auto

import ply.lex as lex
import ply.yacc as yacc
import wadler_lindig as wl
from pydantic import BaseModel, ConfigDict, Field

from .error import GrammarErrorHandler
from .resampler import generate_valid_completion


@lru_cache(maxsize=None)
def precedence_from_sequence(
    sequence: Tuple[str | Tuple[str, ...], ...],
    default: int = 0,
) -> Callable[[str], int]:
    sequence = tuple(sequence)
    precedence = {}
    for i, token in enumerate(sequence):
        if isinstance(token, str):
            precedence[token] = i
        elif isinstance(token, tuple):
            for t in token:
                precedence[t] = i

    before_index_default, after_index_default = 0, len(sequence)

    def from_sequence(token: str) -> int:
        return precedence.get(token, default)

    def with_precedence(
        after: str | None = None,
        before: str | None = None,
    ) -> int:
        if before is None and after is None:
            raise ValueError("Either before or after must be provided")
        before_index, after_index = before_index_default, after_index_default
        if before is not None:
            before_index = from_sequence(before)
        if after is not None:
            after_index = from_sequence(after)
        if after_index < before_index:
            raise ValueError(
                f"Precedence after {after} and before {before} is not "
                f"satisfiable because {after} has a lower precedence than "
                f"{before}."
            )
        precedence[token] = (after_index + before_index) / 2
        return from_sequence(token)

    return from_sequence, with_precedence


def push_state_and_return(state: str):
    def _inner(t, grammar):
        t.lexer.push_state(state)
        return t
    return _inner


def pop_state_and_return(t, grammar):
    t.lexer.pop_state()
    return t


def literal(dtype: Type, cast: callable = None):
    """
    A production rule pattern used for literals.

    Pattern:
    construct : LITERAL
    """
    def _inner(value):
        _cast = cast or dtype
        return Literal.create(_cast(value), dtype)
    return _inner


def lift_literal(
    dtype: Type,
    prim: "Primitive",
    cast: callable = None,
):
    """
    A production rule pattern used for lifting literals.

    Pattern:
    construct : LITERAL
    """
    def _inner(value):
        _cast = cast or dtype
        return prim.bind(Literal.create(_cast(value), dtype))
    return _inner


def unop_prefix(prim: "Primitive", *pparams):
    """
    A production rule pattern frequently used for unary prefix operations.

    Pattern:
    construct : OPERATOR construct
    """
    def _inner(_, right):
        return prim.bind(right, *pparams)
    return _inner


def unop_postfix(prim: "Primitive", *pparams):
    """
    A production rule pattern frequently used for unary postfix operations.

    Pattern:
    construct : construct OPERATOR
    """
    def _inner(left, _):
        return prim.bind(left, *pparams)
    return _inner


def binop_infix(prim: "Primitive", *pparams):
    """
    A production rule pattern frequently used for binary infix operations.

    Pattern:
    construct : construct_left OPERATOR construct_right
    """
    def _inner(left, _, right):
        return prim.bind(left, right, *pparams)
    return _inner


def binop_prefix(prim: "Primitive", *pparams):
    """
    A production rule pattern frequently used for binary prefix operations.

    Pattern:
    construct : OPERATOR construct_left construct_right
    """
    def _inner(_, left, right):
        return prim.bind(left, right, *pparams)
    return _inner


def binop_postfix(prim: "Primitive", *pparams):
    """
    A production rule pattern frequently used for binary postfix operations.

    Pattern:
    construct : construct_left construct_right OPERATOR
    """
    def _inner(left, right, _):
        return prim.bind(left, right, *pparams)
    return _inner


def enter_group():
    """
    A production rule pattern used for grouping.

    Pattern:
    construct : LPAREN construct RPAREN
    """
    def _inner(_, inner, __):
        return inner
    return _inner


def unit_lift():
    """
    A production rule pattern used for lifting one kind of construct into
    another.

    Pattern:
    construct : construct
    """
    def _inner(inner):
        return inner
    return _inner


def named_function_call(prim: "Primitive", *pparams):
    """
    A production rule pattern frequently used for function calls.

    Pattern:
    construct : NAME LPAREN construct RPAREN
    """
    def _inner(_, __, expr, ___):
        return prim.bind(expr, *pparams)
    return _inner


def parameterised_named_function_call(prim: "Primitive", *pparams):
    """
    A production rule pattern frequently used for function calls.

    Pattern:
    construct : NAME LPAREN construct parameters RPAREN
    """
    def _inner(_, __, expr, parameters, ___):
        return prim.bind(expr, parameters, *pparams)
    return _inner


def named_function_bind(prim: "Primitive", *pparams):
    """
    A production rule pattern frequently used for function calls.

    Pattern:
    construct : name LPAREN construct RPAREN
    """
    def _inner(name, _, expr, __):
        return prim.bind(name, expr, *pparams)
    return _inner


def parameterised_named_function_bind(prim: "Primitive", *pparams):
    """
    A production rule pattern frequently used for function calls.

    Pattern:
    construct : name LPAREN construct parameters RPAREN
    """
    def _inner(name, _, expr, parameters, __):
        return prim.bind(name, expr, parameters, *pparams)
    return _inner


def config_primitives():
    """
    Configure a registry of primitives.

    Returns
    -------
    constructor: callable
        An alternative constructor for the `Primitive` class that
        automatically includes the primitive in the registry.
    registry: dict
        A registry of primitives.
    """
    registry = {}
    def _inner(
        name: str,
        parameters: Tuple[Any, ...] = (),
        is_associative: bool = False,
        is_terminal: bool = False,
    ):
        if name in registry:
            raise ValueError(f"Primitive {name} already registered")
        prim = Primitive(
            name=name,
            parameters=parameters,
            is_associative=is_associative,
            is_terminal=is_terminal,
        )
        registry[name] = prim
        return prim
    _inner.__doc__ = Primitive.__doc__
    _inner.__name__ = Primitive.__name__
    return _inner, registry


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
        # print(f"Primitive {self.name} bound with {pparams}")
        return type(self)(
            name=self.name,
            parameters=tuple(pparams),
            is_associative=self.is_associative,
            is_terminal=self.is_terminal,
        )

    def get_parameters(self) -> Any:
        if len(self.parameters) == 1:
            return self.parameters[0]
        else:
            return self.parameters

    @property
    def value(self) -> Any:
        #TODO: Handle multiple levels of wrapping
        param = self.parameters[0]
        if (len(self.parameters) == 1) and isinstance(param, Literal):
            return param.value
        else:
            raise ValueError(
                f"Method `value` is not supported on "
                f"non-literal wrapping primitive {self.name}"
            )

    def __repr__(self):
        return wl.pformat(self)

    def __eq__(self, other):
        return self.name == other.name and self.parameters == other.parameters

    def __hash__(self):
        return hash((self.name, self.parameters))

    def __call__(self, context):
        cache_hit = context.subcontexts.get(
            'cache', {}
        ).get(self, NotInCache())
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
        return context.with_result(self.value)


@dataclasses.dataclass(frozen=True)
class NotEvaluated:
    """Sentinel value for unevaluated primitives."""
    pass


@dataclasses.dataclass(frozen=True)
class NotInCache:
    """Sentinel value for primitives not in the cache."""
    pass


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
class ProductionRule:
    """
    A production rule in BNF format.

    Production rules are used to define the grammar of the language. They
    consist of a name, a rule, and an implementation function, which typically
    builds an abstract syntax tree (AST) when the rule is matched.

    Parameters
    ----------
    name: str
        Name of the production function (e.g.,
        `'p_expression_condition_equal'`).
    rule: str
        The BNF rule (e.g.,
        `'expression : expression CONDITION_EQUAL expression'`).
    implementation: callable
        The actual implementation function.
    """
    name: str
    rule: str
    implementation: callable

    def __post_init__(self):
        """Validate the production rule."""
        if ':' not in self.rule:
            raise ValueError(
                "Production rule must contain ':' to separate LHS and RHS"
            )

    def __repr__(self):
        return wl.pformat(self)

    def materialise(self, grammar: 'DynamicGrammar') -> callable:
        """Create a PLY-compatible production function."""
        def production_func(p):
            p[0] = self.implementation(*p[1:])
        name = f'p_{self.name}'
        production_func.__name__ = name
        production_func.__doc__ = self.rule
        return name, production_func


class Associativity(Enum):
    """Associativity of operators."""
    LEFT = auto()
    RIGHT = auto()
    NONE = auto()  # For non-operator tokens


@dataclasses.dataclass
class Namespaces:
    """
    A collection of token namespaces for organising reserved words and other
    token categories.

    A namespace is a mapping from regex patterns to token names, allowing
    for flexible organization of tokens into different categories (e.g.,
    reserved words, operators, literals, etc.).

    Parameters
    ----------
    namespaces: Dict[str, Dict[str, str]]
        A mapping from namespace names to mappings of regex patterns to
        token names. When tokens don't explicitly specify a namespace, they
        are placed in the default namespace for their state; if they don't
        have an explicit state, they are placed in the default namespace for
        the grammar.
    default_namespace: str
        The name of the default namespace for tokens that don't specify
        a namespace explicitly.
    """
    namespaces: Dict[str, Dict[str, str]] = dataclasses.field(
        default_factory=dict
    )
    default_namespace: str = "default"

    def __post_init__(self):
        """Validate the namespaces."""
        if not isinstance(self.namespaces, dict):
            raise ValueError("Namespaces must be a dictionary")
        
        for namespace_name, namespace_mapping in self.namespaces.items():
            if not isinstance(namespace_name, str):
                raise ValueError("Namespace names must be strings")
            if not isinstance(namespace_mapping, dict):
                raise ValueError("Namespace mappings must be dictionaries")
            for regex, token_name in namespace_mapping.items():
                if not isinstance(regex, str):
                    raise ValueError("Regex patterns must be strings")
                if not isinstance(token_name, str):
                    raise ValueError("Token names must be strings")

    def add_token(
        self,
        namespace: str,
        regex: str,
        token_name: str,
    ) -> 'Namespaces':
        """
        Add a token to a specific namespace.

        Parameters
        ----------
        namespace: str
            The namespace to add the token to.
        regex: str
            The regex pattern for the token.
        token_name: str
            The name of the token.

        Returns
        -------
        Namespaces
            A new Namespaces instance with the token added.
        """
        new_namespaces = dict(self.namespaces)
        if namespace not in new_namespaces:
            new_namespaces[namespace] = {}
        new_namespaces[namespace] = {
            **new_namespaces[namespace],
            regex: token_name
        }
        self.namespaces = new_namespaces
        return self

    def add_namespace(
        self,
        namespace: str,
        namespace_mapping: Dict[str, str] | None = None,
    ) -> 'Namespaces':
        """Add a namespace to the collection."""
        new_namespaces = dict(self.namespaces)
        new_namespaces[namespace] = namespace_mapping
        self.namespaces = new_namespaces
        return self

    def get_namespace(self, namespace: str) -> Dict[str, str]:
        """
        Get all tokens in a specific namespace.

        Parameters
        ----------
        namespace: str
            The name of the namespace.

        Returns
        -------
        Dict[str, str]
            A mapping from regex patterns to token names in the namespace.
        """
        return self.namespaces.get(namespace, {})

    def get_all_tokens(self) -> Dict[str, str]:
        """
        Get all tokens across all namespaces as a flat mapping.

        Returns
        -------
        Dict[str, str]
            A mapping from regex patterns to token names across all
            namespaces.
        """
        all_tokens = {}
        for namespace_mapping in self.namespaces.values():
            all_tokens.update(namespace_mapping)
        return all_tokens

    def merge(self, other: 'Namespaces') -> 'Namespaces':
        """
        Merge this namespaces collection with another.

        Parameters
        ----------
        other: Namespaces
            The other namespaces collection to merge with.

        Returns
        -------
        Namespaces
            A new Namespaces instance containing tokens from both collections.
        """
        merged_namespaces = dict(self.namespaces)
        for namespace_name, namespace_mapping in other.namespaces.items():
            if namespace_name in merged_namespaces:
                # Merge the mappings, with other taking precedence on conflicts
                merged_namespaces[namespace_name] = {
                    **merged_namespaces[namespace_name],
                    **namespace_mapping
                }
            else:
                merged_namespaces[namespace_name] = namespace_mapping

        return dataclasses.replace(self, namespaces=merged_namespaces)

    def __repr__(self):
        return wl.pformat(self)

    def __add__(self, other: 'Namespaces') -> 'Namespaces':
        """Merge two namespaces collections."""
        return self.merge(other)

    def __getattr__(self, name: str) -> Dict[str, str]:
        """Get a namespace by name."""
        try:
            namespaces = object.__getattribute__(self, 'namespaces')
        except AttributeError:
            return object.__getattribute__(self, name)
        if name == 'namespaces':
            return namespaces
        if name in namespaces:
            return self.get_namespace(name)
        return object.__getattribute__(self, name)


@dataclasses.dataclass(frozen=True)
class Token:
    """
    A token in the grammar.

    A token is a named regex pattern that can be matched against the input
    string. It can also have an optional function to handle the token, a lexer
    state, a precedence, an associativity, and a category.

    The token name is used to identify the token in the grammar.

    Parameters
    ----------
    name: str
        The name of the token (e.g., 'PLUS').
    regex: str
        The regex pattern for the token (e.g., r'\\+').
    function: Optional[callable]
        An optional function to handle the token.
    state: Optional[str]
        An optional lexer state for the token.
    precedence: int
        The precedence level of the token. A higher number means tighter
        binding.
    associativity: Associativity
        The associativity of the token.
    namespace: Optional[str]
        The namespace this token belongs to. If None, the token will be
        placed in the default namespace for its state. Namespaces can be
        used to group tokens together and disambiguate them, or to maintain
        lists of reserved words.
    category: Optional[str]
        An optional category for error context matching.
    """
    name: str
    regex: str
    function: Optional[callable] = None
    state: Optional[str] = None
    precedence: int = 0
    associativity: Associativity = Associativity.NONE
    namespace: Optional[str] = None
    category: Optional[str] = None

    def __post_init__(self):
        """Validate the token."""
        if not self.name:
            raise ValueError("Token name cannot be empty")
        if not self.regex:
            raise ValueError("Token regex cannot be empty")
        if self.function and not callable(self.function):
            raise ValueError("Token function must be callable")
        if self.state and not isinstance(self.state, str):
            raise ValueError("Token state must be a string")
        if isinstance(self.precedence, Mapping):
            object.__setattr__(
                self,
                'precedence',
                self.precedence.get(self.name, 0),
            )
        elif isinstance(self.precedence, Callable):
            object.__setattr__(
                self,
                'precedence',
                self.precedence(self.name),
            )
        if not isinstance(self.precedence, int):
            raise ValueError(
                "Token precedence must be an integer, a callable, "
                "or a mapping"
            )
        if not isinstance(self.associativity, Associativity):
            raise ValueError(
                "Token associativity must be an Associativity enum"
            )
        if self.namespace is not None and not isinstance(self.namespace, str):
            raise ValueError("Token namespace must be a string or None")

    def generate_example(self) -> str:
        """
        Generate an example value for this token based on its regex pattern.

        This is a crude AI-generated placeholder. We should use a more
        principled approach to generating examples through deterministic
        traversal of the regex AST.
        """
        return generate_valid_completion(self.regex)

    def materialise(
        self,
        grammar: 'DynamicGrammar',
    ) -> Tuple[str, Callable | str]:
        """Create a PLY-compatible token function or regex."""
        if self.state:
            state_name = f'{self.state}_'
        else:
            state_name = 'ANY_'
        name = f't_{state_name}{self.name}'

        if not self.function:
            return name, self.regex

        def token_func(t):
            return self.function(t, grammar)

        func = token_func
        func.__name__ = name
        func.__doc__ = self.regex
        return name, func


@dataclasses.dataclass(frozen=True)
class GrammarComponent:
    """
    A composable component of a grammar that can be merged with others.

    A grammar component is a collection of tokens, states, and production
    rules. It can be used to build a larger grammar by merging with other
    components.

    Parameters
    ----------
    tokens: Tuple[Token, ...]
        The tokens in the component.
    states: Tuple[Tuple[str, str], ...]
        The states in the component.
    production_rules: Tuple[ProductionRule, ...]
        The production rules in the component.
    """
    tokens: Tuple[Token, ...] = dataclasses.field(default_factory=tuple)
    states: Tuple[Tuple[str, str], ...] = dataclasses.field(
        default_factory=tuple,
    )
    production_rules: Tuple[ProductionRule, ...] = dataclasses.field(
        default_factory=tuple,
    )
    _is_built: bool = dataclasses.field(default=False)

    def __post_init__(self):
        """Validate the component's attributes."""
        # Validate tokens
        if not all(isinstance(t, Token) for t in self.tokens):
            raise ValueError("All tokens must be Token instances")

        # Validate states
        if not all(
            isinstance(s, tuple) and len(s) == 2 and
            isinstance(s[0], str) and isinstance(s[1], str)
            for s in self.states
        ):
            raise ValueError("States must be tuples of (str, str)")

        # Validate production rules
        if not all(
            isinstance(rule, ProductionRule)
            for rule in self.production_rules
        ):
            raise ValueError(
                "All production rules must be ProductionRule instances"
            )

    def __repr__(self):
        return wl.pformat(self)

    def build(self) -> 'GrammarComponent':
        """Build the component, making it ready for use in a grammar."""
        if self._is_built:
            raise ValueError("Component is already built")

        return dataclasses.replace(self, _is_built=True)

    def merge(self, other: 'GrammarComponent') -> 'GrammarComponent':
        """Merge this component with another, returning a new component."""
        if not self._is_built or not other._is_built:
            raise ValueError("Both components must be built before merging")

        # Check for production rule conflicts
        self_rules = {rule.name for rule in self.production_rules}
        other_rules = {rule.name for rule in other.production_rules}
        conflicts = self_rules & other_rules
        if conflicts:
            raise ValueError(f"Conflicting production rules: {conflicts}")

        # Check for token name conflicts
        self_tokens = {token.name for token in self.tokens}
        other_tokens = {token.name for token in other.tokens}
        conflicts = self_tokens & other_tokens
        if conflicts:
            raise ValueError(f"Conflicting tokens: {conflicts}")

        # Check for namespace conflicts (same regex in different namespaces)
        #TODO: If this is too slow, we can either refactor to use a more
        #      efficient data structure, or allow unsafe/skipping the check.
        self_namespaces = {}
        for token in self.tokens:
            namespace = token.namespace or token.state or "default"
            if namespace not in self_namespaces:
                self_namespaces[namespace] = {}
            self_namespaces[namespace][token.regex] = token.name

        other_namespaces = {}
        for token in other.tokens:
            namespace = token.namespace or token.state or "default"
            if namespace not in other_namespaces:
                other_namespaces[namespace] = {}
            other_namespaces[namespace][token.regex] = token.name

        # Check for regex conflicts within the same namespace
        for namespace in set(
            self_namespaces.keys()
        ) | set(
            other_namespaces.keys()
        ):
            self_regexes = set(self_namespaces.get(namespace, {}).keys())
            other_regexes = set(other_namespaces.get(namespace, {}).keys())
            conflicts = self_regexes & other_regexes
            if conflicts:
                # Check if the conflicting regexes map to different token
                # names
                for regex in conflicts:
                    self_name = self_namespaces.get(
                        namespace,
                        {},
                    ).get(regex)
                    other_name = other_namespaces.get(
                        namespace,
                        {},
                    ).get(regex)
                    if self_name != other_name:
                        raise ValueError(
                            f"Conflicting regex '{regex}' in namespace "
                            f"'{namespace}': '{self_name}' vs '{other_name}'"
                        )

        return dataclasses.replace(
            self,
            tokens=self.tokens + other.tokens,
            states=self.states + other.states,
            production_rules=self.production_rules + other.production_rules,
            _is_built=True,
        )

    def __add__(self, other: 'GrammarComponent') -> 'GrammarComponent':
        return self.merge(other)


@dataclasses.dataclass(frozen=True)
class DynamicGrammar:
    """A grammar composed from multiple components."""
    components: Tuple[GrammarComponent, ...]
    error_handler: GrammarErrorHandler = dataclasses.field(
        default_factory=GrammarErrorHandler
    )
    productions: Optional[Tuple[ProductionRule, ...]] = dataclasses.field(
        default=None,
        init=False,
    )
    _is_initialized: bool = dataclasses.field(
        default=False,
        init=False,
    )
    _example_cache_file: str = dataclasses.field(
        default='.grammar_examples.json',
        init=False,
    )
    _lexer: Optional[Any] = dataclasses.field(
        default=None,
        init=False,
    )
    _parser: Optional[Any] = dataclasses.field(
        default=None,
        init=False,
    )
    _namespaces: Optional[Namespaces] = dataclasses.field(
        default=None,
        init=False,
    )

    def __post_init__(self):
        """Build the grammar from components."""
        if self._is_initialized:
            return

        # Build all components
        built_components = [c.build() for c in self.components]

        # Merge components
        base = built_components[0]
        for component in built_components[1:]:
            base = base.merge(component)
        object.__setattr__(self, 'productions', base.production_rules)

        # Set up PLY-compatible grammar attributes
        object.__setattr__(
            self,
            'tokens',
            tuple(token.name for token in base.tokens),
        )
        object.__setattr__(
            self,
            'states',
            base.states,
        )
        # Build precedence rules from token properties
        precedence_rules = []
        for token in base.tokens:
            if token.associativity != Associativity.NONE:
                precedence_rules.append((
                    (
                        'left'
                        if token.associativity == Associativity.LEFT
                        else 'right'
                    ),
                    token.name,
                ))
        # Sort by precedence level (earlier = tighter binding)
        precedence_rules.sort(key=lambda x: next(
            t.precedence for t in base.tokens if t.name == x[1]
        ))
        object.__setattr__(self, 'precedence', tuple(precedence_rules))

        # Register production rules
        for rule in base.production_rules:
            object.__setattr__(self, *rule.materialise(self))

        # Register token rules
        for token in base.tokens:
            object.__setattr__(self, *token.materialise(self))

        # Load or generate example values
        try:
            import json
            import os
            if os.path.exists(self._example_cache_file):
                with open(self._example_cache_file, 'r') as f:
                    cached_examples = json.load(f)
                # Update error handler with cached examples
                error_handler = self.error_handler.materialise_examples(
                    base.tokens,
                    cached_examples,
                )
            else:
                # Generate and cache examples
                error_handler = self.error_handler.materialise_examples(
                    base.tokens,
                )
                # Cache the examples
                with open(self._example_cache_file, 'w') as f:
                    json.dump(error_handler.example_values, f, indent=2)
        except (ImportError, IOError, json.JSONDecodeError):
            # Fallback to generating examples without caching
            error_handler = self.error_handler.materialise_examples(
                base.tokens,
            )
        object.__setattr__(
            self,
            'error_handler',
            error_handler,
        )

        # Register error handlers
        object.__setattr__(
            self,
            't_error',
            self.error_handler.create_token_error_function(),
        )
        object.__setattr__(
            self,
            'p_error',
            self.error_handler.create_parser_error_function(),
        )

        # Build namespaces from tokens
        namespaces = Namespaces()
        for token in base.tokens:
            if token.namespace is not None:
                namespaces = namespaces.add_token(
                    token.namespace, token.regex, token.name
                )
            elif token.state is not None:
                namespaces = namespaces.add_token(
                    token.state, token.regex, token.name
                )
            else:
                # Add to default namespace
                namespaces = namespaces.add_token(
                    namespaces.default_namespace, token.regex, token.name
                )
        object.__setattr__(self, '_namespaces', namespaces)

        lexer = lex.lex(module=self)
        parser = yacc.yacc(module=self)
        lexer.grammar = self
        parser.grammar = self
        self.error_handler._set_parser(parser)
        object.__setattr__(self, '_lexer', lexer)
        object.__setattr__(self, '_parser', parser)

        # Mark as initialized
        object.__setattr__(self, '_is_initialized', True)

    def __getattr__(self, name: str) -> Any:
        """Handle dynamic attribute access for PLY compatibility."""
        if name.startswith('t_'):
            # Handle token patterns using namespaces
            all_tokens = self._namespaces.get_all_tokens()
            if name in all_tokens:
                return lambda t: all_tokens[name]
        return super().__getattribute__(name)

    def __repr__(self):
        return wl.pformat(self)

    @property
    def namespaces(self) -> Namespaces:
        """Get the full namespaces collection."""
        return self._namespaces

    def input(self, data: str) -> Any:
        """Lex the input data."""
        return self._lexer.input(data)

    def parse(self, data: str) -> Any:
        """Parse the input data."""
        return self._parser.parse(data)

    def refresh(self) -> 'DynamicGrammar':
        """Refresh the lexer and parser."""
        lexer = lex.lex(module=self)
        parser = yacc.yacc(module=self)
        return dataclasses.replace(
            self,
            _lexer=lexer,
            _parser=parser,
        )

    def __lexer__(self):
        return self._lexer

    def __parser__(self):
        return self._parser


class Subcontext(BaseModel):
    """Base class for composable execution context features."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of this subcontext."""
        return self.model_dump()

    def update_state(self, state: Dict[str, Any]) -> 'Subcontext':
        """Update the state of this subcontext."""
        return self.model_copy(update=state)


class CacheSubcontext(Subcontext):
    """Cache functionality as a composable subcontext."""
    cache: Dict[str, Any] = Field(default_factory=dict)

    def get_cached(self, key: str) -> Optional[Any]:
        """Get cached value."""
        return self.cache.get(key)

    def set_cached(self, key: str, value: Any) -> 'CacheSubcontext':
        """Set cached value."""
        new_cache = dict(self.cache)
        new_cache[key] = value
        return self.model_copy(update={'cache': new_cache})


class CacheSubcontextMixin:
    """Mixin for cache subcontext."""
    def __add_subcontext__(self):
        """Add cache subcontext to parent."""
        subcontexts = self.subcontexts
        if '__cache' not in subcontexts:
            subcontexts['__cache'] = CacheSubcontext()
        object.__setattr__(self, 'subcontexts', subcontexts)

    def with_cache(self) -> 'ExecutionContext':
        """Add cache functionality to the context."""
        return self.with_subcontext('cache', CacheSubcontext())

    def get_cached(self, key: str) -> Optional[Any]:
        """Get cached value if cache subcontext exists."""
        cache_subcontext = self.get_subcontext('cache')
        if cache_subcontext:
            return cache_subcontext.get_cached(key)
        return None

    def set_cached(self, key: str, value: Any) -> 'ExecutionContext':
        """Set cached value if cache subcontext exists."""
        cache_subcontext = self.get_subcontext('cache')
        if cache_subcontext:
            new_cache = cache_subcontext.set_cached(key, value)
            return self.with_subcontext('cache', new_cache)
        return self


class TypedState(BaseModel):
    """Type-safe context state with Pydantic validation."""
    eval: Optional[Any] = None
    _default_field: str = 'eval'
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    def pop(self, *pparams) -> Tuple['TypedState', 'TypedState']:
        """Pop values from the state."""
        if not pparams:
            pparams = (self._default_field,)
        print(self.__class__)
        result = self.__class__(**{
            key: getattr(self, key)
            for key in pparams
        })
        state = self.__class__.model_validate(
            self.model_dump(exclude=pparams)
        )
        print(result, state)
        return result, state

    def update(self, *pparams, **update) -> 'TypedState':
        """Update the state."""
        if pparams:
            update = {
                **update,
                self._default_field: pparams[0],
            }
        return self.model_copy(update=update)


class UninitialisedState(TypedState):
    """Uninitialised state."""
    def pop(self, *pparams):
        raise ValueError("State is not initialised")

    def update(self, *pparams, **update):
        raise ValueError("State is not initialised")


class ExecutionContext(BaseModel):
    __state__: Type[TypedState] = TypedState
    interpreter: Dict[str, Callable] = Field(
        description="Mapping of operation names to callable implementations",
        default_factory=dict,
    )
    state: TypedState = Field(default_factory=UninitialisedState)
    subcontexts: Dict[str, Subcontext] = Field(default_factory=dict)
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    def model_post_init(self, __context__):
        """Post-initialization hook."""
        object.__setattr__(self, 'state', self.__state__())
        for parent in self.__class__.__mro__:
            if hasattr(parent, '__add_subcontext__'):
                parent.__add_subcontext__(self)

    def pop(self, *pparams):
        """Pop values from the context state."""
        result, state = self.state.pop(*pparams)
        return result, self.with_state(state)

    def update_state(self, **update) -> 'ExecutionContext':
        """Update the state of the context."""
        return self.model_copy(update={'state': self.state.update(**update)})

    def with_state(
        self,
        state: Any = None,
        **update,
    ) -> 'ExecutionContext':
        """Create context with updated result."""
        if isinstance(state, TypedState):
            return self.model_copy(update={'state': state})
        if (state is not None) and (not isinstance(state, TypedState)):
            update = {
                **update,
                self.state._default_field: state,
            }
        new_state = self.__state__(**update)
        return self.model_copy(update={'state': new_state})

    def get_state(self) -> Optional[Any]:
        """Get the current state."""
        return self.state

    def with_result(
        self,
        result: Any = None,
    ) -> 'ExecutionContext':
        """Create context with updated result."""
        return self.model_copy(update={'state': self.state.update(result)})

    def get_result(self) -> Optional[Any]:
        """Get the current result."""
        return getattr(self.get_state(), self.state._default_field)

    def with_subcontext(
        self,
        name: str,
        subcontext: Subcontext,
    ) -> 'ExecutionContext':
        """Add or update a subcontext with validation."""
        new_subcontexts = dict(self.subcontexts)
        new_subcontexts[f"__subcontext_{name}"] = subcontext
        return self.model_copy(update={'subcontexts': new_subcontexts})

    def get_subcontext(self, name: str) -> Optional[Subcontext]:
        """Get a subcontext by name."""
        return self.subcontexts.get(f"__subcontext_{name}", None)


@dataclasses.dataclass(frozen=True)
class TransformProcessor:
    """
    Enhanced processor with built-in initialization and finalization support.
    """
    grammar: DynamicGrammar
    preprocessors: Tuple[Mapping[str, str] | callable, ...]
    postprocessors: Tuple[callable, ...]
    interpreters: InterpretersDispatch
    context_class: Type[ExecutionContext]
    initialisation_hooks: Dict[str, callable] = dataclasses.field(
        default_factory=dict
    )
    finalisation_hooks: Dict[str, callable] = dataclasses.field(
        default_factory=dict
    )
    default_interpreter: str | None = None

    def __post_init__(self):
        if (
            hasattr(self.grammar, '__call__') and
            not hasattr(self.grammar, 'components')
        ):
            object.__setattr__(self, 'grammar', self.grammar())
        object.__setattr__(self, 'postprocessors', tuple(self.postprocessors))

    def __repr__(self):
        return wl.pformat(self)

    def _preprocess(self, expr: str) -> str:
        for preprocessor in self.preprocessors:
            if isinstance(preprocessor, Mapping):
                expr = re.sub(
                    rf'\b({"|".join(
                        re.escape(key) for key in preprocessor
                    )})',
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
        context: ExecutionContext,
    ) -> Tuple[Primitive, ExecutionContext]:
        for postprocessor in self.postprocessors:
            expr, context = postprocessor(expr, context)
        return expr, context

    def process(
        self,
        expression: str,
        interpreter: str | None = None,
    ) -> Tuple[Primitive, ExecutionContext]:
        expression = self._preprocess(expression)
        expression = self._parse(expression)
        context = self.context_class(
            interpreter=self.interpreters[
                interpreter or self.default_interpreter
            ],
        )
        expression, context = self._postprocess(expression, context)
        return expression, context

    def register_initialisation(self, interpreter_name: str, hook: callable):
        """Register initialisation hook for an interpreter."""
        object.__setattr__(self, 'initialisation_hooks', {
            **self.initialisation_hooks,
            interpreter_name: hook
        })

    def register_finalisation(self, interpreter_name: str, hook: callable):
        """Register finalisation hook for an interpreter."""
        object.__setattr__(self, 'finalisation_hooks', {
            **self.finalisation_hooks,
            interpreter_name: hook
        })

    def transform(
        self,
        expression: str,
        interpreter: str | None = None,
        **init_params,
    ) -> Any:
        """Transform expression with initialization and finalization."""
        # Parse AST
        interpreter = interpreter or self.default_interpreter
        ast, context = self.process(
            expression=expression,
            interpreter=interpreter,
        )
        if 'context' in init_params:
            context = init_params.pop('context')

        # Run initialization hook if present
        if interpreter in self.initialisation_hooks:
            ast, context = self.initialisation_hooks[interpreter](
                ast,
                context,
                **init_params,
            )

        # Execute AST
        result_context = ast(context)

        # Run finalization hook if present
        if interpreter in self.finalisation_hooks:
            result_context = self.finalisation_hooks[interpreter](
                result_context
            )

        return result_context.get_result()

    def __call__(self, expr: str, **params) -> Primitive:
        return self.transform(expr, **params)
