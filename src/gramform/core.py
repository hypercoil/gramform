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

    def from_sequence(token: str) -> int:
        return precedence.get(token, default)

    return from_sequence


def push_state_and_return(state: str):
    def _inner(t):
        t.lexer.push_state(state)
        return t
    return _inner


def pop_state_and_return(t):
    t.lexer.pop_state()
    return t


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
        # print(f"Primitive {self.name} bound with {pparams}")
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


def literal(dtype: Type):
    def _inner(value):
        return Literal.create(dtype(value), dtype)
    return _inner


def unop_prefix(prim: Primitive):
    def _inner(_, right):
        return prim.bind(right)
    return _inner


def unop_postfix(prim: Primitive):
    def _inner(left, _):
        return prim.bind(left)
    return _inner


def binop_infix(prim: Primitive):
    def _inner(left, _, right):
        return prim.bind(left, right)
    return _inner


def binop_prefix(prim: Primitive):
    def _inner(_, left, right):
        return prim.bind(left, right)
    return _inner


def binop_postfix(prim: Primitive):
    def _inner(left, right, _):
        return prim.bind(left, right)
    return _inner


def enter_group():
    def _inner(_, inner, __):
        return inner
    return _inner


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
        if not self.name.startswith('p_'):
            raise ValueError("Production function names must start with 'p_'")
        if ':' not in self.rule:
            raise ValueError(
                "Production rule must contain ':' to separate LHS and RHS"
            )

    def __repr__(self):
        return wl.pformat(self)

    def materialise(self) -> callable:
        """Create a PLY-compatible production function."""
        def production_func(p):
            p[0] = self.implementation(*p[1:])
        production_func.__name__ = self.name
        production_func.__doc__ = self.rule
        return production_func


class Associativity(Enum):
    """Associativity of operators."""
    LEFT = auto()
    RIGHT = auto()
    NONE = auto()  # For non-operator tokens


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
        The name of the token (e.g., 'CONCATENATE').
    regex: str
        The regex pattern for the token (e.g., r'\+').
    function: Optional[callable]
        An optional function to handle the token.
    state: Optional[str]
        An optional lexer state for the token.
    precedence: int
        The precedence level of the token. A higher number means tighter
        binding.
    associativity: Associativity
        The associativity of the token.
    is_reserved: bool
        Whether this is a reserved word.
    category: Optional[str]
        An optional category for error context matching.
    """
    name: str
    regex: str
    function: Optional[callable] = None
    state: Optional[str] = None
    precedence: int = 0
    associativity: Associativity = Associativity.NONE
    is_reserved: bool = False
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

    def generate_example(self) -> str:
        """
        Generate an example value for this token based on its regex pattern.

        This is a crude AI-generated placeholder. We should use a more
        principled approach to generating examples through deterministic
        traversal of the regex AST.
        """
        return generate_valid_completion(self.regex)

    def materialise(self) -> Tuple[str, Callable | str]:
        """Create a PLY-compatible token function or regex."""
        if self.state:
            state_name = f'{self.state}_'
        else:
            state_name = 'ANY_'
        name = f't_{state_name}{self.name}'

        if not self.function:
            return name, self.regex

        func = self.function
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

        # Check for token conflicts
        self_tokens = {token.name for token in self.tokens}
        other_tokens = {token.name for token in other.tokens}
        conflicts = self_tokens & other_tokens
        if conflicts:
            raise ValueError(f"Conflicting tokens: {conflicts}")

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
class DynamicGrammar(Grammar):
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
            setattr(self, rule.name, rule.materialise())

        # Register token rules
        for token in base.tokens:
            setattr(self, *token.materialise())

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
        setattr(
            self,
            't_error',
            self.error_handler.create_token_error_function(),
        )
        setattr(
            self,
            'p_error',
            self.error_handler.create_parser_error_function(),
        )
        # Build reserved words mapping
        reserved = {
            token.name: token.name
            for token in base.tokens
            if token.is_reserved
        }
        object.__setattr__(self, '_reserved', reserved)

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
            # Handle token patterns
            if name in self._reserved:
                return lambda t: self._reserved[name]
        return super().__getattribute__(name)

    def __repr__(self):
        return wl.pformat(self)

    @property
    def reserved(self) -> Dict[str, str]:
        return self._reserved

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
