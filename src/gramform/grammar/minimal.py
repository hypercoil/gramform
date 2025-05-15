# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
DataFrames
~~~~~~~~~~
Grammar for DataFrame operations.
"""
import dataclasses
import re
from typing import Any, Iterable, Mapping, Tuple, Type

import ply.lex as lex
import ply.yacc as yacc
import wadler_lindig as wl


PRIMITIVES = {}
_RESERVED = {
    'I_': 'INDICATOR',
    'd_': 'BACKDIFF',
    'dd_': 'BACKDIFF_INCLUSIVE',
    'AND_': 'INTERSECTION_REDUCE',
    'OR_': 'UNION_REDUCE',
    'n_': 'FIRST_N',
    'v_': 'CUMUL_VAR',
}


@dataclasses.dataclass(frozen=True)
class Grammar:
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

    def __post_init__(self):
        # Wasteful to do this every time we bind.
        PRIMITIVES[self.name] = type(self)

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


CONCATENATE = Primitive("CONCATENATE", is_associative=True)
POWER = Primitive("POWER")
BACKDIFF = Primitive("BACKDIFF")
RANGE = Primitive("RANGE")
ENUM = Primitive("ENUM", is_associative=True)
INDICATOR = Primitive("INDICATOR")
UNION = Primitive("UNION", is_associative=True)
UNION_REDUCE = Primitive("UNION_REDUCE")
INTERSECTION = Primitive("INTERSECTION", is_associative=True)
INTERSECTION_REDUCE = Primitive("INTERSECTION_REDUCE")
NEGATION = Primitive("NEGATION")
SCATTER = Primitive("SCATTER")
FIRST_N = Primitive("FIRST_N")
CUMUL_VAR = Primitive("CUMUL_VAR")
ASSIGNMENT = Primitive("ASSIGNMENT")
COLLECT_PARAMETERS = Primitive("COLLECT_PARAMETERS")
CONDITION_EQUAL = Primitive("CONDITION_EQUAL", is_associative=True)
CONDITION_NOT_EQUAL = Primitive("CONDITION_NOT_EQUAL", is_associative=True)
CONDITION_LESS = Primitive("CONDITION_LESS")
CONDITION_LESS_EQUAL = Primitive("CONDITION_LESS_EQUAL")
CONDITION_GREATER = Primitive("CONDITION_GREATER")
CONDITION_GREATER_EQUAL = Primitive("CONDITION_GREATER_EQUAL")
VARIABLE = Primitive("VARIABLE", is_terminal=True)
EXECUTION_HEAD = Primitive("EXECUTION_HEAD")


def confound_formula_preprocessor():
    return {
        'wm': 'white_matter',
        'gsr': 'global_signal',
        'gs': 'global_signal',
        'rps': 'trans_x + trans_y + trans_z + rot_x + rot_y + rot_z',
        'fd': 'framewise_displacement',
        'dv': 'std_dvars',
        'acc': 'a_comp_cor',
        'wcc': 'w_comp_cor',
        'ccc': 'c_comp_cor',
    }


class MinimalGrammar:
    tokens = (
        'CONCATENATE',
        'POWER',
        'POWER_INCLUSIVE',
        'RANGE',
        'BACKDIFF',
        'BACKDIFF_INCLUSIVE',
        'ENUM_SEP',
        'ARG_SEP',
        'KV_SEP',
        'INDICATOR',
        'CONDITION_EQUAL',
        'CONDITION_NOT_EQUAL',
        'CONDITION_LESS',
        'CONDITION_LESS_EQUAL',
        'CONDITION_GREATER',
        'CONDITION_GREATER_EQUAL',
        'UNION',
        'UNION_REDUCE',
        'INTERSECTION',
        'INTERSECTION_REDUCE',
        'NEGATION',
        'FIRST_N',
        'CUMUL_VAR',
        'SCATTER',
        'LPAREN',
        'RPAREN',
        'LBRACKET',
        'RBRACKET',
        'LBRACE',
        'RBRACE',
        'begin_param',
        'end_param',
        'VARIABLE',
        'FLOAT',
        'INTEGER',
    )

    states = (
        ('param', 'exclusive'),
    )

    t_CONCATENATE = r'\+'
    t_POWER = r'\^'
    t_POWER_INCLUSIVE = r'\^\^'
    t_RANGE = r'\-'
    t_ANY_ENUM_SEP = r','
    t_param_ARG_SEP = r';'
    t_param_KV_SEP = r'='
    t_CONDITION_EQUAL = r'='
    t_CONDITION_NOT_EQUAL = r'(<>|!=|~=)'
    t_CONDITION_LESS = r'<'
    t_CONDITION_LESS_EQUAL = r'<='
    t_CONDITION_GREATER = r'>'
    t_CONDITION_GREATER_EQUAL = r'>='
    t_UNION = r'\|\|'
    t_INTERSECTION = r'\&\&'
    t_NEGATION = r'!'
    t_SCATTER = r'\:\:\:'
    t_LPAREN = r'\('
    t_RPAREN = r'\)'
    t_LBRACKET = r'\['
    t_RBRACKET = r'\]'
    t_LBRACE = r'\{'
    t_RBRACE = r'\}'
    t_ANY_FLOAT = r'\d+\.\d*'
    t_ANY_INTEGER = r'\d+'
    t_ANY_ignore = ' \t'

    def t_ANY_VARIABLE(t):
        r'[a-zA-Z_][a-zA-Z0-9_]*'
        if t.value in _RESERVED:
            t.type = _RESERVED[t.value]
        return t

    def t_ANY_error(t):
        raise ValueError(f"Illegal character '{t.value}'")
    
    def t_begin_param(t):
        r'\{\{'
        t.lexer.push_state('param')
        return t

    def t_param_end_param(t):
        r'\}\}'
        t.lexer.pop_state()
        return t

    precedence = (
        (
            'left',
            'CONDITION_EQUAL',
            'CONDITION_NOT_EQUAL',
            'CONDITION_LESS',
            'CONDITION_LESS_EQUAL',
            'CONDITION_GREATER',
            'CONDITION_GREATER_EQUAL',
        ),
        ('left', 'CONCATENATE'),
        ('right', 'SCATTER', 'CUMUL_VAR', 'FIRST_N'),
        ('right', 'NEGATION'),
        ('left', 'UNION'),
        ('left', 'INTERSECTION'),
        ('right', 'UNION_REDUCE'),
        ('right', 'INTERSECTION_REDUCE'),
        (
            'left',
            'POWER',
            'POWER_INCLUSIVE',
        ),
        (
            'right',
            'BACKDIFF',
            'BACKDIFF_INCLUSIVE',
        ),
        ('left', 'ENUM_SEP'),
        ('left', 'RANGE'),
        ('left', 'begin_param', 'end_param'),
        ('left', 'ARG_SEP'),
        ('left', 'KV_SEP'),
    )

    def p_expression_concatenate(p):
        'expression : expression CONCATENATE expression'
        p[0] = CONCATENATE.bind(p[1], p[3])

    def p_expression_power(p):
        'expression : expression POWER expression'
        p[0] = POWER.bind(p[1], p[3])

    def p_expression_power_inclusive(p):
        'expression : expression POWER_INCLUSIVE expression'
        p[0] = POWER.bind(p[1], RANGE.bind(
            Literal.create(1, int),
            p[3],
        ))

    def p_expression_backdiff(p):
        'expression : BACKDIFF parameter LPAREN expression RPAREN'
        p[0] = BACKDIFF.bind(p[4], p[2])

    def p_expression_backdiff_inclusive(p):
        'expression : BACKDIFF_INCLUSIVE parameter LPAREN expression RPAREN'
        p[0] = BACKDIFF.bind(p[4], RANGE.bind(
            Literal.create(0, int),
            p[2],
        ))

    def p_expression_range(p):
        'expression : expression RANGE expression'
        p[0] = RANGE.bind(p[1], p[3])

    def p_expression_enum(p):
        'expression : expression ENUM_SEP expression'
        p[0] = ENUM.bind(p[1], p[3])

    def p_expression_indicator(p):
        'expression : INDICATOR parameter'
        p[0] = INDICATOR.bind(p[2])

    def p_expression_condition_equal(p):
        'expression : expression CONDITION_EQUAL expression'
        p[0] = CONDITION_EQUAL.bind(p[1], p[3])

    def p_expression_condition_not_equal(p):
        'expression : expression CONDITION_NOT_EQUAL expression'
        p[0] = CONDITION_NOT_EQUAL.bind(p[1], p[3])

    def p_expression_condition_less(p):
        'expression : expression CONDITION_LESS expression'
        p[0] = CONDITION_LESS.bind(p[1], p[3])

    def p_expression_condition_less_equal(p):
        'expression : expression CONDITION_LESS_EQUAL expression'
        p[0] = CONDITION_LESS_EQUAL.bind(p[1], p[3])

    def p_expression_condition_greater(p):
        'expression : expression CONDITION_GREATER expression'
        p[0] = CONDITION_GREATER.bind(p[1], p[3])

    def p_expression_condition_greater_equal(p):
        'expression : expression CONDITION_GREATER_EQUAL expression'
        p[0] = CONDITION_GREATER_EQUAL.bind(p[1], p[3])

    def p_expression_union(p):
        'expression : expression UNION expression'
        p[0] = UNION.bind(p[1], p[3])

    def p_expression_union_reduce(p):
        'expression : UNION_REDUCE expression'
        p[0] = UNION_REDUCE.bind(p[2])

    def p_expression_intersection(p):
        'expression : expression INTERSECTION expression'
        p[0] = INTERSECTION.bind(p[1], p[3])

    def p_expression_intersection_reduce(p):
        'expression : INTERSECTION_REDUCE expression'
        p[0] = INTERSECTION_REDUCE.bind(p[2])

    def p_expression_negation(p):
        'expression : NEGATION expression'
        p[0] = NEGATION.bind(p[2])

    def p_expression_first_n(p):
        'expression : FIRST_N parameter'
        p[0] = FIRST_N.bind(p[2])

    def p_expression_cumul_var(p):
        'expression : CUMUL_VAR parameter'
        p[0] = CUMUL_VAR.bind(p[2])

    def p_expression_scatter(p):
        'expression : SCATTER expression'
        p[0] = SCATTER.bind(p[2])

    def p_expression_paren_term(p):
        'expression : LPAREN expression RPAREN'
        p[0] = p[2]

    def p_expression_parameter(p):
        'parameter : LBRACKET expression RBRACKET'
        p[0] = p[2]

    def p_expression_parameterisation(p):
        'parameter : begin_param expression end_param'
        parameters = p[2]
        if isinstance(parameters, tuple):
            p[0] = COLLECT_PARAMETERS.bind(*parameters)
        else:
            p[0] = COLLECT_PARAMETERS.bind(parameters)

    def p_param_expr(p):
        'expression : expression ARG_SEP expression'
        left, right = p[1], p[3]
        if not isinstance(left, Iterable):
            left = (left,)
        if not isinstance(right, Iterable):
            right = (right,)
        p[0] = tuple(left) + tuple(right)

    def p_param_expr_key_val(p):
        'expression : expression KV_SEP expression'
        p[0] = ASSIGNMENT.bind(p[1], p[3])

    def p_expression_term_variable(p):
        'expression : VARIABLE'
        p[0] = VARIABLE.bind(p[1])

    def p_expression_term_integer(p):
        'expression : INTEGER'
        p[0] = Literal.create(int(p[1]), int)

    def p_expression_term_float(p):
        'expression : FLOAT'
        p[0] = Literal.create(float(p[1]), float)

    def p_error(p):
        raise ValueError(f"Syntax error: {p}")

    def __lexer__(self, **params):
        return lex.lex(module=self, **params)

    def __parser__(self, **params):
        return yacc.yacc(module=self, **params)


def MinimalGrammarLexer(**params):
    lexer = lex.lex(module=MinimalGrammar, **params)
    return lexer


def MinimalGrammarParser(**params):
    parser = yacc.yacc(module=MinimalGrammar, **params)
    return parser


def ppr_associative_flatten(tree):
    def _flatten(children, to_flatten):
        for child, flatten in zip(children, to_flatten):
            if flatten:
                yield from child.parameters
            else:
                yield child

    if tree.is_terminal:
        return tree
    children = [
        ppr_associative_flatten(child)
        for child in tree.parameters
    ]
    to_flatten = [
        getattr(child, 'name', None) == tree.name
        and tree.is_associative
        for child in children
    ]
    children = tuple(_flatten(children, to_flatten))
    return tree.bind(*children)


def ppr_common_subexpression(tree):
    subexpressions = {}
    # TODO: This is a cache, but it's not used.
    # We don't use this cache, but it's here in case it simplifies a
    # future implementation. If not, we should remove it.
    cache = set()

    def _collect(tree, subexpressions):
        children = []
        for child in tree.parameters:
            if child in subexpressions and not child.is_terminal:
                children.append(subexpressions[child])
                cache.add(child)
            else:
                if not child.is_terminal:
                    child, subexpressions = _collect(child, subexpressions)
                    subexpressions[child] = child
                children.append(child)

        return tree.bind(*children), subexpressions

    tree, _ = _collect(tree, subexpressions)
    return tree


def ppr_execution_head(tree):
    if tree.name != 'EXECUTION_HEAD':
        tree = EXECUTION_HEAD.bind(tree)
    return tree


def main():
    # expr = '(x+y+z)^^2+(x+y+z)+((x+y+z)^2+(x+y+z))^3.13-5'
    # expr = '(x+y+z)^^2-3 + I_[x=y] + d_[1,4-5](x)'
    # expr = ':::!((I_[x=y] && I_[x=z]) || I_[x>=w]) + AND_(I_[x=y] + I_[x=z] + OR_(I_[x=w] + I_[x=v])) + v_{{test; x=1; y=2; z=3}}'
    # expr = '(x+y+z)^^2 + (x+y+z)^^2 + (x+y+z)^^2'
    expr = 'd_[1]((x+y)^^2 + (x+y)^^2) + d_[1]((x+y)^^2 + (x+y)^^2)'
    lexer = MinimalGrammarLexer()
    parser = MinimalGrammarParser()
    lexer.input(expr)
    for tok in lexer:
        print(tok)
    result = parser.parse(expr)
    print(result)
    result = ppr_associative_flatten(result)
    result = ppr_common_subexpression(result)
    result = ppr_execution_head(result)
    print(result)


if __name__ == "__main__":
    main()
