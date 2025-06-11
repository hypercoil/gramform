# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
DataFrames
~~~~~~~~~~
Grammar for DataFrame operations.
"""
from dataclasses import dataclass, field
from typing import Tuple

from gramform.core import (
    Associativity,
    DynamicGrammar,
    GrammarComponent,
    GrammarErrorHandler,
    Literal,
    pop_state_and_return,
    precedence_from_sequence,
    Primitive,
    ProductionRule,
    push_state_and_return,
    Token,
)


# Base primitives
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


_RESERVED = {
    'I_': 'INDICATOR',
    'd_': 'BACKDIFF',
    'dd_': 'BACKDIFF_INCLUSIVE',
    'AND_': 'INTERSECTION_REDUCE',
    'OR_': 'UNION_REDUCE',
    'NOT_': 'NEGATION_SURFACE',
    'n_': 'FIRST_N',
    'v_': 'CUMUL_VAR',
}


TOKEN_PRECEDENCE = (
    "CONCATENATE",
    ("SCATTER", "CUMUL_VAR", "FIRST_N"),
    "NEGATION",
    "UNION",
    "INTERSECTION",
    "UNION_REDUCE",
    "INTERSECTION_REDUCE",
    "NEGATION_SURFACE",
    (
        "CONDITION_EQUAL",
        "CONDITION_NOT_EQUAL",
        "CONDITION_LESS",
        "CONDITION_LESS_EQUAL",
        "CONDITION_GREATER",
        "CONDITION_GREATER_EQUAL",
    ),
    (
        "POWER",
        "POWER_INCLUSIVE",
    ),
    (
        "BACKDIFF",
        "BACKDIFF_INCLUSIVE",
    ),
    "ENUM_SEP",
    "RANGE",
    ("begin_param", "end_param"),
    "ARG_SEP",
    "KV_SEP",
)
from_sequence = precedence_from_sequence(TOKEN_PRECEDENCE)


def variable(t):
    """Handle variable tokens and reserved words."""
    if t.value in _RESERVED:
        t.type = _RESERVED[t.value]
    return t


@dataclass(frozen=True)
class BasicOperatorsComponent(GrammarComponent):
    """Component for basic operators."""
    tokens: Tuple[Token, ...] = (
        # Basic operators
        Token(
            'CONCATENATE',
            r'\+',
            precedence=from_sequence,
            associativity=Associativity.LEFT,
        ),
        Token(
            'RANGE',
            r'\-',
            precedence=from_sequence,
            associativity=Associativity.LEFT,
        ),
        Token(
            'ENUM_SEP',
            r',',
            precedence=from_sequence,
            associativity=Associativity.LEFT,
        ),

        # Parentheses and brackets
        Token('LPAREN', r'\('),
        Token('RPAREN', r'\)'),
        Token('LBRACKET', r'\['),
        Token('RBRACKET', r'\]'),
        Token('LBRACE', r'\{'),
        Token('RBRACE', r'\}'),

        # Numbers
        Token('FLOAT', r'\d+\.\d*'),
        Token('INTEGER', r'\d+'),

        # Whitespace
        Token('ignore', ' \t'),
    )

    states: Tuple[Tuple[str, str], ...] = ()

    production_rules: Tuple[ProductionRule, ...] = (
        # Basic arithmetic
        ProductionRule(
            'p_expression_concatenate',
            'expression : expression CONCATENATE expression',
            lambda left, _, right: CONCATENATE.bind(left, right)
        ),
        ProductionRule(
            'p_expression_range',
            'expression : expression RANGE expression',
            lambda left, _, right: RANGE.bind(left, right)
        ),
        ProductionRule(
            'p_expression_enum',
            'expression : expression ENUM_SEP expression',
            lambda left, _, right: ENUM.bind(left, right)
        ),

        # Parentheses
        ProductionRule(
            'p_expression_paren_term',
            'expression : LPAREN expression RPAREN',
            lambda _, inner, __: inner
        ),

        # Numbers
        ProductionRule(
            'p_expression_term_integer',
            'expression : INTEGER',
            lambda terminal: Literal.create(int(terminal), int)
        ),
        ProductionRule(
            'p_expression_term_float',
            'expression : FLOAT',
            lambda terminal: Literal.create(float(terminal), float)
        ),
    )


@dataclass(frozen=True)
class ConditionComponent(GrammarComponent):
    """Component for condition handling."""
    tokens: Tuple[Token, ...] = (
        Token(
            'CONDITION_EQUAL',
            r'=',
            precedence=from_sequence,
            associativity=Associativity.LEFT,
        ),
        Token(
            'CONDITION_NOT_EQUAL',
            r'(<>|!=|~=)',
            precedence=from_sequence,
            associativity=Associativity.LEFT,
        ),
        Token(
            'CONDITION_LESS',
            r'<',
            precedence=from_sequence,
            associativity=Associativity.LEFT,
        ),
        Token(
            'CONDITION_LESS_EQUAL',
            r'<=',
            precedence=from_sequence,
            associativity=Associativity.LEFT,
        ),
        Token(
            'CONDITION_GREATER',
            r'>',
            precedence=from_sequence,
            associativity=Associativity.LEFT,
        ),
        Token(
            'CONDITION_GREATER_EQUAL',
            r'>=',
            precedence=from_sequence,
            associativity=Associativity.LEFT,
        ),
    )

    production_rules: Tuple[ProductionRule, ...] = (
        ProductionRule(
            'p_expression_condition_equal',
            'expression : expression CONDITION_EQUAL expression',
            lambda left, _, right: CONDITION_EQUAL.bind(left, right)
        ),
        ProductionRule(
            'p_expression_condition_not_equal',
            'expression : expression CONDITION_NOT_EQUAL expression',
            lambda left, _, right: CONDITION_NOT_EQUAL.bind(left, right)
        ),
        ProductionRule(
            'p_expression_condition_less',
            'expression : expression CONDITION_LESS expression',
            lambda left, _, right: CONDITION_LESS.bind(left, right)
        ),
        ProductionRule(
            'p_expression_condition_less_equal',
            'expression : expression CONDITION_LESS_EQUAL expression',
            lambda left, _, right: CONDITION_LESS_EQUAL.bind(left, right)
        ),
        ProductionRule(
            'p_expression_condition_greater',
            'expression : expression CONDITION_GREATER expression',
            lambda left, _, right: CONDITION_GREATER.bind(left, right)
        ),
        ProductionRule(
            'p_expression_condition_greater_equal',
            'expression : expression CONDITION_GREATER_EQUAL expression',
            lambda left, _, right: CONDITION_GREATER_EQUAL.bind(left, right)
        ),
    )


@dataclass(frozen=True)
class BooleanLogicComponent(GrammarComponent):
    """Component for boolean logic."""
    tokens: Tuple[Token, ...] = (
        Token(
            'UNION',
            r'\|\|',
            precedence=from_sequence,
            associativity=Associativity.LEFT,
        ),
        Token(
            'INTERSECTION',
            r'\&\&',
            precedence=from_sequence,
            associativity=Associativity.LEFT,
        ),
        Token(
            'NEGATION',
            r'!',
            precedence=from_sequence,
            associativity=Associativity.RIGHT,
        ),
        Token(
            'INDICATOR',
            r'I_',
            is_reserved=True,
            precedence=from_sequence,
        ),
        Token(
            'INTERSECTION_REDUCE',
            r'AND_',
            is_reserved=True,
            precedence=from_sequence,
        ),
        Token(
            'UNION_REDUCE',
            r'OR_',
            is_reserved=True,
            precedence=from_sequence,
        ),
        Token(
            'NEGATION_SURFACE',
            r'NOT_',
            is_reserved=True,
            precedence=from_sequence,
        ),
        Token(
            'SCATTER',
            r'\:\:\:',
            precedence=1,
            associativity=Associativity.RIGHT,
        ),
    )

    production_rules: Tuple[ProductionRule, ...] = (
        ProductionRule(
            'p_expression_union',
            'expression : expression UNION expression',
            lambda left, _, right: UNION.bind(left, right)
        ),
        ProductionRule(
            'p_expression_intersection',
            'expression : expression INTERSECTION expression',
            lambda left, _, right: INTERSECTION.bind(left, right)
        ),
        ProductionRule(
            'p_expression_negation',
            'expression : NEGATION expression',
            lambda _, right: NEGATION.bind(right)
        ),
        ProductionRule(
            'p_expression_indicator',
            'expression : INDICATOR parameter',
            lambda _, parameter: INDICATOR.bind(parameter)
        ),
        ProductionRule(
            'p_expression_intersection_reduce',
            'expression : INTERSECTION_REDUCE LPAREN expression RPAREN',
            lambda _, __, right, ___: INTERSECTION_REDUCE.bind(right)
        ),
        ProductionRule(
            'p_expression_union_reduce',
            'expression : UNION_REDUCE LPAREN expression RPAREN',
            lambda _, __, right, ___: UNION_REDUCE.bind(right)
        ),
        ProductionRule(
            'p_expression_negation_surface',
            'expression : NEGATION_SURFACE LPAREN expression RPAREN',
            lambda _, __, right, ___: INDICATOR.bind(NEGATION.bind(right))
        ),
        ProductionRule(
            'p_expression_scatter',
            'expression : SCATTER expression',
            lambda _, right: SCATTER.bind(right)
        ),
    )


@dataclass(frozen=True)
class ParameterComponent(GrammarComponent):
    """Component for parameter handling."""

    tokens: Tuple[Token, ...] = (
        # Parameter state tokens
        Token(
            'begin_param',
            r'\{\{',
            function=push_state_and_return('param'),
            precedence=from_sequence,
        ),
        Token(
            'end_param',
            r'\}\}',
            function=pop_state_and_return,
            precedence=from_sequence,
        ),
        Token(
            'ARG_SEP',
            r';',
            state='param',
            precedence=from_sequence,
            associativity=Associativity.LEFT,
        ),
        Token(
            'KV_SEP',
            r'=',
            state='param',
            precedence=from_sequence,
            associativity=Associativity.LEFT,
        ),
    )

    states: Tuple[Tuple[str, str], ...] = (
        ('param', 'inclusive'),
    )

    production_rules: Tuple[ProductionRule, ...] = (
        ProductionRule(
            'p_expression_parameter',
            'parameter : LBRACKET expression RBRACKET',
            lambda _, inner, __: inner
        ),
        ProductionRule(
            'p_expression_parameterisation',
            'parameter : begin_param expression end_param',
            lambda _, inner, __: (
                COLLECT_PARAMETERS.bind(*inner)
                if isinstance(inner, tuple)
                else COLLECT_PARAMETERS.bind(inner)
            )
        ),
        ProductionRule(
            'p_param_expr',
            'expression : expression ARG_SEP expression',
            lambda left, _, right: COLLECT_PARAMETERS.bind(*(
                tuple(left if isinstance(left, tuple) else (left,)) +
                tuple(right if isinstance(right, tuple) else (right,))
            ))
        ),
        ProductionRule(
            'p_param_expr_key_val',
            'expression : expression KV_SEP expression',
            lambda left, _, right: ASSIGNMENT.bind(left, right)
        ),
    )


@dataclass(frozen=True)
class VariableComponent(GrammarComponent):
    """Component for variable handling."""

    tokens: Tuple[Token, ...] = (
        Token(
            'VARIABLE',
            r'[a-zA-Z_][a-zA-Z0-9_]*',
            function=variable,
            precedence=from_sequence,
        ),
    )

    production_rules: Tuple[ProductionRule, ...] = (
        ProductionRule(
            'p_expression_term_variable',
            'expression : VARIABLE',
            lambda terminal: VARIABLE.bind(terminal)
        ),
    )


@dataclass(frozen=True)
class SpecialOperatorsComponent(GrammarComponent):
    """Component for special operators like indicators and backdiff."""

    tokens: Tuple[Token, ...] = (
        Token(
            'POWER',
            r'\^',
            precedence=from_sequence,
            associativity=Associativity.LEFT,
        ),
        Token(
            'POWER_INCLUSIVE',
            r'\^\^',
            precedence=from_sequence,
            associativity=Associativity.LEFT,
        ),
        Token(
            'BACKDIFF',
            r'd_',
            is_reserved=True,
            precedence=from_sequence,
        ),
        Token(
            'BACKDIFF_INCLUSIVE',
            r'dd_',
            is_reserved=True,
            precedence=from_sequence,
        ),
        Token(
            'FIRST_N',
            r'n_',
            is_reserved=True,
            precedence=from_sequence,
        ),
        Token(
            'CUMUL_VAR',
            r'v_',
            is_reserved=True,
            precedence=from_sequence,
        ),
    )

    production_rules: Tuple[ProductionRule, ...] = (
        ProductionRule(
            'p_expression_power',
            'expression : expression POWER expression',
            lambda left, _, right: POWER.bind(left, right)
        ),
        ProductionRule(
            'p_expression_power_inclusive',
            'expression : expression POWER_INCLUSIVE expression',
            lambda left, _, right: POWER.bind(left, RANGE.bind(
                Literal.create(1, int),
                right,
            ))
        ),
        ProductionRule(
            'p_expression_backdiff',
            'expression : BACKDIFF parameter LPAREN expression RPAREN',
            lambda _, parameter, __, inner, ___: BACKDIFF.bind(inner, parameter)
        ),
        ProductionRule(
            'p_expression_backdiff_inclusive',
            'expression : BACKDIFF_INCLUSIVE parameter LPAREN expression RPAREN',
            lambda _, parameter, __, inner, ___: BACKDIFF.bind(inner, RANGE.bind(
                Literal.create(0, int),
                parameter,
            ))
        ),
        ProductionRule(
            'p_expression_first_n',
            'expression : FIRST_N parameter',
            lambda _, parameter: FIRST_N.bind(parameter)
        ),
        ProductionRule(
            'p_expression_cumul_var',
            'expression : CUMUL_VAR parameter',
            lambda _, parameter: CUMUL_VAR.bind(parameter)
        ),
    )


class MinimalGrammar(DynamicGrammar):
    """Grammar for DataFrame operations using composable components."""

    def __init__(self):
        super().__init__(
            components=(
                BasicOperatorsComponent(),
                ConditionComponent(),
                BooleanLogicComponent(),
                ParameterComponent(),
                VariableComponent(),
                SpecialOperatorsComponent(),
            ),
            error_handler=GrammarErrorHandler(
                error_contexts={
                    'PARAM': "Invalid parameter syntax",
                    'VARIABLE': "Invalid variable name",
                    'OPERATOR': "Invalid operator usage",
                    'CONDITION': "Invalid condition expression",
                    'PARENTHESIS': "Mismatched parentheses",
                    'BRACKET': "Mismatched brackets",
                    'BRACE': "Mismatched braces",
                    'NUMBER': "Invalid number format",
                    'RESERVED': "Invalid use of reserved word",
                }
            )
        )
