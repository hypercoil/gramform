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
    binop_infix,
    enter_group,
    DynamicGrammar,
    GrammarComponent,
    GrammarErrorHandler,
    Literal,
    literal,
    pop_state_and_return,
    precedence_from_sequence,
    Primitive,
    ProductionRule,
    push_state_and_return,
    Token,
    unop_prefix,
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
            category='OPERATOR',
        ),
        Token(
            'RANGE',
            r'\-',
            precedence=from_sequence,
            associativity=Associativity.LEFT,
            category='OPERATOR',
        ),
        Token(
            'ENUM_SEP',
            r',',
            precedence=from_sequence,
            associativity=Associativity.LEFT,
            category='DELIMITER',
        ),

        # Parentheses and brackets
        Token('LPAREN', r'\(', category='PARENTHESIS'),
        Token('RPAREN', r'\)', category='PARENTHESIS'),
        Token('LBRACKET', r'\[', category='BRACKET'),
        Token('RBRACKET', r'\]', category='BRACKET'),
        Token('LBRACE', r'\{', category='BRACE'),
        Token('RBRACE', r'\}', category='BRACE'),

        # Numbers
        Token('FLOAT', r'\d+\.\d*', category='LITERAL'),
        Token('INTEGER', r'\d+', category='LITERAL'),

        # Whitespace
        Token('ignore', ' \t'),
    )

    states: Tuple[Tuple[str, str], ...] = ()

    production_rules: Tuple[ProductionRule, ...] = (
        # Basic arithmetic
        ProductionRule(
            'p_expression_concatenate',
            'expression : expression CONCATENATE expression',
            binop_infix(CONCATENATE),
        ),
        ProductionRule(
            'p_expression_range',
            'expression : expression RANGE expression',
            binop_infix(RANGE),
        ),
        ProductionRule(
            'p_expression_enum',
            'expression : expression ENUM_SEP expression',
            binop_infix(ENUM),
        ),

        # Parentheses
        ProductionRule(
            'p_expression_paren_term',
            'expression : LPAREN expression RPAREN',
            enter_group(),
        ),

        # Numbers
        ProductionRule(
            'p_expression_term_integer',
            'expression : INTEGER',
            literal(int),
        ),
        ProductionRule(
            'p_expression_term_float',
            'expression : FLOAT',
            literal(float),
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
            category='CONDITION',
        ),
        Token(
            'CONDITION_NOT_EQUAL',
            r'(<>|!=|~=)',
            precedence=from_sequence,
            associativity=Associativity.LEFT,
            category='CONDITION',
        ),
        Token(
            'CONDITION_LESS',
            r'<',
            precedence=from_sequence,
            associativity=Associativity.LEFT,
            category='CONDITION',
        ),
        Token(
            'CONDITION_LESS_EQUAL',
            r'<=',
            precedence=from_sequence,
            associativity=Associativity.LEFT,
            category='CONDITION',
        ),
        Token(
            'CONDITION_GREATER',
            r'>',
            precedence=from_sequence,
            associativity=Associativity.LEFT,
            category='CONDITION',
        ),
        Token(
            'CONDITION_GREATER_EQUAL',
            r'>=',
            precedence=from_sequence,
            associativity=Associativity.LEFT,
            category='CONDITION',
        ),
    )

    production_rules: Tuple[ProductionRule, ...] = (
        ProductionRule(
            'p_expression_condition_equal',
            'expression : expression CONDITION_EQUAL expression',
            binop_infix(CONDITION_EQUAL),
        ),
        ProductionRule(
            'p_expression_condition_not_equal',
            'expression : expression CONDITION_NOT_EQUAL expression',
            binop_infix(CONDITION_NOT_EQUAL),
        ),
        ProductionRule(
            'p_expression_condition_less',
            'expression : expression CONDITION_LESS expression',
            binop_infix(CONDITION_LESS),
        ),
        ProductionRule(
            'p_expression_condition_less_equal',
            'expression : expression CONDITION_LESS_EQUAL expression',
            binop_infix(CONDITION_LESS_EQUAL),
        ),
        ProductionRule(
            'p_expression_condition_greater',
            'expression : expression CONDITION_GREATER expression',
            binop_infix(CONDITION_GREATER),
        ),
        ProductionRule(
            'p_expression_condition_greater_equal',
            'expression : expression CONDITION_GREATER_EQUAL expression',
            binop_infix(CONDITION_GREATER_EQUAL),
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
            category='LOGICAL_OPERATOR',
        ),
        Token(
            'INTERSECTION',
            r'\&\&',
            precedence=from_sequence,
            associativity=Associativity.LEFT,
            category='LOGICAL_OPERATOR',
        ),
        Token(
            'NEGATION',
            r'!',
            precedence=from_sequence,
            associativity=Associativity.RIGHT,
            category='LOGICAL_OPERATOR',
        ),
        Token(
            'INDICATOR',
            r'I_',
            is_reserved=True,
            precedence=from_sequence,
            category='FUNCTION',
        ),
        Token(
            'INTERSECTION_REDUCE',
            r'AND_',
            is_reserved=True,
            precedence=from_sequence,
            category='FUNCTION',
        ),
        Token(
            'UNION_REDUCE',
            r'OR_',
            is_reserved=True,
            precedence=from_sequence,
            category='FUNCTION',
        ),
        Token(
            'NEGATION_SURFACE',
            r'NOT_',
            is_reserved=True,
            precedence=from_sequence,
            category='FUNCTION',
        ),
        Token(
            'SCATTER',
            r'\:\:\:',
            precedence=1,
            associativity=Associativity.RIGHT,
            category='OPERATOR',
        ),
    )

    production_rules: Tuple[ProductionRule, ...] = (
        ProductionRule(
            'p_expression_union',
            'expression : expression UNION expression',
            binop_infix(UNION),
        ),
        ProductionRule(
            'p_expression_intersection',
            'expression : expression INTERSECTION expression',
            binop_infix(INTERSECTION),
        ),
        ProductionRule(
            'p_expression_negation',
            'expression : NEGATION expression',
            unop_prefix(NEGATION),
        ),
        ProductionRule(
            'p_expression_indicator',
            'expression : INDICATOR parameter',
            unop_prefix(INDICATOR),
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
            unop_prefix(SCATTER),
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
            category='PARAMETER',
        ),
        Token(
            'end_param',
            r'\}\}',
            function=pop_state_and_return,
            precedence=from_sequence,
            category='PARAMETER',
        ),
        Token(
            'ARG_SEP',
            r';',
            state='param',
            precedence=from_sequence,
            associativity=Associativity.LEFT,
            category='DELIMITER',
        ),
        Token(
            'KV_SEP',
            r'=',
            state='param',
            precedence=from_sequence,
            associativity=Associativity.LEFT,
            category='ASSIGNMENT',
        ),
    )

    states: Tuple[Tuple[str, str], ...] = (
        ('param', 'inclusive'),
    )

    production_rules: Tuple[ProductionRule, ...] = (
        ProductionRule(
            'p_expression_parameter',
            'parameter : LBRACKET expression RBRACKET',
            enter_group(),
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
            binop_infix(ASSIGNMENT),
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
            category='IDENTIFIER',
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
            category='OPERATOR',
        ),
        Token(
            'POWER_INCLUSIVE',
            r'\^\^',
            precedence=from_sequence,
            associativity=Associativity.LEFT,
            category='OPERATOR',
        ),
        Token(
            'BACKDIFF',
            r'd_',
            is_reserved=True,
            precedence=from_sequence,
            category='FUNCTION',
        ),
        Token(
            'BACKDIFF_INCLUSIVE',
            r'dd_',
            is_reserved=True,
            precedence=from_sequence,
            category='FUNCTION',
        ),
        Token(
            'FIRST_N',
            r'n_',
            is_reserved=True,
            precedence=from_sequence,
            category='FUNCTION',
        ),
        Token(
            'CUMUL_VAR',
            r'v_',
            is_reserved=True,
            precedence=from_sequence,
            category='FUNCTION',
        ),
    )

    production_rules: Tuple[ProductionRule, ...] = (
        ProductionRule(
            'p_expression_power',
            'expression : expression POWER expression',
            binop_infix(POWER),
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
            unop_prefix(FIRST_N),
        ),
        ProductionRule(
            'p_expression_cumul_var',
            'expression : CUMUL_VAR parameter',
            unop_prefix(CUMUL_VAR),
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
                    'PARAMETER': "Invalid parameter syntax",
                    'IDENTIFIER': "Invalid variable name",
                    'OPERATOR': "Invalid operator usage",
                    'LOGICAL_OPERATOR': "Invalid logical operator usage",
                    'CONDITION': "Invalid condition expression",
                    'PARENTHESIS': "Mismatched parentheses",
                    'BRACKET': "Mismatched brackets",
                    'BRACE': "Mismatched braces",
                    'LITERAL': "Invalid number format",
                    'FUNCTION': "Invalid function usage",
                    'DELIMITER': "Invalid delimiter usage",
                    'ASSIGNMENT': "Invalid assignment syntax",
                    'RESERVED': "Invalid use of reserved word",
                }
            )
        )
