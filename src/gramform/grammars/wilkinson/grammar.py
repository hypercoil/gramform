# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Wilkinson Grammar
~~~~~~~~~~~~~~~~~
Comprehensive grammar for Wilkinson notation supporting statistical modeling.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

from gramform.core import (
    binop_infix,
    enter_group,
    config_primitives,
    literal,
    lift_literal,
    named_function_bind,
    parameterised_named_function_bind,
    unit_lift,
    unop_prefix,
    precedence_from_sequence,
    Associativity,
    DynamicGrammar,
    GrammarComponent,
    ProductionRule,
    Primitive as CorePrimitive,
    Token,
)


class OperationalLevel(Enum):
    FACTOR = 'factor'
    TERM = 'term'
    TERMS = 'terms'
    NONE = 'none'


Primitive, prim_registry = config_primitives()


# Base primitives
NUMERIC_LITERAL = Primitive("NUMERIC_LITERAL", is_terminal=True)
APPEND = Primitive("APPEND", is_associative=True)
REMOVE = Primitive("REMOVE", is_associative=False)
INTERACTION = Primitive("INTERACTION", is_associative=True)
NESTED = Primitive("NESTED", is_associative=False)
POWER = Primitive("POWER", is_associative=True)
VARIABLE = Primitive("VARIABLE", is_terminal=True)
NAMED_FUNCTION = Primitive("NAMED_FUNCTION")
EXECUTE = Primitive("EXECUTE", is_terminal=True)
VARIABLE_COMPLEMENT = Primitive("VARIABLE_COMPLEMENT", is_terminal=True)
UNARY_NEGATION = Primitive("UNARY_NEGATION", is_terminal=True)
FUNCTION_PARAMETER = Primitive("FUNCTION_PARAMETER", is_terminal=True)
FUNCTION_PARAMETERS = Primitive("FUNCTION_PARAMETERS", is_associative=True)


TOKEN_PRECEDENCE = (
    'DOT',
    'PARAM_SEPARATOR',
    ('APPEND', 'REMOVE'),
    ('INTERACTIONS', 'INTERACTION_ONLY', 'NESTED'),
    ('POWER_CARET', 'POWER_DOUBLE_ASTERISK'),
    ('LPAREN', 'RPAREN'),
    'ASSIGN',
    ('NAME', 'NAME_LITERAL'),
    'EVAL_FUNC_ONLY_NAME',
    'STRING_LITERAL',
    'INTEGER_LITERAL',
    'FLOAT_LITERAL',
    'EXECUTE',
)
from_sequence, with_precedence = precedence_from_sequence(TOKEN_PRECEDENCE)


def power_ast(expr, _, power):
    """Construct power AST."""
    return APPEND.bind(
        expr,
        *[
            INTERACTION.bind(*([expr] * i))
            for i in range(2, power.value + 1)
        ],
    )


def dot_named_function_bind(prim: CorePrimitive, *pparams):
    """
    Pattern:
    construct : DOT name LPAREN construct RPAREN
    """
    def _inner(_, name, __, expr, ___):
        return prim.bind(name, expr, *pparams)
    return _inner


def parameterised_dot_named_function_bind(prim: CorePrimitive, *pparams):
    """
    Pattern:
    construct : DOT name LPAREN construct parameters RPAREN
    """
    def _inner(_, name, __, expr, ___, parameters, ____):
        return prim.bind(name, expr, parameters, *pparams)
    return _inner


@dataclass(frozen=True)
class LiteralTerminalsComponent(GrammarComponent):
    """Component for literal terminals."""
    tokens: Tuple[Token, ...] = (
        # Numbers
        Token(
            'FLOAT_LITERAL',
            r'\d+\.\d*',
            precedence=from_sequence,
            category='LITERAL',
        ),
        Token(
            'INTEGER_LITERAL',
            r'\d+',
            precedence=from_sequence,
            category='LITERAL',
        ),
        # Names
        Token(
            'STRING_LITERAL',
            r'"[^"]*"|\'[^\']*\'',
            precedence=from_sequence,
            category='LITERAL',
        ),
    )

    production_rules: Tuple[ProductionRule, ...] = (
        ProductionRule(
            'lift_float',
            'factor : FLOAT_LITERAL',
            lift_literal(float, NUMERIC_LITERAL),
        ),
        ProductionRule(
            'lift_integer',
            'factor : INTEGER_LITERAL',
            lift_literal(int, NUMERIC_LITERAL),
        ),
        ProductionRule(
            'lift_string',
            'factor : STRING_LITERAL',
            literal(str),
        ),
    )


@dataclass(frozen=True)
class BasicOperatorsComponent(GrammarComponent):
    """Component for basic operators."""
    tokens: Tuple[Token, ...] = (
        # Basic operators
        Token(
            'APPEND',
            r'\+',
            precedence=from_sequence,
            associativity=Associativity.LEFT,
            category='OPERATOR',
        ),
        Token(
            'REMOVE',
            r'\-',
            precedence=from_sequence,
            associativity=Associativity.LEFT,
            category='OPERATOR',
        ),
        Token(
            'INTERACTIONS',
            r'\*',
            precedence=from_sequence,
            associativity=Associativity.LEFT,
            category='OPERATOR',
        ),
        Token(
            'INTERACTION_ONLY',
            r'\:',
            precedence=from_sequence,
            associativity=Associativity.LEFT,
            category='OPERATOR',
        ),
        Token(
            'NESTED',
            r'\/',
            precedence=from_sequence,
            associativity=Associativity.LEFT,
            category='OPERATOR',
        ),
        Token(
            'NESTED_IN',
            r'\%in\%',
            precedence=from_sequence,
            associativity=Associativity.LEFT,
            category='OPERATOR',
        ),
        Token(
            'POWER_CARET',
            r'\^',
            precedence=from_sequence,
            associativity=Associativity.RIGHT,
            category='OPERATOR',
        ),
        Token(
            'POWER_DOUBLE_ASTERISK',
            r'\*\*',
            precedence=from_sequence,
            associativity=Associativity.RIGHT,
            category='OPERATOR',
        ),

        # Parentheses and brackets
        Token(
            'LPAREN',
            r'\(',
            precedence=from_sequence,
            category='PARENTHESIS',
        ),
        Token(
            'RPAREN',
            r'\)',
            precedence=from_sequence,
            category='PARENTHESIS',
        ),

        # Whitespace
        Token('ignore', ' \t'),
    )

    production_rules: Tuple[ProductionRule, ...] = (
        ProductionRule(
            'expression_append',
            'expression : expression APPEND term',
            binop_infix(APPEND),
        ),
        ProductionRule(
            'expression_remove',
            'expression : expression REMOVE term',
            binop_infix(REMOVE),
        ),
        ProductionRule(
            'expression_unary_append',
            'expression : APPEND term',
            unop_prefix(APPEND),
        ),
        ProductionRule(
            'expression_unary_remove',
            'term : REMOVE term',
            unop_prefix(UNARY_NEGATION),
        ),
        ProductionRule(
            'term_interaction',
            'term : term INTERACTIONS factor',
            lambda left, _, right: APPEND.bind(
                left, right, INTERACTION.bind(left, right)
            ),
        ),
        ProductionRule(
            'term_interaction_only',
            'term : term INTERACTION_ONLY factor',
            binop_infix(INTERACTION),
        ),
        ProductionRule(
            'term_nested',
            'term : term NESTED factor',
            binop_infix(NESTED),
        ),
        ProductionRule(
            'term_nested_in',
            'term : term NESTED_IN factor',
            lambda left, _, right: NESTED.bind(right, left),
        ),
        ProductionRule(
            'factor_power_caret',
            'factor : factor POWER_CARET factor',
            power_ast,
        ),
        ProductionRule(
            'factor_power_double_asterisk',
            'factor : factor POWER_DOUBLE_ASTERISK factor',
            power_ast,
        ),
        ProductionRule(
            'lift_term',
            'expression : term',
            unit_lift(),
        ),
        ProductionRule(
            'lift_factor',
            'term : factor',
            unit_lift(),
        ),
        ProductionRule(
            'factor_parentheses',
            'factor : LPAREN expression RPAREN',
            enter_group(),
        ),
    )


@dataclass(frozen=True)
class NamesComponent(GrammarComponent):
    """Component for names."""
    tokens: Tuple[Token, ...] = (
        Token(
            'NAME',
            r'[a-zA-Z_][a-zA-Z0-9_]*',
            precedence=from_sequence,
            category='NAME',
        ),
        Token(
            'NAME_LITERAL',
            r'`[^`]*`',
            precedence=from_sequence,
            category='NAME',
        ),
        Token(
            'DOT',
            r'\.',
            precedence=from_sequence,
            category='OPERATOR',
        ),
    )

    production_rules: Tuple[ProductionRule, ...] = (
        ProductionRule(
            'factor_variable',
            'factor : NAME',
            lambda terminal: VARIABLE.bind(terminal),
        ),
        ProductionRule(
            'lift_name',
            'factor : NAME_LITERAL',
            # Remove backticks
            lambda terminal: VARIABLE.bind(terminal[1:-1]),
        ),
        ProductionRule(
            'factor_variable_complement',
            'factor : DOT',
            lambda terminal: VARIABLE_COMPLEMENT.bind(terminal),
        ),
    )


@dataclass(frozen=True)
class ExecutionComponent(GrammarComponent):
    """Component for Python code execution (delimited in curly braces)."""
    tokens: Tuple[Token, ...] = (
        Token(
            'EXECUTE',
            r'\{[^\}]+\}',
            precedence=from_sequence,
            category='EXECUTION',
        ),
        Token(
            'EVAL_FUNC_ONLY_NAME',
            r'^[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*$',
            precedence=from_sequence,
            category='NAME',
        ),
        Token(
            'ASSIGN',
            r'\=',
            precedence=from_sequence,
            category='OPERATOR',
        ),
        Token(
            'PARAM_SEPARATOR',
            r'\,',
            precedence=from_sequence,
            category='OPERATOR',
        ),
    )

    production_rules: Tuple[ProductionRule, ...] = (
        ProductionRule(
            'factor_execute',
            'factor : EXECUTE',
            # Remove curly braces
            lambda terminal: EXECUTE.bind(terminal[1:-1]),
        ),
        ProductionRule(
            'factor_named_function',
            'factor : NAME LPAREN expression RPAREN',
            named_function_bind(NAMED_FUNCTION, OperationalLevel.TERM),
        ),
        ProductionRule(
            'factor_named_function_eval_only',
            'factor : EVAL_FUNC_ONLY_NAME LPAREN expression RPAREN',
            named_function_bind(NAMED_FUNCTION, OperationalLevel.TERM),
        ),
        ProductionRule(
            'factor_named_function_factor_level',
            'factor : DOT NAME LPAREN expression RPAREN',
            dot_named_function_bind(NAMED_FUNCTION, OperationalLevel.FACTOR),
        ),
        ProductionRule(
            'factor_named_function_eval_only_factor_level',
            'factor : DOT EVAL_FUNC_ONLY_NAME LPAREN expression RPAREN',
            dot_named_function_bind(NAMED_FUNCTION, OperationalLevel.FACTOR),
        ),
        ProductionRule(
            'parameter_assign',
            'parameter : NAME ASSIGN expression',
            binop_infix(FUNCTION_PARAMETER),
        ),
        ProductionRule(
            'parameters_lift_parameter',
            'parameters : parameter',
            unit_lift(),
        ),
        ProductionRule(
            'parameters_append_parameter',
            'parameters : parameters PARAM_SEPARATOR parameter',
            binop_infix(FUNCTION_PARAMETERS),
        ),
        ProductionRule(
            'factor_named_function_parameterised',
            (
                'factor : '
                'NAME LPAREN expression PARAM_SEPARATOR parameters RPAREN'
            ),
            parameterised_named_function_bind(
                NAMED_FUNCTION,
                OperationalLevel.TERM,
            ),
        ),
        ProductionRule(
            'factor_named_function_eval_only_parameterised',
            (
                'factor : '
                'EVAL_FUNC_ONLY_NAME '
                'LPAREN expression PARAM_SEPARATOR parameters RPAREN'
            ),
            parameterised_named_function_bind(
                NAMED_FUNCTION,
                OperationalLevel.TERM,
            ),
        ),
        ProductionRule(
            'factor_named_function_factor_level_parameterised',
            (
                'factor : '
                'DOT NAME '
                'LPAREN expression PARAM_SEPARATOR parameters RPAREN'
            ),
            parameterised_dot_named_function_bind(
                NAMED_FUNCTION,
                OperationalLevel.FACTOR,
            ),
        ),
        ProductionRule(
            'factor_named_function_eval_only_factor_level_parameterised',
            (
                'factor : '
                'DOT EVAL_FUNC_ONLY_NAME '
                'LPAREN expression PARAM_SEPARATOR parameters RPAREN'
            ),
            parameterised_dot_named_function_bind(
                NAMED_FUNCTION,
                OperationalLevel.FACTOR,
            ),
        ),
    )


class WilkinsonGrammar(DynamicGrammar):
    """Grammar for Wilkinson notation."""
    def __init__(self):
        super().__init__(
            components=(
                LiteralTerminalsComponent(),
                BasicOperatorsComponent(),
                NamesComponent(),
                ExecutionComponent(),
            ),
        )


if __name__ == "__main__":
    grammar = WilkinsonGrammar()
    result = grammar.parse("dog + cat + (rat*dog + cat:dog)^2")
    print(result)
    assert 0
