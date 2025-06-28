# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Wilkinson Grammar
~~~~~~~~~~~~~~~~~
Comprehensive grammar for Wilkinson notation supporting statistical modeling.
"""
from dataclasses import dataclass
from typing import Tuple

from gramform.core import (
    binop_infix,
    enter_group,
    config_primitives,
    literal,
    named_function_bind,
    unit_lift,
    precedence_from_sequence,
    Associativity,
    DynamicGrammar,
    GrammarComponent,
    ProductionRule,
    Token,
)


Primitive, prim_registry = config_primitives()


# Base primitives
CONCATENATE = Primitive("CONCATENATE", is_associative=True)
REMOVAL = Primitive("REMOVAL", is_associative=False)
INTERACTION = Primitive("INTERACTION", is_associative=True)
NESTED = Primitive("NESTED", is_associative=False)
POWER = Primitive("POWER", is_associative=True)
VARIABLE = Primitive("VARIABLE", is_terminal=True)
NAMED_FUNCTION = Primitive("NAMED_FUNCTION")


TOKEN_PRECEDENCE = (
    ('CONCATENATE', 'REMOVAL'),
    ('INTERACTIONS', 'INTERACTION_ONLY', 'NESTED'),
    ('POWER_CARET', 'POWER_DOUBLE_ASTERISK'),
    ('LPAREN', 'RPAREN'),
    ('NAME', 'NAME_LITERAL'),
    'STRING_LITERAL',
    'INTEGER_LITERAL',
    'FLOAT_LITERAL',
)
from_sequence, with_precedence = precedence_from_sequence(TOKEN_PRECEDENCE)


def power_ast(expr, _, power):
    """Construct power AST."""
    return CONCATENATE.bind(
        expr,
        *[
            INTERACTION.bind(*([expr] * i))
            for i in range(2, power.value + 1)
        ],
    )


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
            literal(float),
        ),
        ProductionRule(
            'lift_integer',
            'factor : INTEGER_LITERAL',
            literal(int),
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
            'CONCATENATE',
            r'\+',
            precedence=from_sequence,
            associativity=Associativity.LEFT,
            category='OPERATOR',
        ),
        Token(
            'REMOVAL',
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
            'expression_concatenate',
            'expression : expression CONCATENATE term',
            binop_infix(CONCATENATE),
        ),
        ProductionRule(
            'expression_removal',
            'expression : expression REMOVAL term',
            binop_infix(REMOVAL),
        ),
        ProductionRule(
            'term_interaction',
            'term : term INTERACTIONS factor',
            lambda left, _, right: CONCATENATE.bind(
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
            'factor_named_function',
            'factor : NAME LPAREN expression RPAREN',
            named_function_bind(NAMED_FUNCTION),
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
            ),
        )


if __name__ == "__main__":
    grammar = WilkinsonGrammar()
    result = grammar.parse("dog + cat + (rat*dog + cat:dog)^2")
    print(result)
    assert 0
