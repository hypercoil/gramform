"""
Tests for the minimal grammar implementation.
"""
import pytest
from gramform.core import TransformationContext, ppr_execution_head
from gramform.grammar.minimal import (
    MinimalGrammar,
)
from gramform.postprocessors import (
    ppr_associative_flatten,
    ppr_common_subexpression,
)


def test_basic_arithmetic():
    """Test basic arithmetic operations."""
    expr = '(x+y+z)^^2+(x+y+z)+((x+y+z)^2+(x+y+z))^3.13-5'
    grammar = MinimalGrammar()
    lexer = grammar._lexer
    parser = grammar._parser

    # Test lexing
    lexer.input(expr)
    tokens = list(lexer)
    assert len(tokens) > 0
    assert any(t.type == 'POWER_INCLUSIVE' for t in tokens)
    assert any(t.type == 'POWER' for t in tokens)

    # Test parsing and postprocessing
    result = parser.parse(expr)
    context = TransformationContext()
    result, context = ppr_associative_flatten(result, context)
    result, context = ppr_common_subexpression(result, context)
    result, context = ppr_execution_head(result, context)

    # Verify the structure of the parsed expression
    assert result.name == 'EXECUTION_HEAD'
    assert len(result.parameters) == 1


def test_indicators_and_backdiff():
    """Test indicator and backdiff operations."""
    expr = '(x+y+z)^2-3 + I_[x=y] + d_[1,4-5](x)'
    grammar = MinimalGrammar()
    lexer = grammar._lexer
    parser = grammar._parser

    # Test lexing
    lexer.input(expr)
    tokens = list(lexer)
    assert any(t.type == 'INDICATOR' for t in tokens)
    assert any(t.type == 'BACKDIFF' for t in tokens)

    # Test parsing and postprocessing
    result = parser.parse(expr)
    context = TransformationContext()
    result, context = ppr_associative_flatten(result, context)
    result, context = ppr_common_subexpression(result, context)
    result, context = ppr_execution_head(result, context)

    # Verify the structure
    assert result.name == 'EXECUTION_HEAD'
    assert len(result.parameters) == 1


def test_complex_boolean_operations():
    """Test complex boolean operations with indicators and reductions."""
    expr = ':::!((I_[x=y] && I_[x=z]) || I_[x>=w]) + AND_(I_[x=y] + I_[x=z] + OR_(I_[x=w] + I_[x=v])) + v_{{test; x=1; y=2; z=3}}'
    grammar = MinimalGrammar()
    lexer = grammar._lexer
    parser = grammar._parser

    # Test lexing
    lexer.input(expr)
    tokens = list(lexer)
    assert any(t.type == 'SCATTER' for t in tokens)
    assert any(t.type == 'INTERSECTION_REDUCE' for t in tokens)
    assert any(t.type == 'UNION_REDUCE' for t in tokens)

    # Test parsing and postprocessing
    result = parser.parse(expr)
    context = TransformationContext()
    result, context = ppr_associative_flatten(result, context)
    result, context = ppr_common_subexpression(result, context)
    result, context = ppr_execution_head(result, context)

    # Verify the structure
    assert result.name == 'EXECUTION_HEAD'
    assert len(result.parameters) == 1


def test_common_subexpression_elimination():
    """Test common subexpression elimination."""
    expr = '(x+y+z)^^2 + (x+y+z)^^2 + (x+y+z)^^2'
    grammar = MinimalGrammar()
    parser = grammar._parser

    # Test parsing and postprocessing
    result = parser.parse(expr)
    context = TransformationContext()
    result, context = ppr_associative_flatten(result, context)
    result, context = ppr_common_subexpression(result, context)
    result, context = ppr_execution_head(result, context)

    # Verify that common subexpressions were eliminated
    assert result.name == 'EXECUTION_HEAD'
    assert len(result.parameters) == 1


def test_backdiff_with_common_subexpressions():
    """Test backdiff operations with common subexpressions."""
    expr = 'd_[1]((x+y)^^2 + (x+y)^^2) + d_[1]((x+y)^^2 + (x+y)^^2)'
    grammar = MinimalGrammar()
    parser = grammar._parser

    # Test parsing and postprocessing
    result = parser.parse(expr)
    context = TransformationContext()
    result, context = ppr_associative_flatten(result, context)
    result, context = ppr_common_subexpression(result, context)
    result, context = ppr_execution_head(result, context)

    # Verify the structure and optimization
    assert result.name == 'EXECUTION_HEAD'
    assert len(result.parameters) == 1


def test_invalid_expressions():
    """Test handling of invalid expressions."""
    invalid_exprs = [
        'x +',  # Incomplete expression
        'x + + y',  # Double operator
        'x + (y',  # Unmatched parenthesis
        'x + [y]',  # Invalid bracket usage
    ]

    grammar = MinimalGrammar()
    parser = grammar._parser

    for expr in invalid_exprs:
        with pytest.raises(Exception):
            parser.parse(expr)


def test_reserved_words():
    """Test handling of reserved words."""
    expr = 'I_[x=y] + d_[1](x) + AND_(y) + OR_(z) + NOT_(w)'
    grammar = MinimalGrammar()
    lexer = grammar._lexer
    parser = grammar._parser

    # Test lexing
    lexer.input(expr)
    tokens = list(lexer)
    assert any(t.type == 'INDICATOR' for t in tokens)
    assert any(t.type == 'BACKDIFF' for t in tokens)
    assert any(t.type == 'INTERSECTION_REDUCE' for t in tokens)
    assert any(t.type == 'UNION_REDUCE' for t in tokens)

    # Test parsing
    result = parser.parse(expr)
    context = TransformationContext()
    result, context = ppr_associative_flatten(result, context)
    result, context = ppr_common_subexpression(result, context)
    result, context = ppr_execution_head(result, context)

    assert result.name == 'EXECUTION_HEAD'
    assert len(result.parameters) == 1


def test_lexer_error_reporting():
    """Test that lexer errors provide detailed context."""
    grammar = MinimalGrammar()
    lexer = grammar._lexer

    # Test invalid character
    with pytest.raises(ValueError) as exc_info:
        lexer.input("x + @ y")
        list(lexer)
    error_msg = str(exc_info.value)
    assert "Lexical error at line 1, column 5:" in error_msg
    assert "x + @ y" in error_msg
    assert "    ^" in error_msg
    assert "Illegal character '@'" in error_msg

    # Test invalid number format
    with pytest.raises(ValueError) as exc_info:
        lexer.input("x + 1.2.3")
        list(lexer)
    error_msg = str(exc_info.value)
    assert "Lexical error at line 1, column 8:" in error_msg
    assert "x + 1.2.3" in error_msg
    assert "       ^" in error_msg

    # Test invalid variable name
    with pytest.raises(ValueError) as exc_info:
        lexer.input("x + 1@var")
        list(lexer)
    error_msg = str(exc_info.value)
    assert "Lexical error at line 1, column 6:" in error_msg
    assert "x + 1@var" in error_msg
    assert "     ^" in error_msg
