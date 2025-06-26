"""
Tests for the minimal grammar implementation.
"""
import pytest
from gramform.grammars.minimaltest.grammar import (
    MinimalGrammar,
)
from gramform.grammars.minimaltest.transform import (
    DataFrameContext,
    get_processor,
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
    context = DataFrameContext()
    result, context = ppr_associative_flatten(result, context)
    result, context = ppr_common_subexpression(result, context)

    # Verify the structure of the parsed expression
    assert result.name == 'CONCATENATE'
    assert len(result.parameters) == 5
    assert tuple(e.name for e in result.parameters) == (
        'POWER',
        'VARIABLE',
        'VARIABLE',
        'VARIABLE',
        'POWER',
    )


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
    context = DataFrameContext()
    result, context = ppr_associative_flatten(result, context)
    result, context = ppr_common_subexpression(result, context)

    # Verify the structure
    assert result.name == 'CONCATENATE'
    assert len(result.parameters) == 3
    assert tuple(e.name for e in result.parameters) == (
        'POWER', 'INDICATOR', 'BACKDIFF'
    )


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
    context = DataFrameContext()
    result, context = ppr_associative_flatten(result, context)
    result, context = ppr_common_subexpression(result, context)

    # Verify the structure
    assert result.name == 'CONCATENATE'
    assert len(result.parameters) == 3
    assert tuple(e.name for e in result.parameters) == (
        'SCATTER', 'INTERSECTION_REDUCE', 'CUMUL_VAR'
    )


def test_common_subexpression_elimination():
    """Test common subexpression elimination."""
    expr = '(x+y+z)^^2 + (x+y+z)^^2 + (x+y+z)^^2'
    grammar = MinimalGrammar()
    parser = grammar._parser

    # Test parsing and postprocessing
    result = parser.parse(expr)
    context = DataFrameContext()
    result, context = ppr_associative_flatten(result, context)
    result, context = ppr_common_subexpression(result, context)

    # Verify that common subexpressions were eliminated
    assert result.name == 'CONCATENATE'
    assert len(result.parameters) == 3
    assert tuple(e.name for e in result.parameters) == (
        'POWER', 'POWER', 'POWER'
    )


def test_backdiff_with_common_subexpressions():
    """Test backdiff operations with common subexpressions."""
    expr = 'd_[1]((x+y)^^2 + (x+y)^^2) + d_[1]((x+y)^^2 + (x+y)^^2)'
    grammar = MinimalGrammar()
    parser = grammar._parser

    # Test parsing and postprocessing
    result = parser.parse(expr)
    context = DataFrameContext()
    result, context = ppr_associative_flatten(result, context)
    result, context = ppr_common_subexpression(result, context)

    # Verify the structure and optimization
    assert result.name == 'CONCATENATE'
    assert len(result.parameters) == 2


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
    context = DataFrameContext()
    result, context = ppr_associative_flatten(result, context)
    result, context = ppr_common_subexpression(result, context)

    assert result.name == 'CONCATENATE'
    assert len(result.parameters) == 5
    assert tuple(e.name for e in result.parameters) == (
        'INDICATOR',
        'BACKDIFF',
        'INTERSECTION_REDUCE',
        'UNION_REDUCE',
        'INDICATOR',
    )


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


def test_parser_error_reporting():
    """Test that parser errors provide detailed context and suggestions."""
    grammar = MinimalGrammar()
    parser = grammar._parser

    # Test missing operand (EOF error)
    with pytest.raises(ValueError) as exc_info:
        parser.parse("x + ")
    error_msg = str(exc_info.value)
    assert "Unexpected end of input" in error_msg
    assert "Last valid token was '+' of type 'CONCATENATE'" in error_msg
    assert "Expected one of:" in error_msg
    # Check for expected tokens based on LALR(1) state analysis
    assert any(token in error_msg for token in ['VARIABLE', 'INTEGER', 'FLOAT', 'LPAREN'])

    # Test mismatched parentheses
    with pytest.raises(ValueError) as exc_info:
        parser.parse("x + (y + )")
    error_msg = str(exc_info.value)
    assert "Syntax error at line 1, column 10:" in error_msg
    assert "x + (y + )" in error_msg
    assert "         ^" in error_msg
    assert "Expected one of:" in error_msg
    # Should show valid completions based on state analysis
    assert (
        "Valid completions could be:" in error_msg.replace('\n', '').replace('  ', '') or
        "Valid completions could be:" in error_msg
    )

    # Test invalid operator sequence
    with pytest.raises(ValueError) as exc_info:
        parser.parse("x + + y")
    error_msg = str(exc_info.value)
    assert "Syntax error at line 1, column 5:" in error_msg
    assert "x + + y" in error_msg
    assert "    ^" in error_msg
    assert "Unexpected token '+'" in error_msg
    assert "Expected one of:" in error_msg

    # Test invalid parameter syntax
    with pytest.raises(ValueError) as exc_info:
        parser.parse("I_[x = = y]")
    error_msg = str(exc_info.value)
    assert "Syntax error" in error_msg
    assert "Unexpected token '='" in error_msg
    assert "Expected one of:" in error_msg

    # Test invalid condition syntax
    with pytest.raises(ValueError) as exc_info:
        parser.parse("x = = y")
    error_msg = str(exc_info.value)
    assert "Syntax error" in error_msg
    assert "Unexpected token '='" in error_msg
    assert "Expected one of:" in error_msg


def test_error_recovery_suggestions():
    """Test that error messages provide helpful recovery suggestions."""
    grammar = MinimalGrammar()
    parser = grammar._parser

    # Test incomplete expression (EOF error)
    with pytest.raises(ValueError) as exc_info:
        parser.parse("x + (y +")
    error_msg = str(exc_info.value)
    assert "Unexpected end of input" in error_msg
    assert "Expected one of:" in error_msg
    assert "Valid completions could be:" in error_msg
    # Should show recovery suggestions
    assert "Recovery:" in error_msg

    # Test invalid operator usage (use invalid operator)
    with pytest.raises(ValueError) as exc_info:
        parser.parse("x ** y")
    error_msg = str(exc_info.value)
    assert "Lexical error" in error_msg
    assert "Illegal character '*'" in error_msg

    # Test invalid parameter usage
    # with pytest.raises(ValueError) as exc_info:
    #     parser.parse("I_[x;y]")
    # error_msg = str(exc_info.value)
    # assert "Syntax error" in error_msg
    # assert "Unexpected token ';'" in error_msg
    # assert "Expected one of:" in error_msg

    # Test EOF error
    with pytest.raises(ValueError) as exc_info:
        parser.parse("x + (y + z")
    error_msg = str(exc_info.value)
    assert "Unexpected end of input" in error_msg
    assert "Last valid token was" in error_msg
    assert "Expected one of:" in error_msg


def test_multiline_error_reporting():
    """Test error reporting with multiline input."""
    grammar = MinimalGrammar()
    parser = grammar._parser

    # Test error in multiline expression (lexical error due to newline)
    with pytest.raises(ValueError) as exc_info:
        parser.parse("""
            x + y
            z + @ w
            a + b
        """)
    error_msg = str(exc_info.value)
    assert "Lexical error at line 2, column 0:" in error_msg
    assert "x + y" in error_msg or "z + @ w" in error_msg

    # Test error in nested expression (lexical error due to newline)
    with pytest.raises(ValueError) as exc_info:
        parser.parse("""
            x + (
                y + )
            )
        """)
    error_msg = str(exc_info.value)
    assert "Lexical error" in error_msg
    # Lexical errors don't include parser-specific sections like "Valid completions"


def test_error_context_specificity():
    """Test that error messages are specific to the context."""
    grammar = MinimalGrammar()
    parser = grammar._parser

    # Test parameter context (use invalid syntax)
    with pytest.raises(ValueError) as exc_info:
        parser.parse("I_[x = = y]")
    error_msg = str(exc_info.value)
    # The context message should be about condition expression, not parameter syntax
    assert "Invalid condition expression" in error_msg

    # Test condition context
    with pytest.raises(ValueError) as exc_info:
        parser.parse("x = = y")
    error_msg = str(exc_info.value)
    assert "Invalid condition expression" in error_msg

    # Test operator context
    with pytest.raises(ValueError) as exc_info:
        parser.parse("x + + y")
    error_msg = str(exc_info.value)
    assert "Invalid operator usage" in error_msg


def test_lalr1_state_analysis():
    """Test that error analysis uses LALR(1) state machine information."""
    grammar = MinimalGrammar()
    parser = grammar._parser

    # Test that error analysis provides state-specific information
    with pytest.raises(ValueError) as exc_info:
        parser.parse("x + (y + )")
    error_msg = str(exc_info.value)

    # Should show valid tokens based on current state
    assert "Expected one of:" in error_msg

    # Should show valid completions based on state stack analysis
    assert (
        "Valid completions could be:" in error_msg.replace('\n', '').replace('  ', '') or
        "Valid completions could be:" in error_msg
    )


def test_recovery_strategies():
    """Test that recovery strategies are attempted and reported."""
    grammar = MinimalGrammar()
    parser = grammar._parser

    # Test panic mode recovery suggestion (use valid syntax error)
    with pytest.raises(ValueError) as exc_info:
        parser.parse("x + (y + z && w")  # Missing closing parenthesis
    error_msg = str(exc_info.value)
    # Should suggest skipping until synchronization token
    assert "Recovery:" in error_msg or "Expected one of:" in error_msg

    # Test phrase level recovery
    with pytest.raises(ValueError) as exc_info:
        parser.parse("x + (y + z")
    error_msg = str(exc_info.value)
    # Should suggest completing the expression
    assert (
        "Valid completions could be:" in error_msg.replace('\n', '').replace('  ', '') or
        "Valid completions could be:" in error_msg
    )


def test_error_analysis_components():
    """Test that error analysis components work correctly."""
    grammar = MinimalGrammar()
    parser = grammar._parser

    # Test that ParseState is properly updated
    assert hasattr(grammar.error_handler, '_parse_state')
    assert hasattr(grammar.error_handler, '_error_analyzer')
    assert hasattr(grammar.error_handler, '_recovery_strategy')

    # Test that error analysis provides structured information
    with pytest.raises(ValueError) as exc_info:
        parser.parse("x + (y + )")
    error_msg = str(exc_info.value)

    # Should contain structured error information
    assert "Syntax error at line" in error_msg
    assert "Unexpected token" in error_msg
    assert "Expected one of:" in error_msg


def test_token_error_reporting():
    """Test that token errors provide detailed context."""
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


def test_basic_processor():
    import pandas as pd
    processor = get_processor()
    result = processor.process('d_[1]((x+y)^^2 + (x+y)^^2)')
    result = processor(
        'dd_[3]((x+y)^2,4-5 + (x+y)^2,4-5)',
        data=pd.DataFrame(
            {'x': [1, 2, 3], 'y': [4, 5, 6]},
            index=[1, 2, 3],
        ),
    )
    result = processor(
        'NOT_((x=y && x=z) || x>=w) + AND_(I_[x=y] + I_[x=z] + OR_(I_[x=w] + I_[x=v]))',
        data=pd.DataFrame(
            {'x': [1, 2, 3], 'y': [3, 2, 1], 'z': [2, 2, 2], 'w': [0, 2, 3], 'v': [1, 0, 0]},
            index=[1, 2, 3],
        ),
    )
