# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Tests for Wilkinson formula parser against formulaic ground truth.
"""
import pytest
import pandas as pd
import formulaic
from gramform.grammars.wilkinson.transform import get_processor


def compare_formulas(our_result, formulaic_result, comparison_type="exact"):
    """Compare our result with formulaic result using specified comparison type."""
    if comparison_type == "structured":
        return our_result == formulaic_result
    our_terms = list(our_result)
    formulaic_terms = list(formulaic_result)
    if comparison_type == "exact":
        # Exact match using formulaic.Formula comparison
        return formulaic.Formula(our_terms) == formulaic.Formula(formulaic_terms)
    elif comparison_type == "set":
        # Set-based comparison (order-independent)
        return set(our_terms) == set(formulaic_terms)
    else:
        raise ValueError(f"Unknown comparison type: {comparison_type}")


def _test_formula_equivalence(wilkinson_expr: str, formulaic_expr: str = None, comparison_type: str = "exact"):
    """Test that our Wilkinson parser produces the same result as formulaic."""
    if formulaic_expr is None:
        formulaic_expr = wilkinson_expr

    # Our Wilkinson parser
    processor = get_processor()
    our_result = processor(wilkinson_expr)

    # Formulaic's parser (ground truth)
    try:
        formulaic_result = formulaic.Formula(formulaic_expr)
    except Exception as e:
        pytest.fail(f"Formulaic failed to parse '{formulaic_expr}': {e}")

    # Compare results
    match = compare_formulas(our_result, formulaic_result, comparison_type)

    if not match:
        our_terms = list(our_result)
        formulaic_terms = list(formulaic_result)
        pytest.fail(
            f"Formula mismatch for '{wilkinson_expr}':\n"
            f"  Our result: {our_terms}\n"
            f"  Formulaic: {formulaic_terms}\n"
            f"  Comparison type: {comparison_type}"
        )


# Test cases for basic operations
BASIC_OPERATIONS = [
    ("x + y", "x + y"),
    ("x:y", "x:y"),
    ("x^2", "x^2"),
    ("dog + cat", "dog + cat"),
    ("rat*dog", "rat*dog"),
    ("cat:dog", "cat:dog"),
    ("3:x:2", "6:x"),
    ("x * y + 2:(x + z)", "y + x:y + 2:(x + z)"),
]


@pytest.mark.parametrize("wilkinson_expr,formulaic_expr", BASIC_OPERATIONS)
def test_basic_operations(wilkinson_expr, formulaic_expr):
    """Test basic Wilkinson operations."""
    _test_formula_equivalence(wilkinson_expr, formulaic_expr)


# Test cases for complex expressions
COMPLEX_EXPRESSIONS = [
    ("(rat*dog + cat:dog)^2", "(rat*dog + cat:dog)^2", "exact"),
    ("dog + cat + (rat*dog + cat:dog)^2", "dog + cat + (rat*dog + cat:dog)^2", "exact"),
    ("(x + (y + z + z:w)^2)^3", "(x + (y + z + z:w)^2)^3", "set"),
    ("(x + y + y:z)^3", "(x + y + y:z)^3", "set"),
]


@pytest.mark.parametrize("wilkinson_expr,formulaic_expr,comparison_type", COMPLEX_EXPRESSIONS)
def test_complex_expressions(wilkinson_expr, formulaic_expr, comparison_type):
    """Test complex Wilkinson expressions."""
    _test_formula_equivalence(wilkinson_expr, formulaic_expr, comparison_type)


# Test cases for associativity and precedence
ASSOCIATIVITY_PRECEDENCE = [
    ("x + y - x - 1", "x + y - x - 1"),
    ("x - 1 - y - 0", "x - 1 - y - 0"),
    ("x + (y + 0)", "x + y + 0"),  # + is associative in ours
    ("x + -1 + y + 0", "x + -1 + y + 0"),
    ("(a + b + c) / (m + n) / (w + x + y + z)", "(a + b + c) / (m + n) / (w + x + y + z)"),
]


@pytest.mark.parametrize("wilkinson_expr,formulaic_expr", ASSOCIATIVITY_PRECEDENCE)
def test_associativity_and_precedence(wilkinson_expr, formulaic_expr):
    """Test associativity and precedence rules."""
    _test_formula_equivalence(wilkinson_expr, formulaic_expr)


# Test cases for function calls (using set comparison for the last two)
FUNCTION_CALLS = [
    ("y + bs(x) + bs(z) + bs(lag({w*x}))", "y + bs(x) + bs(z) + bs(lag(w*x))", "exact"),
    ("y + bs(x) + bs(z) + bs(.lag(w*x))", "y + bs(x) + bs(z) + bs(lag(w)) + bs(lag(x)) + bs(lag(w)*lag(x))", "set"),
    ("bs(x * y, df=4, degree=3)", "bs(x, df=4, degree=3) + bs(y, df=4, degree=3) + bs(x * y, df=4, degree=3)", "set"),
]


@pytest.mark.parametrize("wilkinson_expr,formulaic_expr,comparison_type", FUNCTION_CALLS)
def test_function_calls(wilkinson_expr, formulaic_expr, comparison_type):
    """Test function calls and special syntax."""
    _test_formula_equivalence(wilkinson_expr, formulaic_expr, comparison_type)


STRUCTURED_FORMULAE = [
    ("y ~ x", "y ~ x", "structured"),
    ("y ~ x + z", "y ~ x + z", "structured"),
    ("y ~ x + z | w + v", "y ~ x + z | w + v", "structured"),
    (
        "y ~ dog + cat + (rat*dog + cat:dog)^2 | v + bs(x) + bs(z) + bs(lag({w*x}))",
        "y ~ dog + cat + (rat*dog + cat:dog)^2 | v + bs(x) + bs(z) + bs(lag(w*x))",
        "structured"
    ),
]


@pytest.mark.parametrize("wilkinson_expr,formulaic_expr,comparison_type", STRUCTURED_FORMULAE)
def test_structured_formulae(wilkinson_expr, formulaic_expr, comparison_type):
    """Test structured formulae."""
    _test_formula_equivalence(wilkinson_expr, formulaic_expr, comparison_type)


def test_errors():
    """Test that errors are raised for invalid formulas."""
    processor = get_processor()
    with pytest.raises(ValueError):
        processor('1:x + 2:x')


def test_model_matrix_generation():
    """Test model matrix generation with sample data."""
    # Sample data
    data = pd.DataFrame({
        "x": [0., -4., 5., -2.],
        "y": ["cat", "cat", "dog", "cat"],
        "z": [4, 4, 12, 1],
        "w": [3., 6., 9., -1.]
    })

    # Test formula with various operations
    formula_str = 'y + np.abs(x) + bs(x) + bs(z) + bs(lag(w*x, 1))'

    # Test with formulaic
    formulaic_formula = formulaic.Formula(formula_str)
    formulaic_matrix = formulaic_formula.get_model_matrix(data)

    # Verify formulaic works
    assert formulaic_matrix.shape[0] == len(data) - 1 # Subtract one for lag
    assert len(formulaic_matrix.columns) > 0

    # Test with our parser
    processor = get_processor()
    our_result = processor(formula_str)

    # Verify our parser produces a result
    assert our_result is not None
    assert len(list(our_result)) > 0


def test_processor_creation():
    """Test that the processor can be created successfully."""
    processor = get_processor()
    assert processor is not None
    assert hasattr(processor, 'transform')


def test_simple_variable():
    """Test simple variable parsing."""
    processor = get_processor()
    result = processor('x')
    assert result is not None
    assert len(list(result)) == 2 # specified term + intercept


def test_simple_concatenation():
    """Test simple concatenation parsing."""
    processor = get_processor()
    result = processor('x + y')
    assert result is not None
    assert len(list(result)) >= 2


def test_interaction():
    """Test interaction parsing."""
    processor = get_processor()
    result = processor('x:y')
    assert result is not None
    assert len(list(result)) >= 1


def test_power_operation():
    """Test power operation parsing."""
    processor = get_processor()
    result = processor('x^2')
    assert result is not None
    assert len(list(result)) >= 1


def test_nested_expression():
    """Test nested expression parsing."""
    processor = get_processor()
    result = processor('(x + y) * z')
    assert result is not None
    assert len(list(result)) >= 1


def test_function_with_parameters():
    """Test function with parameters parsing."""
    processor = get_processor()
    result = processor('bs(x, df=3)')
    assert result is not None
    assert len(list(result)) >= 1


def test_complex_nested_expression():
    """Test complex nested expression parsing."""
    processor = get_processor()
    result = processor('(x + y)^2 + (a + b):(c + d)')
    assert result is not None
    assert len(list(result)) >= 1


def test_empty_expression():
    """Test handling of empty expression."""
    processor = get_processor()
    with pytest.raises(Exception):
        processor('')


def test_invalid_syntax():
    """Test handling of invalid syntax."""
    processor = get_processor()
    with pytest.raises(Exception):
        processor('x + + y')  # Double operator


# Stand-in test: we should add model matrix eval here
def test_missing_variable():
    """Test handling of missing variable reference."""
    processor = get_processor()
    # This should work as we're just parsing, not evaluating
    result = processor('nonexistent_var')
    assert result is not None


def test_special_characters():
    """Test handling of special characters in variable names."""
    processor = get_processor()
    result = processor('`var with spaces`')
    assert result is not None


def test_numeric_literals():
    """Test handling of numeric literals."""
    processor = get_processor()
    result = processor('x + 1 + 2.5')
    assert result is not None
    assert len(list(result)) >= 2


# String literals are not supported yet.
# def test_string_literals():
#     """Test handling of string literals."""
#     processor = get_processor()
#     result = processor('x + "string_literal"')
#     assert result is not None
#     assert len(list(result)) >= 2
