"""
Tests for regex sampler.
"""
import pytest
from gramform.resampler import generate_valid_completion

def test_simple_literal():
    assert generate_valid_completion("a") == "a"

def test_alternation():
    assert generate_valid_completion("(a|b)") in ["a", "b"]

def test_optional():
    assert generate_valid_completion("a?") == "a"

def test_star():
    assert generate_valid_completion("a*") == "a"

def test_plus():
    assert generate_valid_completion("a+") == "a"

def test_sequence():
    assert generate_valid_completion("abc") == "abc"

def test_complex_pattern():
    result = generate_valid_completion("(foo|bar)baz+(bat)*")
    assert result.startswith(("foo", "bar"))
    assert "baz" in result
    assert result.endswith("baz") or result.endswith("bat")

def test_nested_groups():
    assert generate_valid_completion("(a(b|c))") in ["ab", "ac"]

def test_character_class():
    assert generate_valid_completion("[abc]") in ["a", "b", "c"]

def test_negated_character_class():
    result = generate_valid_completion("[^abc]")
    assert result not in ["a", "b", "c"]
    assert len(result) == 1

def test_quantifiers():
    assert generate_valid_completion("a{2,4}") == "aa"
    assert generate_valid_completion("a{2}") == "aa"

def test_escape_sequences():
    assert generate_valid_completion(r"\d") in "0123456789"
    assert generate_valid_completion(r"\w") in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"

def test_anchors():
    assert generate_valid_completion("^a") == "a"
    assert generate_valid_completion("a$") == "a"

def test_dot():
    result = generate_valid_completion(".")
    assert len(result) == 1
    assert result != "\n"

def test_empty_pattern():
    assert generate_valid_completion("") == ""

def test_invalid_pattern():
    with pytest.raises(Exception):
        generate_valid_completion("(")  # Unclosed parenthesis

# New comprehensive test cases
def test_range():
    assert generate_valid_completion("[a-z]") in "abcdefghijklmnopqrstuvwxyz"
    assert generate_valid_completion("[0-9]") in "0123456789"

def test_branch():
    result = generate_valid_completion("(foo|bar)baz+(qux)*")
    assert result.startswith(("foo", "bar"))
    assert "baz" in result
    assert result.endswith("baz") or result.endswith("qux")

def test_anchors_ignored():
    assert generate_valid_completion("^a$") == "a"

def test_complex_regex():
    result = generate_valid_completion("(a|b)(c|d)(e|f)")
    assert len(result) == 3
    assert result[0] in "ab"
    assert result[1] in "cd"
    assert result[2] in "ef"
