# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
`re`-sampling
~~~~~~~~~~~~~
Create an example string that matches a given regular expression.

This module largely uses internal APIs from the `re` module. Specifically,
it uses the `re._parser` module to parse the regex pattern into an abstract
syntax tree (AST), and then uses the `generate_from_ast` function to generate
a string that matches the regex pattern.

Given the dependency on internal APIs, we might want to entirely integrate
the regex parsing from Secret Labs AB here:
Constants: https://github.com/python/cpython/blob/main/Lib/re/_constants.py
Parser: https://github.com/python/cpython/blob/main/Lib/re/_parser.py
This depends on C extensions in _sre, which are also presumably subject to
change without notice; the potential difficulty of integrating these is the
reason we're just using internal APIs here.
"""
from re._parser import parse as parse_regex
from re._parser import (
    LITERAL,
    SUBPATTERN,
    BRANCH,
    MAX_REPEAT,
    IN,
    RANGE,
    NEGATE,
    ANY,
    AT,
    CATEGORY,
)
from typing import List, Tuple
import random
import string


class RegexNode:
    """Base class for regex AST nodes."""
    def generate(self) -> str:
        raise NotImplementedError


class LiteralNode(RegexNode):
    def __init__(self, char: int):
        self.char = char

    def generate(self) -> str:
        return chr(self.char)


class SubpatternNode(RegexNode):
    def __init__(self, pattern: List[Tuple]):
        self.pattern = pattern

    def generate(self) -> str:
        return "".join(generate_from_ast(self.pattern))


class BranchNode(RegexNode):
    def __init__(self, branches: List[List[Tuple]]):
        self.branches = branches

    def generate(self) -> str:
        # Randomly select one branch
        branch = random.choice(self.branches)
        return "".join(generate_from_ast(branch))


class MaxRepeatNode(RegexNode):
    def __init__(self, min_repeat: int, max_repeat: int, pattern: List[Tuple]):
        self.min_repeat = min_repeat
        self.max_repeat = max_repeat
        self.pattern = pattern

    def generate(self) -> str:
        # For *, +, ?, {n}, {n,}, {n,m}
        if self.min_repeat == 0 and self.max_repeat == 1:  # ?
            return "".join(generate_from_ast(self.pattern))
        elif self.min_repeat == 0 and self.max_repeat == -1:  # *
            return "".join(generate_from_ast(self.pattern))
        elif self.min_repeat == 1 and self.max_repeat == -1:  # +
            return "".join(generate_from_ast(self.pattern))
        else:  # {n} or {n,m}
            n = self.min_repeat
            return "".join(generate_from_ast(self.pattern) * n)


class InNode(RegexNode):
    def __init__(self, items: List[Tuple]):
        self.items = items

    def generate(self) -> str:
        if not self.items:
            return ""

        # Handle character classes
        if len(self.items) == 1 and self.items[0][0] == NEGATE:
            # Negated character class
            negated_chars = set()
            for item in self.items[0][1]:
                if item[0] == RANGE:
                    start, end = item[1]
                    negated_chars.update(chr(i) for i in range(start, end + 1))
                else:
                    negated_chars.add(chr(item[1]))

            # Find a character not in the negated set
            for char in string.printable:
                if char not in negated_chars:
                    return char
            return " "  # Fallback
        else:
            # Regular character class
            valid_chars = []
            for item in self.items:
                if item[0] == RANGE:
                    start, end = item[1]
                    valid_chars.extend(chr(i) for i in range(start, end + 1))
                else:
                    valid_chars.append(chr(item[1]))
            return random.choice(valid_chars)


class AnyNode(RegexNode):
    def generate(self) -> str:
        return random.choice(string.printable.replace("\n", ""))


class CategoryNode(RegexNode):
    def __init__(self, category: int):
        self.category = category

    def generate(self) -> str:
        if self.category == 4:  # \d
            return random.choice(string.digits)
        elif self.category == 2:  # \w
            return random.choice(string.ascii_letters + string.digits + "_")
        elif self.category == 3:  # \s
            return random.choice(string.whitespace)
        else:
            return " "  # Fallback


def generate_from_ast(ast: List[Tuple], depth=0) -> str:
    indent = '  ' * depth
    # print(f"{indent}AST: {ast}")
    result = []
    for node in ast:
        t = node[0]
        v = node[1]
        # print(f"{indent}Processing node: {node}")
        if t == LITERAL:
            result.append(chr(v))
        elif t == SUBPATTERN:
            # v = (groupnum, add_flags, del_flags, subpattern)
            result.append(generate_from_ast(v[3], depth+1))
        elif t == BRANCH:
            # v = (None, [branch1, branch2, ...])
            # Pick the first branch for determinism
            result.append(generate_from_ast(v[1][0], depth+1))
        elif t == MAX_REPEAT:
            # v = (min_repeat, max_repeat, subpattern)
            min_repeat, max_repeat, subpattern = v
            n = min_repeat if min_repeat > 0 else 1
            for _ in range(n):
                result.append(generate_from_ast(subpattern, depth+1))
        elif t == IN:
            # v = list of (LITERAL, x) or (RANGE, (a, b)) or (NEGATE, None)
            if v and v[0][0] == NEGATE:
                negated = set()
                for item in v[1:]:
                    if item[0] == LITERAL:
                        negated.add(chr(item[1]))
                    elif item[0] == RANGE:
                        a, b = item[1]
                        negated.update(chr(i) for i in range(a, b+1))
                for c in string.printable:
                    if c not in negated and c != '\n':
                        result.append(c)
                        break
            else:
                for item in v:
                    if item[0] == LITERAL:
                        result.append(chr(item[1]))
                        break
                    elif item[0] == RANGE:
                        a, _ = item[1]
                        result.append(chr(a))
                        break
        elif t == ANY:
            result.append(next(c for c in string.printable if c != '\n'))
        elif t == CATEGORY:
            if v == 4:  # \d
                result.append('0')
            elif v == 2:  # \w
                result.append('a')
            elif v == 3:  # \s
                result.append(' ')
            else:
                result.append(' ')
        elif t == AT:
            continue
        else:
            # print(f"{indent}Unknown node type: {t}")
            continue
    joined = ''.join(result)
    # print(f"{indent}Result at depth {depth}: {joined}")
    return joined


def generate_valid_completion(regex: str) -> str:
    """
    Generate a valid completion for a given regex pattern.

    Parameters
    ----------
    regex: str
        A regular expression pattern

    Returns
    -------
    str
        A string that matches the regex pattern

    Raises
    ------
    Exception
        If the regex pattern is invalid
    """
    if not regex:
        return ""

    try:
        ast = parse_regex(regex)
        # print(f"Top-level AST for pattern '{regex}': {ast}")
        return generate_from_ast(ast)
    except Exception as e:
        raise Exception(f"Invalid regex pattern: {str(e)}")
