# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for data tag grammar
"""
from pkg_resources import resource_filename
from gramform.tagops import DataTagGrammar


def dataset():
    return {
        'a': 'A',
        'b': 'B',
        'c': 'C',
        'd': 'D',
        'e': 'E',
    }


def tags():
    return {
        'a': 'a',
        'b': 'b',
        'c': 'c',
        'd': 'd',
        'e': 'e',
        'ab': {'a', 'b'},
        'bc': {'b', 'c'},
        'cd': {'c', 'd'},
        'de': {'d', 'e'},
        'abc': {'a', 'b', 'c'},
        'bcd': {'b', 'c', 'd'},
        'cde': {'c', 'd', 'e'},
        'abcd': {'a', 'b', 'c', 'd'},
        'bcde': {'b', 'c', 'd', 'e'},
        'abcde': {'a', 'b', 'c', 'd', 'e'},
    }


def test_tags():
    grammar = DataTagGrammar()
    f = grammar.compile('~a&bcd')
    assert(set(f(tags(), **dataset()).keys()) == {'b', 'c', 'd'})

    f = grammar.compile('~bcd|!a')
    assert(set(f(tags(), **dataset()).keys()) == {'a', 'b', 'c', 'd', 'e'})

    f = grammar.compile('~bcd&~a')
    assert(set(f(tags(), **dataset()).keys()) == {'e'})

    g = grammar.compile('~(bcd|a)')
    assert(
        set(f(tags(), **dataset()).keys()) ==
        set(g(tags(), **dataset()).keys())
    )

    f = grammar.compile('~(abc^bcde)^bcd')
    assert(set(f(tags(), **dataset()).keys()) == {'d'})
