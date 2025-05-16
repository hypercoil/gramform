# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Postprocessors
~~~~~~~~~~~~~~
Postprocessors are used to transform the tree after it has been parsed.
"""


def ppr_associative_flatten(tree):
    def _flatten(children, to_flatten):
        for child, flatten in zip(children, to_flatten):
            if flatten:
                yield from child.parameters
            else:
                yield child

    if tree.is_terminal:
        return tree
    children = [
        ppr_associative_flatten(child)
        for child in tree.parameters
    ]
    to_flatten = [
        getattr(child, 'name', None) == tree.name
        and tree.is_associative
        for child in children
    ]
    children = tuple(_flatten(children, to_flatten))
    return tree.bind(*children)


def ppr_common_subexpression(tree):
    subexpressions = {}
    # TODO: This is a cache, but it's not used.
    # We don't use this cache, but it's here in case it simplifies a
    # future implementation. If not, we should remove it.
    cache = set()

    def _collect(tree, subexpressions):
        children = []
        for child in tree.parameters:
            if child in subexpressions and not child.is_terminal:
                children.append(subexpressions[child])
                cache.add(child)
            else:
                if not child.is_terminal:
                    child, subexpressions = _collect(child, subexpressions)
                    subexpressions[child] = child
                children.append(child)

        return tree.bind(*children), subexpressions

    tree, _ = _collect(tree, subexpressions)
    return tree