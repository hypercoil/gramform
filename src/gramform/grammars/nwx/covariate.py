# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
``nwx`` covariate program (minimal, Phase 1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The :data:`CovariateProgram` is a closed union of :data:`CovariateOp`. ``nwx``
*emits* a program; the engine materialises columns (spec decision 4). This
module ships the Phase-1 subset the runnable slice exercises -- the 36P-style
confound vocabulary: shorthand expansions, backward differences, and inclusive
numeric power. The full vocabulary (``CompCorSelect``, ``Indicator``,
``SetOp``, ``Scatter``) lands in Phase 6.

The source of truth for the surfaces is ``grammars/minimaltest/`` (the ``ply``
port of the confound vocabulary), transcribed in
``docs/nwx/covariate-vocabulary.md`` -- **not** the deleted ``dfops.py``.

The Wilkinson term algebra owns the shared glyphs (``^`` = crossing-power,
``-`` = term removal, ``||`` = uncorrelated random effects); the confound layer
contributes only its non-colliding operators (``^^`` numeric power, ``d_``/
``dd_`` differences).
"""

from dataclasses import dataclass

# --- shorthand preprocessor expansions (from minimaltest) ------------------
# ``csf`` is intentionally absent: it is a passthrough column name, NOT a
# shorthand (a common slip the legacy README implied otherwise). An
# unrecognised name is a plain column lookup, never an error.
SHORTHAND_EXPANSIONS: dict[str, tuple[str, ...]] = {
    'wm': ('white_matter',),
    'gsr': ('global_signal',),
    'gs': ('global_signal',),
    'rps': (
        'trans_x',
        'trans_y',
        'trans_z',
        'rot_x',
        'rot_y',
        'rot_z',
    ),
    'fd': ('framewise_displacement',),
    'dv': ('std_dvars',),
    'acc': ('a_comp_cor',),
    'wcc': ('w_comp_cor',),
    'ccc': ('c_comp_cor',),
}


def is_shorthand(name: str) -> bool:
    """Whether ``name`` is a recognised confound shorthand (``csf`` is not)."""
    return name in SHORTHAND_EXPANSIONS


@dataclass(frozen=True)
class Shorthand:
    """A preprocessor expansion of a confound shorthand (e.g. ``rps``)."""

    name: str

    @property
    def expansion(self) -> tuple[str, ...]:
        return SHORTHAND_EXPANSIONS[self.name]


@dataclass(frozen=True)
class Derivative:
    """
    Backward (temporal) difference of confound columns.

    ``d_`` is the exclusive difference (``inclusive=False``: orders
    ``1..order``); ``dd_`` is inclusive (``inclusive=True``: orders
    ``0..order``, retaining the originals). The engine materialises the
    differenced columns.
    """

    operands: tuple[str, ...]
    order: int = 1
    inclusive: bool = False


@dataclass(frozen=True)
class Power:
    """
    Inclusive numeric power of confound columns (the ``^^`` surface).

    ``inclusive`` distinguishes ``^^`` (keep the original, orders
    ``1..order``) from an exclusive power. This is the *numeric* power, kept
    distinct from the Wilkinson crossing-power ``^`` (see the vocabulary doc).
    """

    operands: tuple[str, ...]
    order: int = 1
    inclusive: bool = True


# Closed union (match-exhaustive), extended in Phase 6. The engine consumes
# these; nobody adds a variant without an engine that understands it.
CovariateOp = Shorthand | Derivative | Power
CovariateProgram = tuple[CovariateOp, ...]
