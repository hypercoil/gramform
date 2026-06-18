# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
``nwx`` -- the neuroimaging Wilkinson extension.

A pure model-specification layer: parse a Wilkinson-style formula into an
immutable :class:`~gramform.grammars.nwx.spec.ModelGraph` IR that an external
engine lowers onto ``nitrix``. ``nwx`` never imports ``nitrix`` or ``jax``.

See ``docs/nwx/spec.md`` and ``docs/nwx/implementation-plan.md``.
"""
