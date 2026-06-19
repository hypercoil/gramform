# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
DataFrame / confound-vocabulary grammar.

A ``ply`` grammar for DataFrame column expressions and a compact
confound-formula vocabulary whose shorthands expand *to* fMRIPrep / BIDS-style
column names; the source of truth harvested by the ``nwx`` covariate layer (see
``docs/nwx/covariate-vocabulary.md``). Formerly ``minimaltest``.
"""
