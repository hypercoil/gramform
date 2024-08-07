# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Noxfile
"""
import nox

@nox.session()
def clean(session):
    session.install('coverage[toml]')
    session.run('coverage', 'erase')

@nox.session(python=["3.10", "3.11"])
def tests(session):
    session.install('jax[cpu]')
    session.install('.[full,dev]')
    session.run(
        'pytest',
        '--cov', 'gramform',
        '--cov-append',
        'tests/',
    )
    session.run('ruff', 'check', 'src/gramform')
    session.run('blue', '--check', 'src/gramform')

@nox.session()
def report(session):
    session.install('coverage[toml]')
    session.run(
        'coverage',
        'report', '--fail-under=90',
        "--omit='*test*,*__init__*'",
    )
    session.run(
        'coverage',
        'html',
        "--omit='*test*,*__init__*'",
    )
    session.run(
        'coverage',
        'xml',
        "--omit='*test*,*__init__*'",
    )
