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

@nox.session(python=["3.12", "3.13", "3.14"])
def tests(session):
    session.install('.[dev]')
    session.run(
        'pytest',
        '--cov', 'gramform',
        '--cov-append',
        'tests/',
    )
    session.run('ruff', 'check', 'src/gramform')
    session.run('ruff', 'format', '--check', 'src/gramform')


@nox.session()
def typecheck(session):
    """Type-check the nwx IR/contract surface (no-op until nwx exists)."""
    import os
    if not os.path.isdir('src/gramform/grammars/nwx'):
        session.skip('nwx package not present yet')
    session.install('.[dev]')
    session.run('pyright', 'src/gramform/grammars/nwx')

@nox.session()
def report(session):
    session.install('coverage[toml]')
    # Ratcheting floor: now that the typed, well-tested nwx surface has landed
    # (phases 1-7), the total sits at ~78%. Raise this toward the sibling
    # standard (90+) as the legacy `grammars/dataframe` narwhals materialiser
    # (~31%, the largest uncovered surface) gains tests or is retired.
    session.run(
        'coverage',
        'report', '--fail-under=78',
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
