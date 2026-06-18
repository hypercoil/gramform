# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
nwx firewall (post-Phase-0 invariant): the gramform grammar surface must not
transitively import ``jax`` or ``nitrix``. The model-specification layer is
pure; numerics live in a separate engine (see ``docs/nwx/spec.md`` §1).

Run in a subprocess so the check is unaffected by other tests that may have
imported ``jax``/``nitrix`` into the session.
"""

import subprocess
import sys

_PROBE = """
import importlib, sys

MODULES = [
    "gramform.core",
    "gramform.grammars.wilkinson.transform",
    "gramform.grammars.minimaltest.transform",
]
# Forward-compat: include the nwx surface once it exists (Phase 1+).
try:
    importlib.import_module("gramform.grammars.nwx.transform")
except ModuleNotFoundError:
    pass

for mod in MODULES:
    importlib.import_module(mod)

forbidden = sorted(m for m in ("jax", "jaxlib", "nitrix") if m in sys.modules)
assert not forbidden, f"forbidden imports pulled in: {forbidden}"
print("clean")
"""


def test_grammar_surface_is_jax_and_nitrix_free():
    result = subprocess.run(
        [sys.executable, '-c', _PROBE],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f'firewall probe failed:\n{result.stdout}\n{result.stderr}'
    )
    assert 'clean' in result.stdout
