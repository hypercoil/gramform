# -*- coding: utf-8 -*-
"""
Make the engine importable as ``nwx_reference_engine`` and ``gramform``
importable from ``src`` when these tests are run directly (the engine is a
standalone consumer, not part of the gramform package install).
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
for path in (_REPO / 'src', _REPO / 'examples'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
