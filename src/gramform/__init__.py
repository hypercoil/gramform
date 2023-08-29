# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
Formula interfaces
"""
from .grammar import (
    Grammar,
)

try:
    from .dfops import (
        ConfoundFormulaGrammar,
        ColumnSelectInterpreter,
        DeduplicateRootNode,
    )
except ImportError:
    pass

try:
    from .imops import (
        ImageMathsGrammar,
        NiftiFileInterpreter,
        NiftiFileSink,
        NiftiObjectInterpreter,
        NiftiObjectSink,
    )
except ImportError:
    pass

from .tagops import (
    DataTagGrammar,
    TagSelectInterpreter,
    ReturnSelection,
)
