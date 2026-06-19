# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
``nwx`` smooth-term interpreter (Phase 4)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Lowers the GAM/GAMM smooth constructors ``s`` / ``te`` / ``ti`` / ``t2`` (spec
§4.3) onto :class:`~gramform.grammars.nwx.spec.SmoothSpec`. These names are
special **only in call position** (a bare ``s`` stays a column lookup, R6); the
calls reuse the existing parameterised ``NAME(...)`` productions, so this
module adds no grammar -- it registers a handler into the shared
``NAMED_FUNCTION_HANDLERS`` table. Like ``noise()`` / random effects, a smooth
is routed structurally out of the fixed-term stream into ``NwxState.smooth``
(returning no fixed term), which ``LHS_RHS_STRUCTURE`` consumes into
``ModelSpec.smooth``. Imports neither ``jax`` nor ``nitrix``.

::

    s(age, k=6, bs="cr", by=dx)   1-D penalised smooth; by -> factor/varycoef
    te(x, z, k=5)                 anisotropic tensor product (tensor=True)
    ti(x, z)                      pure interaction tensor
    s(g, bs="re")                 random intercept AS a smooth (GAMM bridge)
    s(g, x, bs="re")              random slope of x by g (slope var -> by)

``by_kind`` (factor vs continuous) is **data-dependent** -- a Wilkinson surface
cannot tell whether ``dx`` is categorical -- so it is left ``None`` here and
resolved by the engine / validator (spec §4.3, §8). Plain ``bs`` / ``ns`` /
``poly`` are *not* smooths; they remain data transforms (a later phase).
"""

from gramform.core import Literal, Primitive
from gramform.grammars.nwx.spec import (
    BasisKind,
    FactorSpec,
    SmoothSpec,
    TermSpec,
)
from gramform.grammars.nwx.transform import (
    NAMED_FUNCTION_HANDLERS,
    NwxContext,
    NwxError,
)

#: Smooth constructors (special only in call position).
SMOOTH_CONSTRUCTORS = frozenset({'s', 'te', 'ti', 't2'})

#: ``bs=`` basis string -> ``BasisKind``. All of these ship in nitrix v3
#: (ps/cc/tp/te v1; cr/gp/mrf §3.2; the GAMM-bridge re/fs §2/§3.1).
_BASIS = {
    'tp': BasisKind.TPRS,
    'tprs': BasisKind.TPRS,
    'ts': BasisKind.TPRS,
    'ps': BasisKind.PS,
    'cc': BasisKind.CC,
    'cp': BasisKind.PS,
    'cr': BasisKind.CR,
    'cs': BasisKind.CR,
    'gp': BasisKind.GP,
    'mrf': BasisKind.MRF,
    're': BasisKind.RE,
    'fs': BasisKind.FS,
}
#: Cyclic ``bs=`` strings (carry a period; not auto-periodised to the data).
_CYCLIC = frozenset({'cc', 'cp'})
#: GAMM-bridge bases whose second positional arg is a slope variable (-> by).
_RE_BASES = frozenset({BasisKind.RE, BasisKind.FS})
_TRUE = frozenset({'TRUE', 'True', 'true', 'T', 'yes'})
_DEFAULT_K = 10


def _scalar(ast: object) -> int | float | str:
    """A smooth parameter value AST -> a Python scalar / name."""
    if isinstance(ast, Literal):
        value = ast.value
        if isinstance(value, str):
            return value.strip('"\'')
        if isinstance(value, (int, float)):
            return value
    elif isinstance(ast, Primitive):
        if ast.name == 'NUMERIC_LITERAL':
            value = ast.value
            if isinstance(value, (int, float)):
                return value
        elif ast.name in ('VARIABLE', 'EXECUTE'):
            params = ast.get_parameters()
            if isinstance(params, str):
                return params
    raise NwxError(f'unsupported smooth parameter value {ast!r}')


def _collect_params(
    ast: object,
) -> tuple[list[object], dict[str, object]]:
    """Flatten a ``parameters`` AST into positional value-ASTs + named args."""
    positional: list[object] = []
    named: dict[str, object] = {}

    def walk(node: object) -> None:
        if isinstance(node, Primitive):
            if node.name == 'FUNCTION_PARAMETERS':
                for child in node.parameters:
                    walk(child)
                return
            if node.name == 'PARAMETER':
                positional.append(node.parameters[0])
                return
            if node.name == 'NAMED_PARAMETER':
                named[node.parameters[0]] = node.parameters[1]
                return
        positional.append(node)

    walk(ast)
    return positional, named


def _eval_factor(
    ast: object,
    context: NwxContext,
) -> tuple[NwxContext, FactorSpec]:
    """Evaluate a covariate / by-variable AST into a single factor."""
    if not isinstance(ast, Primitive):
        raise NwxError(f'expected a covariate expression, got {ast!r}')
    context = ast(context)
    return context, _as_factor(context.get_result())


def _as_factor(result: object) -> FactorSpec:
    """Coerce an evaluated covariate expression into a single factor."""
    if isinstance(result, FactorSpec):
        return result
    if isinstance(result, TermSpec) and len(result.factors) == 1:
        return result.factors[0]
    if isinstance(result, tuple) and len(result) == 1:
        return _as_factor(result[0])
    raise NwxError(
        f'a smooth covariate must be a single factor, got {result!r}'
    )


def build_smooth(node: Primitive, context: NwxContext) -> NwxContext:
    name = node.parameters[0]
    # parameters = (name, first_expr, [params], OperationalLevel) -- drop the
    # name and the trailing operational-level marker.
    args = node.parameters[1:-1]
    first_cov, *rest = args
    positional, named = _collect_params(rest[0]) if rest else ([], {})

    # Evaluate the smoothed covariates (the first positional arg + any further
    # positional args) into factors.
    covariates: list[FactorSpec] = []
    for cov_ast in [first_cov, *positional]:
        context, factor = _eval_factor(cov_ast, context)
        covariates.append(factor)

    tensor = name in ('te', 'ti', 't2')
    bs = named.get('bs')
    if bs is not None:
        key = str(_scalar(bs))
        if key not in _BASIS:
            raise NwxError(f'unknown smooth basis bs={key!r}')
        basis = _BASIS[key]
        cyclic = key in _CYCLIC
    else:
        basis = BasisKind.TENSOR if tensor else BasisKind.TPRS
        cyclic = False

    by = None
    by_kind = None  # data-dependent (factor vs continuous): engine resolves.
    # For `re`/`fs` GAMM bridges, the second positional arg is the slope var.
    if basis in _RE_BASES and len(covariates) > 1:
        by = covariates[1]
        covariates = covariates[:1]
    elif 'by' in named:
        context, by = _eval_factor(named['by'], context)

    k = int(_scalar(named['k'])) if 'k' in named else _DEFAULT_K
    penalty_order = int(_scalar(named['m'])) if 'm' in named else 2
    fx = str(_scalar(named['fx'])) in _TRUE if 'fx' in named else False

    spec = SmoothSpec(
        covariates=tuple(covariates),
        basis=basis,
        k=k,
        penalty_order=penalty_order,
        by=by,
        by_kind=by_kind,
        cyclic=cyclic,
        tensor=tensor,
        fx=fx,
    )

    # All nwx smooth bases ship in nitrix v3 (ps/cc/tp/te, cr/gp/mrf, re/fs),
    # so no backend-awareness warning is raised here.
    context = context.update_state(smooth=context.state.smooth + (spec,))
    # A smooth contributes no fixed term.
    return context.with_result(())


for _smooth_name in SMOOTH_CONSTRUCTORS:
    NAMED_FUNCTION_HANDLERS[_smooth_name] = build_smooth
