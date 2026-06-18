# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
``nwx`` -- the ``ModelSpec`` IR (spec ``docs/nwx/spec.md`` §5)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A typed, immutable intermediate representation of a *model* (not a design
matrix). Every node is ``@dataclass(frozen=True)`` and hashable; collections
are ``tuple`` (no mappings in IR fields -- key/value data is
``tuple[tuple[K, V], ...]``, which is hashable, pytree-clean, and
value-comparable for golden tests). Closed ``Union``s + ``Enum``/``Literal``
describe the ``nwx``-owned variants; the single open, engine-owned seam is the
:class:`Lowerable` ``Protocol``.

``nwx`` defines these dataclasses and the ``Lowerable`` protocol; an external
engine consumes them. That is the only coupling. This module imports neither
``jax`` nor ``nitrix`` (a CI firewall asserts it).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Protocol, runtime_checkable

from gramform.grammars.nwx.covariate import CovariateProgram

# ---------------------------------------------------------------------------
# §5 closed enumerations
# ---------------------------------------------------------------------------


class Level(Enum):
    """Multi-level (node-graph) data-aggregation level."""

    RUN = 'run'
    SESSION = 'session'
    SUBJECT = 'subject'
    DATASET = 'dataset'


class Family(Enum):
    """GL(A)M(M) response family. Members beyond the first three are reserved
    (IR-ready, gated by the nitrix v3 FR), not a v1 promise."""

    GAUSSIAN = 'gaussian'
    BINOMIAL = 'binomial'
    POISSON = 'poisson'
    # reserved (nitrix v3 §4)
    GAMMA = 'gamma'
    NEGBINOMIAL = 'negbinomial'
    TWEEDIE = 'tweedie'
    BETA = 'beta'


class Link(Enum):
    """Link function. Members beyond the first three are reserved."""

    IDENTITY = 'identity'
    LOG = 'log'
    LOGIT = 'logit'
    # reserved
    PROBIT = 'probit'
    INVERSE = 'inverse'
    SQRT = 'sqrt'


class Mode(Enum):
    """Residualisation mode. ``SOFT`` is reserved (nitrix v3 §5.2)."""

    AGGRESSIVE = 'aggressive'
    NONAGGRESSIVE = 'nonaggressive'
    # reserved
    SOFT = 'soft'


class Test(Enum):
    """Contrast test statistic."""

    T = 't'
    F = 'F'


class Relation(Enum):
    """Grouping-factor relation: a single factor, or one interaction grouping
    factor (one variance component over cells). Genuine crossing is *multiple*
    ``RandomEffectSpec`` s, never ``relation='crossed'``."""

    SINGLE = 'single'
    INTERACTION = 'interaction'


class Structure(Enum):
    """Random-effect covariance structure (matches the nitrix v3 ``lme_fit``
    structure-dispatch ladder)."""

    SCALAR = 'scalar'
    DIAGONAL = 'diagonal'
    UNSTRUCTURED = 'unstructured'


class Combine(Enum):
    """Multi-level combination: fixed-effects (precision-weighted, no between
    variance) vs mixed-effects (estimate the between-level variance)."""

    FIXED = 'fixed'
    MIXED = 'mixed'


class BasisKind(Enum):
    """Smooth basis. The first four are shipped; the rest are reserved."""

    PS = 'ps'
    CC = 'cc'
    TPRS = 'tprs'
    TENSOR = 'tensor'
    # reserved
    CR = 'cr'
    GP = 'gp'
    MRF = 'mrf'
    RE = 're'
    FS = 'fs'


class Severity(Enum):
    """Diagnostic severity."""

    ERROR = 'error'
    WARNING = 'warning'


class ContrastCoding(Enum):
    """Categorical factor contrast coding. Default (``None``) defers the
    choice to the engine (treatment coding, conventionally)."""

    TREATMENT = 'treatment'
    SUM = 'sum'
    HELMERT = 'helmert'
    POLY = 'poly'


# ---------------------------------------------------------------------------
# §5 term sources -- a CLOSED union (match-exhaustive), not a Protocol
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Lookup:
    """A column looked up by name in the covariate frame."""

    name: str


@dataclass(frozen=True)
class Const:
    """A numeric literal (an intercept is ``Const(1.0)``)."""

    value: float


@dataclass(frozen=True)
class PyExpr:
    """An inline Python transform (the ``{...}`` execute form)."""

    code: str


@dataclass(frozen=True)
class CovariateRef:
    """A reference into the node's :data:`CovariateProgram` by position."""

    op_index: int


@dataclass(frozen=True)
class Referent:
    """A referent emitted by a nested frame or an inbound multi-level stage.

    ``kind`` is ``'_hat'`` (fitted), ``'_tilde'`` (residualised), or the name
    of an inbound cope; ``stage`` names the producing node.
    """

    stage: str
    kind: str


TermSource = Lookup | Const | PyExpr | CovariateRef | Referent


# ---------------------------------------------------------------------------
# §5 terms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorSpec:
    """A single factor: a term source plus optional categorical coding."""

    source: TermSource
    coding: ContrastCoding | None = None


@dataclass(frozen=True)
class TermSpec:
    """A model term: a product of factors.

    A term's *role* (response, fixed, random, ...) is **structural** -- which
    tuple of the :class:`ModelSpec` it lives in -- never a field. ``order`` is
    a computed property: the count of non-constant factors (an intercept term,
    whose only factor is ``Const(1.0)``, has order 0).
    """

    factors: tuple[FactorSpec, ...]

    @property
    def order(self) -> int:
        return sum(1 for f in self.factors if not isinstance(f.source, Const))

    @property
    def is_intercept(self) -> bool:
        return self.factors == (FactorSpec(Const(1.0)),)


#: The canonical intercept term.
INTERCEPT: TermSpec = TermSpec(factors=(FactorSpec(Const(1.0)),))


# ---------------------------------------------------------------------------
# §5 smooths, random effects, error structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SmoothSpec:
    """A penalised / smooth term (GAM / GAMM)."""

    covariates: tuple[FactorSpec, ...]
    basis: BasisKind
    k: int
    penalty_order: int = 2
    by: FactorSpec | None = None
    by_kind: Literal['factor', 'continuous'] | None = None
    cyclic: bool = False
    tensor: bool = False
    fx: bool = False
    bounds: tuple[float, float] | None = None


@dataclass(frozen=True)
class GroupingSpec:
    """A random-effect grouping factor (a single factor or an interaction
    grouping factor, per :class:`Relation`)."""

    factors: tuple[FactorSpec, ...]
    relation: Relation


@dataclass(frozen=True)
class RandomEffectSpec:
    """A random-effect block: terms varying by a grouping factor, with a
    covariance ``structure`` (scalar / diagonal / unstructured)."""

    group: GroupingSpec
    terms: tuple[TermSpec, ...]
    structure: Structure


@dataclass(frozen=True)
class CorrelationSpec:
    """A within-group correlation structure (AR / CAR / compound symmetry)."""

    kind: Literal['ar1', 'car1', 'cs']
    index: FactorSpec
    group: FactorSpec


@dataclass(frozen=True)
class WeightSpec:
    """A heteroscedasticity (variance) structure."""

    kind: Literal['varIdent', 'varPower']
    arg: FactorSpec


@dataclass(frozen=True)
class ErrorSpec:
    """The residual error / correlation structure."""

    correlation: CorrelationSpec | None = None
    heteroscedasticity: WeightSpec | None = None


# ---------------------------------------------------------------------------
# §5 residualisation, family, contrasts, estimation, inference, response
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResidualiseSpec:
    """A confound-residualisation directive. ``AGGRESSIVE`` projects onto the
    orthogonal complement of ``noise``; ``NONAGGRESSIVE`` preserves variance
    shared with ``signal`` (and requires a non-empty ``signal`` set)."""

    target: tuple[TermSpec, ...]
    noise: tuple[TermSpec, ...]
    signal: tuple[TermSpec, ...] = ()
    mode: Mode = Mode.AGGRESSIVE


@dataclass(frozen=True)
class FamilySpec:
    """The response family and link."""

    family: Family = Family.GAUSSIAN
    link: Link = Link.IDENTITY


@dataclass(frozen=True)
class ContrastSpec:
    """A named estimand: a linear combination of coefficient names + a test.

    Weights are a tuple of ``(coefficient_name, weight)`` pairs (no mapping).
    """

    name: str
    weights: tuple[tuple[str, float], ...]
    test: Test


@dataclass(frozen=True)
class EstimationSpec:
    """The estimator and standard-error / dof options."""

    estimator: Literal['ols', 'wls', 'irls', 'reml', 'ml', 'flame'] = 'ols'
    se: Literal['model', 'robust', 'cluster'] = 'model'
    robust_variant: Literal['hc0', 'hc1', 'hc2', 'hc3'] | None = None
    cluster_by: str | None = None
    dof: Literal['residual', 'satterthwaite', 'kr'] | None = None


@dataclass(frozen=True)
class InferenceSpec:
    """The inference / multiple-comparison procedure."""

    kind: Literal['parametric', 'permutation']
    enhancement: (
        Literal['voxel', 'tfce', 'cluster_extent', 'cluster_mass'] | None
    ) = None
    cluster_threshold: float | None = None
    n_perm: int | None = None
    correction: Literal['fdr', 'bonferroni', 'fwe', 'rft'] | None = None


@dataclass(frozen=True)
class ResponseSpec:
    """The response (LHS): its terms plus auxiliary markers (trials / weights /
    offset) as a tuple of ``(role, factor)`` pairs."""

    terms: tuple[TermSpec, ...]
    aux: tuple[tuple[str, FactorSpec], ...] = ()


# ---------------------------------------------------------------------------
# §5 the ModelSpec and the multi-level graph
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """A complete single-node model specification.

    The *populated-fields -> solver* dispatch is the engine's responsibility
    (spec §9); the IR describes the model, never which ``nitrix`` function
    fires.
    """

    response: ResponseSpec
    fixed: tuple[TermSpec, ...]
    partial: tuple[TermSpec, ...] = ()  # in-model nuisance (noise() on a RHS)
    random: tuple[RandomEffectSpec, ...] = ()
    smooth: tuple[SmoothSpec, ...] = ()
    family: FamilySpec = field(default_factory=FamilySpec)
    errors: ErrorSpec | None = None
    residualise: tuple[ResidualiseSpec, ...] = ()
    estimands: tuple[ContrastSpec, ...] = ()
    estimation: EstimationSpec = field(default_factory=EstimationSpec)
    inference: InferenceSpec | None = None  # node-level
    covariates: CovariateProgram = ()


@dataclass(frozen=True)
class Carry:
    """A quantity propagated along an :class:`Edge`. ``contrast`` binds to an
    upstream :class:`ContrastSpec` name."""

    contrast: str
    quantities: Literal['cope_varcope', 'estimates']


@dataclass(frozen=True)
class Edge:
    """A multi-level dataflow edge between nodes."""

    source: str
    dest: str
    filter: tuple[tuple[str, str], ...]
    carry: Carry


@dataclass(frozen=True)
class ModelNode:
    """A node in the multi-level graph: a named, levelled model."""

    name: str
    level: Level
    group_by: tuple[str, ...]
    combine: Combine
    spec: ModelSpec


@dataclass(frozen=True)
class ModelGraph:
    """The top-level IR: a graph of nodes + dataflow edges. A single-node
    formula lowers to a one-node graph."""

    nodes: tuple[ModelNode, ...]
    edges: tuple[Edge, ...] = ()
    inference: InferenceSpec | None = None  # graph-level (distinct per-node)


# ---------------------------------------------------------------------------
# §5 diagnostics (validation output) and the engine seam
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Diagnostic:
    """A static-validation finding (spec §8)."""

    severity: Severity
    code: str
    message: str
    where: str


@runtime_checkable
class Lowerable(Protocol):
    """The single open, engine-owned seam. Implemented engine-side; ``nwx``
    never implements it. The engine type is opaque to ``nwx`` (hence
    ``object``)."""

    def lower(self, engine: object) -> object: ...
