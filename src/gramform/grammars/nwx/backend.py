# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
``nwx`` reference-backend capability ledger (spec §7)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The **single source of truth** for which IR features the *reference* engine
(``nitrix``, statistical-modelling-suite) lowers onto a shipped kernel.
``nwx`` is engine-agnostic, but its backend-awareness diagnostics
(:mod:`gramform.grammars.nwx.validate`), the directive parser, and the
engine-contract dry-run all cite this one ledger -- so the "what ships" facts
live in exactly one place and cannot drift apart (they previously did -- the
same sets were duplicated across five sites).

Pure: declares facts (enum / string sets) about the backend; imports no array
library -- the ``jax`` / ``nitrix`` firewall holds.

**Status (nitrix stats-suite GP branch ``feat/stats-gp``, prospective).** This
branch supersedes v3: on top of v3's GLM / GAM / GAMM / (G)LMM +
residualisation + multi-level node graph, it closes the three v3 residuals --
non-canonical links, non-Gaussian random *slopes*, and ``soft`` residualise --
so **every IR axis nwx can express now lowers onto a shipped kernel**:

- **families** -- gaussian/binomial/poisson (v1) + gamma/negbinomial/tweedie
  (§4 ``_FAMILIES``) + beta (``beta_fit``): **all** of nwx's families;
- **links** -- canonical identity/log/logit + non-canonical probit/inverse/sqrt
  (real ``Link`` impls composed via ``Family.with_link``; IRLS consumes
  ``mu_eta``): **all** of nwx's links;
- **random-effect covariance** -- scalar (``reml_fit`` R1),
  diagonal/unstructured (``lme_fit`` R2), nested/crossed
  (``lme_fit(inner=/cross=)`` R3/R4) for Gaussian; the same structures under a
  non-Gaussian family via ``glmm_fit`` random slopes (``z=`` / ``structure=``,
  PQL / Laplace / AGQ): **all** of nwx's structures, every family;
- **smooth bases** -- ps/cc/tp/te (v1) + cr/gp/mrf (§3.2) + re/fs (§2/§3.1):
  **all** of nwx's bases (the GP wave makes ``gp`` HSGP-reduced-rank + ARD);
- **error structures** -- correlation ar1/car1/cs (§1.4), varIdent/varPower
  (§1.4), se=robust/cluster (``sandwich_cov`` §6.2), dof=satterthwaite/kr
  (§1.3): all shipped;
- **residualisation** -- aggressive (``residualise``) + nonaggressive +
  ``soft`` (``partial_residualise`` ridge ``l2=`` / James-Stein shrinkage,
  FR §5.2);
- **inference** -- permutation (Freedman-Lane), TFCE, cluster, FDR-BH,
  Bonferroni.

The **sole** residual the backend-awareness layer still flags is **RFT
inference** -- and that is *intentional* (random-field theory is deliberately
omitted for its known failure modes, not pending), so it is a permanent
exclusion rather than a "not yet". See ``UNSHIPPED_INFERENCE_CORRECTIONS``.
"""

from gramform.grammars.nwx.spec import (
    BasisKind,
    Family,
    Link,
    Mode,
    Structure,
)

# --- fully-shipped axes (every nwx-surfaced value has a kernel) -------------
#: All nwx families ship (v1 gaussian/binomial/poisson + §4 gamma/
#: negbinomial/tweedie + beta_fit).
SHIPPED_FAMILIES = frozenset(Family)
#: All nwx links ship: canonical identity/log/logit + non-canonical
#: probit/inverse/sqrt, the latter as real ``Link`` impls composed onto a
#: family via ``Family.with_link`` (IRLS consumes ``mu_eta``).
SHIPPED_LINKS = frozenset(Link)
#: All nwx smooth bases ship (v1 ps/cc/tp/te + §3.2 cr/gp/mrf + §2/§3.1 re/fs).
SHIPPED_BASES = frozenset(BasisKind)
#: All nwx random-effect covariance structures ship -- Gaussian via
#: reml_fit (R1) / lme_fit (R2-R4); non-Gaussian via ``glmm_fit`` random
#: slopes (PQL / Laplace / AGQ).
SHIPPED_STRUCTURES = frozenset(Structure)
#: All nwx residualise modes ship: aggressive + nonaggressive (ICA-AROMA,
#: ``partial_residualise`` §5.1) + ``soft`` (the ridge / James-Stein shrunk
#: ``partial_residualise``, FR §5.2).
SHIPPED_RESIDUALISE_MODES = frozenset(Mode)

# --- the one intentional exclusion -----------------------------------------
#: Inference corrections deliberately **not** shipped. Permutation / TFCE /
#: cluster / FDR-BH / Bonferroni all ship; random-field theory (``rft``) is
#: *intentionally* omitted for its known failure modes -- a permanent
#: exclusion, not a deferral.
UNSHIPPED_INFERENCE_CORRECTIONS = frozenset({'rft'})


def family_shipped(family: Family) -> bool:
    return family in SHIPPED_FAMILIES


def link_shipped(link: Link) -> bool:
    return link in SHIPPED_LINKS


def basis_shipped(basis: BasisKind) -> bool:
    return basis in SHIPPED_BASES


def structure_shipped(structure: Structure) -> bool:
    return structure in SHIPPED_STRUCTURES


def residualise_mode_shipped(mode: Mode) -> bool:
    return mode in SHIPPED_RESIDUALISE_MODES


def inference_correction_shipped(correction: str | None) -> bool:
    """An inference correction is shipped unless it is in the intentionally
    unshipped set (``rft``). ``None`` / permutation paths are handled by the
    caller."""
    return correction not in UNSHIPPED_INFERENCE_CORRECTIONS
