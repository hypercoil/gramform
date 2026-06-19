# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
``nwx`` reference-backend capability ledger (spec §7)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The **single source of truth** for which IR features the *reference* engine
(``nitrix``, statistical-modelling-suite **v3**) lowers onto a shipped kernel.
``nwx`` is engine-agnostic, but its backend-awareness diagnostics
(:mod:`gramform.grammars.nwx.validate`), the directive parser, and the
engine-contract dry-run all cite this one ledger -- so the "what ships" facts
live in exactly one place and cannot drift apart (they previously did -- the
same sets were duplicated across five sites).

Pure: declares facts (enum / string sets) about the backend; imports no array
library -- the ``jax`` / ``nitrix`` firewall holds.

**Status (nitrix stats-suite v3, merged 2026-06).** v3 ships ``nwx``'s entire
v1 scope -- GLM / GAM / GAMM / (G)LMM + residualisation + the multi-level node
graph -- so nearly every IR feature now has a kernel:

- **families** -- gaussian/binomial/poisson (v1) + gamma/negbinomial/tweedie
  (v3 §4 ``_FAMILIES``) + beta (``beta_fit``): **all** of nwx's families;
- **random-effect covariance** -- scalar (``reml_fit`` R1),
  diagonal/unstructured (``lme_fit`` R2), nested/crossed
  (``lme_fit(inner=/cross=)`` R3/R4): **all** of nwx's structures;
- **smooth bases** -- ps/cc/tp/te (v1) + cr/gp/mrf (§3.2) + re/fs (§2/§3.1):
  **all** of nwx's bases;
- **error structures** -- correlation ar1/car1/cs (§1.4), varIdent/varPower
  (§1.4), se=robust/cluster (``sandwich_cov`` §6.2), dof=satterthwaite/kr
  (§1.3): all shipped;
- **residualisation** -- aggressive + nonaggressive (``partial_residualise``
  §5.1);
- **inference** -- permutation (Freedman-Lane), TFCE, cluster, FDR-BH,
  Bonferroni.

The residual that nwx can *express* but v3 does **not** ship is small, and is
the only thing the backend-awareness layer still flags (the ``UNSHIPPED`` /
narrowed-``SHIPPED`` sets below): non-canonical links, the ``soft``
residualise mode, RFT inference, and a non-Gaussian random *slope* (the GLMM
scalar-RE-only limit).
"""

from gramform.grammars.nwx.spec import (
    BasisKind,
    Family,
    Link,
    Mode,
    Structure,
)

# --- fully-shipped axes (every nwx-surfaced value has a kernel) -------------
#: All nwx families ship (v1 gaussian/binomial/poisson + v3 §4 gamma/
#: negbinomial/tweedie + beta_fit).
SHIPPED_FAMILIES = frozenset(Family)
#: All nwx smooth bases ship (v1 ps/cc/tp/te + v3 §3.2 cr/gp/mrf + §2/§3.1
#: re/fs).
SHIPPED_BASES = frozenset(BasisKind)
#: All nwx random-effect covariance structures ship (scalar reml_fit R1;
#: diagonal/unstructured lme_fit R2; nested/crossed lme_fit R3/R4).
SHIPPED_STRUCTURES = frozenset(Structure)

# --- axes with a residual gap (what the awareness layer still flags) --------
#: Built-in links: each nitrix family carries only its **canonical** link;
#: probit / inverse / sqrt need a hand-built ``Family``, so they are not
#: shipped built-ins.
SHIPPED_LINKS = frozenset({Link.IDENTITY, Link.LOG, Link.LOGIT})
#: Residualise modes: aggressive + nonaggressive (ICA-AROMA,
#: ``partial_residualise`` §5.1) ship; ``SOFT`` (§5.2) does not.
SHIPPED_RESIDUALISE_MODES = frozenset({Mode.AGGRESSIVE, Mode.NONAGGRESSIVE})
#: Parametric inference corrections that do **not** ship (permutation / TFCE /
#: cluster / FDR-BH / Bonferroni all do; random-field theory does not).
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
    """A parametric correction is shipped unless it is in the unshipped set
    (``None`` / permutation paths are handled by the caller)."""
    return correction not in UNSHIPPED_INFERENCE_CORRECTIONS


def glmm_random_slope_unshipped(structure: Structure, family: Family) -> bool:
    """nitrix ``glmm_fit`` ships a **scalar** random effect only, so a
    non-scalar random effect (a random *slope*: diagonal / unstructured) under
    a **non-Gaussian** family is the GLMM-random-slope Tier-2 deferral. A
    Gaussian random slope is served by ``lme_fit`` R2 and *is* shipped."""
    return structure is not Structure.SCALAR and family is not Family.GAUSSIAN
