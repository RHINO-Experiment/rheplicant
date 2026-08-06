"""A campaign of repeated nights, clean and with a deterministic common mode.

Not a test file -- imported by the diagnostics' test modules.

**The design is the SAME every night.** That is not a simplification, it is the
case section 1 describes: same sky at the same LST, same receiver, same thermal
environment. It is also the case that separates the diagnostics. With a design
that varies per epoch, part of a common-mode error lands out of span *differently
each night*, which leaks into the held-out residual and makes every diagnostic
look like it works. With a repeated design the coherent error splits cleanly:

* the **in-span** half biases theta identically in every epoch, leaves no
  residual anywhere, and is detectable by NO data-driven diagnostic;
* the **out-of-span** half shows in the per-epoch chi-square and in the named
  template projection, at sqrt(N) with correct scatter.

``COMMON`` is chosen with both halves nonzero -- measured ``||c_par|| = 2.30704``
and ``||c_perp|| = 2.00937`` in whitened units, against a whole of ``3.05942`` --
because a fixture with only one of them would make half of this plan's tests
vacuous and the other half impossible. ``whitened_split()`` recomputes both, so
the numbers in this sentence are checked rather than remembered.
"""

import jax.numpy as jnp
import numpy as np

from rheplicant.inference.compress import compress_linear
from rheplicant.inference.factorize import Factorization
from rheplicant.inference.parameters import Bind, Latent, ParameterSpace

N_THETA, N_SAMPLES, SIGMA = 2, 8, 0.5
TRUTH = np.array([1.0, -0.5])
NAMES = ("x",)
PRIOR_STD = 2.0

DESIGN = np.random.default_rng(3).normal(size=(N_SAMPLES, N_THETA))
#: One calibration solution, applied to every night. Deterministic: it
#: contributes no variance at all, which is why chi-square per epoch is right,
#: split-half agrees, leave-one-out agrees, and the answer is wrong.
COMMON = np.array([0.9, -0.45, 0.6, 0.3, -0.75, 0.15, 0.45, -0.3])
#: The named systematic template a per-epoch summary projects onto. Here it is
#: the common mode itself, which is the best case;
#: ``test_a_template_that_misses_the_mode_stays_quiet`` covers the case where
#: the analyst guessed wrong.
TEMPLATE = COMMON / np.linalg.norm(COMMON)


def prior_fisher():
    return np.eye(N_THETA) / PRIOR_STD**2


class _Normal:
    """Minimal duck-typed prior: only ``log_prob`` is read downstream.

    Its curvature is what :func:`rheplicant.inference.diagnostics.systematic_floor`
    differentiates for the prior's share of the posterior width, so this must be
    the same Gaussian ``prior_fisher()`` writes down by hand -- ``1 / scale^2``
    per component. The two are pinned against each other in
    ``test_systematic_floor.py``, because a fixture whose two priors disagreed
    would make every crossing epoch in this file wrong by a constant and
    nothing would say so.
    """

    def __init__(self, loc, scale):
        self.loc, self.scale = loc, scale

    def log_prob(self, x):
        return -0.5 * (
            ((x - self.loc) / self.scale) ** 2 + jnp.log(2 * jnp.pi * self.scale**2)
        )


def space():
    """One global latent ``x`` of shape ``(2,)``, with the prior above."""
    latent = Latent("x", init=jnp.zeros(N_THETA), prior=_Normal(0.0, PRIOR_STD))
    return ParameterSpace(
        latents=(latent,),
        bindings=(Bind("x", into=lambda p: p.x),),
    )


def factorization(represents=None):
    """The campaign's declaration, optionally modelling a shared input product.

    ``represents={"beam_map": ("x",)}`` is section 9.5's escape hatch in
    executable form: it says the beam map is *not* an unmodelled shared error
    because the campaign carries a global latent for it. Two epochs may then
    share its hash, and the systematic floor no longer binds on ``x``.
    """
    return Factorization(space(), represents=dict(represents or {}))


def terms(n_epochs, biased, seed=11, templates=None):
    """``n_epochs`` compressed nights, with or without the common mode.

    ``templates=None`` means the fixture's own ``gain_ripple``; pass a mapping
    to name a different one. ``{}`` is a campaign with no template at all, which
    is what pins that the chi-square half needs no guess.
    """
    rng = np.random.default_rng(seed)
    out = []
    for e in range(n_epochs):
        data = (
            DESIGN @ TRUTH
            + SIGMA * rng.normal(size=N_SAMPLES)
            + (COMMON if biased else 0.0)
        )
        out.append(
            compress_linear(
                design={"x": jnp.asarray(DESIGN)},
                observed=jnp.asarray(data),
                noise_std=SIGMA,
                shapes={"x": (N_THETA,)},
                epoch_id=f"n{e}",
                templates={"gain_ripple": jnp.asarray(TEMPLATE)}
                if templates is None
                else templates,
            )
        )
    return tuple(out)


def posterior(collected):
    """``(mean, covariance)`` of theta from a bag of these terms plus the prior."""
    fisher = sum(np.asarray(t.info.fisher()) for t in collected) + prior_fisher()
    b = sum(np.asarray(t.info.factor).T @ np.asarray(t.info.target) for t in collected)
    covariance = np.linalg.inv(fisher)
    return covariance @ b, covariance


def whitened_projector():
    """``P`` onto the whitened design's column space -- what a night can absorb."""
    whitened = DESIGN / SIGMA
    return whitened @ np.linalg.pinv(whitened)


def whitened_split():
    """``(||c_par||, ||c_perp||)`` of the common mode, in whitened units.

    Recomputed rather than quoted so the module docstring's two numbers are a
    measurement. The parallel half is what biases theta and is invisible to
    every data-driven diagnostic; the perpendicular half is the only part any of
    them can see.
    """
    whitened = COMMON / SIGMA
    parallel = whitened_projector() @ whitened
    return float(np.linalg.norm(parallel)), float(
        np.linalg.norm(whitened - parallel)
    )
