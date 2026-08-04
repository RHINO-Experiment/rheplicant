"""BasisTemperatureOperator — a smooth effective T_sys, parameterized by coefficients.

Sits on the graph's reserved ``t_sys_extra`` node, which the canonical
single-antenna template designates for a generic effective-T_sys contribution:
one of the ``t_ant_sum`` leaves, alongside ``ground_pickup`` and the
beam-averaged ``atmosphere``. Like those it is an *effective* temperature by
D13's construction — already carrying whatever beam weighting its author
intended — which is why it enters after ``beam_spill`` rather than through it.

**It is parameterized by coefficients, not cells, and that is the whole point.**
Measured on the assembled graph with a known 5000 K CW tone and a gain free per
time sample (``tests/radio/test_t_sys_basis.py``, at a generic coefficient
point, ``n_time=7``, ``n_freq=5``)::

    free-per-cell T_ant,  tone ON  (5000 K)   n_par=42 rank=35 nullity=7
    free-per-cell T_ant,  tone OFF            n_par=42 rank=35 nullity=7
    (3,2)-basis T_ant,    tone ON  (5000 K)   n_par=13 rank=13 nullity=0
    (3,2)-basis T_ant,    tone OFF            n_par=13 rank=12 nullity=1

Against a free antenna temperature per (time, frequency) cell the tone buys
**exactly nothing** — the nullity is ``n_time`` with it and without it, because
the free cells absorb the whole of ``g[t] x (tone profile)`` sample by sample,
the tone's own channels included.
RHINO's central design choice (paper Sect. 4) therefore only pays once
``T_ant`` is frequency-smooth, and this operator is where that smoothness is
made structural: there is no way to write a free-per-cell fit through it except
by handing it a basis that is *complete* on both axes — any square invertible
pair, of which the identity is the obvious one but complete Legendre matrices
do it just as well, as the free-per-cell row of the table below is built.

The user has confirmed RHINO's antenna temperature IS frequency-smooth, so the
route is physically sound. Which axis matters is measured rather than assumed,
and it is frequency: a basis complete in frequency makes the tone worth nothing
whatever the time axis does — the nullity is the same with the tone on and off,
whatever that nullity happens to be on the grid in hand (1 on the square fixture
below, 7 on the 7x5 one) — while a basis complete in time is still rescued by
it. What is invariant is that the tone changes nothing, not the number. See
:mod:`rheplicant.core.basis` for the full sweep.

**The gain and these coefficients cannot share a linear block.** Each is affine
given the other and their product is not affine in the pair, so
:func:`~rheplicant.inference.linear.check_linearity` refuses a group holding
both — correctly. They are two conditionally-linear blocks of one
:class:`~rheplicant.inference.plan.SamplingPlan`::

    plan = SamplingPlan(space, Block("gain"), Block("t_coeff"))
    est   = plan.estimate(twin, state, observed, noise=sigma)
    draws = plan.sample(twin, state, observed, noise=sigma, key=key, n_sweeps=300)

**A sibling contribution is survivable, and only because the ids are stable.**
``t_sys_extra`` is ``many=True``: a second contribution makes the bare
``p["t_sys_extra"]`` ambiguous rather than silently resolving to the
``SumOperator`` over both, and the per-instance ids ``t_sys_extra_1`` /
``t_sys_extra_2`` are what a binding should name once there is more than one.
Read them off ``assembly.instances``.

**Two routes to the same expansion, and the difference is where smoothness
lives.** This operator holds the coefficients, so the *model* is smooth and a
per-cell fit is not expressible through it. The other route is
:class:`~rheplicant.core.basis.SeparableBasis`'s ``expand`` as a
``Bind`` function, which leaves the operator holding a full ``(n_time,
n_freq)`` leaf and makes the smoothness a property of the
*parameterization*::

    Bind("t_coeff", into=lambda p: p["noise_wave"].t_unc, fn=basis.expand)

That is the route the noise-wave temperatures need, since those leaves belong
to :class:`~rheplicant.radio.instrument.noise_wave.NoiseWaveOperator` and are
full-grid by its contract. Both drive the conjugate exits identically; neither
is a wrapper for the other.
"""

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp

from rheplicant.core.basis import SeparableBasis
from rheplicant.core.errors import StateValidationError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State


class BasisTemperatureOperator(AbstractOperator):
    """An effective temperature expanded on a separable (time, frequency) basis.

    Produces ``time_basis @ coeff @ freq_basis.T`` — always the full
    ``(n_time, n_freq)`` grid, which is the one temperature shape
    :mod:`~rheplicant.radio.instrument.noise_wave` can never misread.

    The two design matrices are ordinary array leaves, the same convention
    :class:`~rheplicant.radio.instrument.noise_wave.NoiseWaveOperator` uses for
    its ``Gamma`` spectra: a known quantity the operator carries, not a
    parameter. What is *inferred* is ``coeff``, and a ParameterSpace should bind
    into that leaf and no other — binding into a design matrix would infer the
    basis, which is a modelling choice and not a measurement.

    Attributes:
        coeff: ``(n_k, n_j)`` coefficients [K] — the differentiable target.
        time_basis: ``(n_time, n_k)`` design matrix on the time axis.
        freq_basis: ``(n_freq, n_j)`` design matrix on the frequency axis.
    """

    requires: ClassVar[tuple[str, ...]] = ("coords.time", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data",)
    graph_node: ClassVar[str] = "t_sys_extra"

    coeff: jax.Array = eqx.field(converter=jnp.asarray)
    time_basis: jax.Array = eqx.field(converter=jnp.asarray)
    freq_basis: jax.Array = eqx.field(converter=jnp.asarray)

    def __check_init__(self):
        # Building the handle IS the design-matrix check: SeparableBasis refuses
        # a matrix that is not 2-D, is empty, or has more functions than
        # samples. Re-run here rather than trusted, because the operator holds
        # its matrices as its own leaves and nothing obliges a caller to have
        # gone through `from_basis`.
        basis = self.basis
        if self.coeff.ndim != 2:
            raise StateValidationError(
                f"coeff has shape {tuple(self.coeff.shape)}: a separable expansion's "
                f"coefficient is 2-D, (n_k, n_j), never {self.coeff.ndim}-D. A 1-D "
                "coefficient would be read by the matrix product as a single ROW and "
                "the expansion would come back with the wrong number of dimensions, "
                "which the sum at t_ant_sum would then broadcast."
            )
        expected = basis.coeff_shape
        if tuple(self.coeff.shape) != expected:
            raise StateValidationError(
                f"coeff has shape {tuple(self.coeff.shape)}, but these design matrices "
                f"take {expected} — {expected[0]} time functions by {expected[1]} "
                "frequency ones. The first axis of the coefficient indexes the TIME "
                "basis; when n_k == n_j the transpose is shape-legal and would return "
                "the transpose of the field you meant, so this check only helps while "
                "the two counts differ."
            )

    @classmethod
    def from_basis(cls, basis: SeparableBasis, coeff) -> "BasisTemperatureOperator":
        """Build from a :class:`~rheplicant.core.basis.SeparableBasis`.

        The ordinary way in, because it is the way that cannot get the two
        design matrices the wrong way round: the basis already knows which is
        which, and ``basis.fit(field)`` turns a temperature you can write down
        into the coefficients that reproduce it.
        """
        return cls(coeff=coeff, time_basis=basis.time, freq_basis=basis.freq)

    @property
    def basis(self) -> SeparableBasis:
        """The two design matrices as one handle — ``expand``, ``fit``, shapes."""
        return SeparableBasis(time=self.time_basis, freq=self.freq_basis)

    def __call__(self, state: State) -> State:
        if state.coords is None or state.coords.time is None or state.coords.freq is None:
            raise StateValidationError(
                "BasisTemperatureOperator requires state.coords with time and freq "
                "axes: the design matrices are built for one specific grid, and "
                "without the axes there is nothing to check them against — the "
                "expansion would still evaluate and would silently describe some "
                "other observation."
            )
        n_time = state.coords.time.shape[0]
        n_freq = state.coords.freq.shape[0]
        if self.time_basis.shape[0] != n_time:
            raise StateValidationError(
                f"time_basis covers {self.time_basis.shape[0]} time samples but "
                f"coords.time has {n_time}. This is also where a swapped pair of "
                "design matrices lands, whenever the two grids are different "
                "lengths; when they are the same length the product is legal and the "
                "expansion comes back transposed."
            )
        if self.freq_basis.shape[0] != n_freq:
            raise StateValidationError(
                f"freq_basis covers {self.freq_basis.shape[0]} channels but "
                f"coords.freq has {n_freq}. A basis built for another band would "
                "otherwise be evaluated on this one and return a smooth, plausible, "
                "wrong temperature."
            )
        return state.with_data(self.time_basis @ self.coeff @ self.freq_basis.T)


__all__ = ["BasisTemperatureOperator"]
