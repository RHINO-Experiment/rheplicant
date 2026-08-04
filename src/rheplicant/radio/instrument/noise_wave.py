"""NoiseWaveOperator — the receiver stage of the noise-wave data model.

Implements the bracket of the Noise-Wave GCR draft's Eq. 1::

    T_sys = T_src c_s + T_unc k_unc + T_cos k_cos + T_sin k_sin + T_rx

with the coupling spectra ``(c_s, k_unc, k_cos, k_sin)`` supplied by
``rhino_cal_jax`` (draft Eqs. 2-6). The physics lives in that package, where it
is cross-checked against the numpy reference it was ported from; this module is
the adapter that gives it a State -> State face and a home on the signal graph.

Placement. This operator sits at the ``noise_wave`` node, downstream of the
``receiver_input`` selector, so ``state.data`` already carries the *selected*
source's ``T_src``. What the selector discards is which source that was — and
that is precisely what the couplings depend on. The operator therefore carries
``Gamma`` per source and re-reads the same switch array the selector used,
``coords.extra["receiver_input"]``.

That is not a convenience, and the counting it supports is the number a real
experiment picks its switching cadence from, so state it exactly.

**The per-channel rank rule.** Each switch position contributes exactly one
equation per frequency channel, so *while every temperature is free per
channel* the design matrix has rank ``min(n_src, k) * n_freq``, where ``k`` is
the number of free temperature **families**. ``k`` is four — ``T_unc, T_cos,
T_sin, T_rx`` — whenever ``T_rx`` is fitted, and three only when ``T_rx`` is
taken as known: ``t_rx`` is a leaf of this operator like the other three, and
its coupling is 1 rather than absent. So a four-family per-channel fit needs
**four** distinct loads to be square, and three loads leave it deficient by
exactly ``n_freq``. Sharing a single ``Gamma`` across the cycle collapses every
source onto the same row and drops the rank to ``n_freq`` whatever ``n_src``
is — the fit then returns a finite, correctly-shaped, wholly prior-driven
answer.

Those are measurements, not arguments:
``tests/radio/test_noise_wave.py::TestPerChannelRankRule`` sweeps ``n_src`` 1-5,
``k`` in {3, 4} and ``n_freq`` in {3, 5, 7} through
:func:`~rheplicant.inference.identifiability.identifiability`.

**The rule is per-channel and nothing more.** The moment the temperatures
become coefficients of a frequency basis — which is what a smooth-spectrum
parameterization does, and what the next tranche makes ordinary — the basis
ties channels together, the per-channel counting stops applying in *both*
directions, and no counting rule replaces it. What survives is a bound,
``rank <= min(n_src * n_freq, k * n_basis)``, and two measured facts about how
loosely it binds (``TestBasisRegimeBreaksTheRule``):

* per-channel counting **understates**. Two loads and a 3-coefficient basis
  identify all ``k * n_basis = 12`` coefficients at ``k = 4``, where
  ``min(n_src, k) * n_basis`` would have said 6.
* the bound **overstates**. A single load whose ``Gamma`` is itself linear in
  frequency gives rank 5 against a bound of 7, because a basis function times a
  low-order coupling is another low-order function and the products are not
  independent. Which loads do that is not visible from ``n_src``.

The scalar case is the ``n_basis = 1`` corner of the same statement: frequency
structure in ``Gamma`` identifies all ``k`` **scalar** temperatures from a
single load, which is why a scalar demonstration says nothing about switching.

So: read a switching cadence off ``min(n_src, k) * n_freq`` for a per-channel
fit, and *measure* every other parameterization with
:func:`~rheplicant.inference.identifiability.identifiability` — the instrument
every number above came from.

``Gamma`` is stored as two real leaves rather than one complex leaf because
:func:`~rheplicant.inference.uncertainty.fisher_information` runs
``jax.jacfwd``, which refuses complex parameters.

The source temperature. ``T_src`` is whatever the selected source delivers, so
for the antenna branch it is the beam-convolved sky --
:class:`~rheplicant.radio.sky.source.SkySourceOperator` upstream feeds it
directly, and ``examples/sky_to_noise_wave.py`` runs that end to end. Two things
about that junction are the caller's responsibility, because neither is a shape:

* the sky projector must return a *temperature*. ``DriftScanProjector`` and
  ``GeneralPointingProjector`` default to ``normalize_beam=False`` (numpy
  limTOD's convention), which returns ``int(B T)`` rather than
  ``int(B T)/int(B)``. Use ``normalize_beam=True`` when the output is destined
  for ``T_src``; a beam the caller normalized by hand is still biased at the
  percent level, since the band-limit truncates the denominator too.
* ``gamma_src``'s row order must match the selector's branch order (the graph's
  in-edge declaration: antenna, then ``cal_loads``). Both are
  ``(n_source, n_freq)``, so a transposition is shape-legal and costs tens of
  kelvin. Read the order off the assembled twin --
  ``assembly["receiver_input"].names`` -- rather than assuming it.

A third join used to be the caller's problem and no longer is. ``SwitchCycle``
range-checks the switch array against its source count, but that check needs
concrete values and is skipped under tracing; JAX's gather semantics would then
clamp the coupling lookup to a neighbouring source while ``SelectOperator``
selected no branch at all. ``SwitchCycle.gather`` now fills out-of-range samples
with NaN, so the two consumers of one switch array cannot disagree in silence.
"""

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp

from rheplicant.core.errors import StateValidationError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State


def _rhino_cal_jax():
    """The rhino_cal_jax module, with an actionable message when it is absent."""
    try:
        import rhino_cal_jax
    except ImportError as exc:  # pragma: no cover - exercised by the import guard
        raise ImportError(
            "NoiseWaveOperator needs the rhino_cal_jax package: install it with "
            "pip install 'rhino-cal-jax @ "
            "git+https://github.com/RHINO-Experiment/rhino-cal.git'"
        ) from exc
    return rhino_cal_jax


#: Every temperature leaf takes one of these, and the phrasing is shared by all
#: four error messages so a caller sees the same list wherever they trip.
_TEMPERATURE_SHAPES = (
    "() scalar, (n_freq,) per channel, (n_time, 1) per time sample, or "
    "(n_time, n_freq) per cell"
)


def _check_temperature(name: str, value: jax.Array, n_freq: int) -> None:
    """Reject a temperature leaf whose shape cannot mean what the caller meant.

    ``rhino_cal_jax.system_temperature`` broadcasts each temperature against a
    ``(n_time, n_freq)`` coupling, so the *legal* shapes are exactly the ones
    that broadcast there: scalar, ``(n_freq,)``, ``(n_time, 1)`` and
    ``(n_time, n_freq)`` (with 1 allowed on either axis). Without this check the
    illegal ones still fail — but as a raw ``ValueError`` from ``jnp.multiply``
    at *call* time, naming neither the leaf nor the convention.

    Only ``n_freq`` is checkable here: ``n_time`` arrives with the state, long
    after ``__check_init__``. That is enough for the case worth catching,
    because the dangerous shape is a bare per-*time* vector and this rejects
    every one of those whose length is not ``n_freq``.

    The residual is irreducible and is stated in the message rather than papered
    over: when ``n_time == n_freq`` a per-time vector and a per-frequency
    spectrum have the identical shape, nothing can tell them apart, and the
    wrong one broadcasts along the wrong axis to a finite, correctly-shaped,
    wrong ``T_sys``. That is upstream's reason for the ``(n_time, 1)`` column
    convention, and it is why this message names the convention every time.
    """
    if value.ndim > 2:
        raise StateValidationError(
            f"{name} has shape {tuple(value.shape)}: a noise-wave temperature is "
            f"0-, 1- or 2-D, never {value.ndim}-D. Legal shapes are "
            f"{_TEMPERATURE_SHAPES}."
        )
    if value.ndim == 1 and value.shape[0] != n_freq:
        raise StateValidationError(
            f"{name} has shape {tuple(value.shape)}, but a 1-D temperature is "
            f"always read as per-FREQUENCY and this operator has n_freq={n_freq}. "
            f"Legal shapes are {_TEMPERATURE_SHAPES}. If you meant one value per "
            f"time sample, pass an explicit (n_time, 1) column: the bare vector "
            f"is not a shape this package can distinguish from a spectrum, and "
            f"when n_time == n_freq it would broadcast along frequency and return "
            f"a finite, correctly-shaped, wrong T_sys."
        )
    if value.ndim == 2 and value.shape[1] not in (1, n_freq):
        raise StateValidationError(
            f"{name} has shape {tuple(value.shape)}: the trailing axis of a 2-D "
            f"temperature is the frequency axis, so it must be n_freq={n_freq} "
            f"(or 1, for a per-time column). Legal shapes are "
            f"{_TEMPERATURE_SHAPES}. A transposed (n_freq, n_time) array lands "
            f"here rather than broadcasting silently."
        )


class NoiseWaveOperator(AbstractOperator):
    """Apply reflection couplings and add the noise-wave temperatures.

    All four temperatures broadcast against the ``(n_time, n_freq)`` couplings,
    so each independently takes ``() scalar``, ``(n_freq,)``, ``(n_time, 1)`` or
    ``(n_time, n_freq)``. **A bare 1-D array is always read as per-frequency**
    — that is ``rhino_cal_jax.system_temperature``'s convention, and this
    operator inherits it. To vary a temperature with time, pass an explicit
    ``(n_time, 1)`` column; a bare ``(n_time,)`` vector is refused at
    construction unless ``n_time == n_freq``, where no check can tell it from a
    spectrum.

    Attributes:
        t_unc: uncorrelated noise-wave temperature [K].
        t_cos: in-phase noise-wave temperature [K].
        t_sin: quadrature noise-wave temperature [K].
        t_rx: receiver offset temperature [K] (the draft's ``T_rx``; the numpy
            reference calls the same quantity ``t_0``). Free like the other
            three — see the module docstring for what that costs in loads.
        gamma_src_re: ``(n_source, n_freq)`` real part of each source's ``Gamma``.
        gamma_src_im: ``(n_source, n_freq)`` imaginary part.
        gamma_rec_re: ``(n_freq,)`` real part of the receiver's ``Gamma``.
        gamma_rec_im: ``(n_freq,)`` imaginary part.
        switch_key: key in ``coords.extra`` holding the per-sample source index.
    """

    requires: ClassVar[tuple[str, ...]] = ("data", "coords.extra")
    provides: ClassVar[tuple[str, ...]] = ("data",)
    graph_node: ClassVar[str] = "noise_wave"

    t_unc: jax.Array = eqx.field(converter=jnp.asarray)
    t_cos: jax.Array = eqx.field(converter=jnp.asarray)
    t_sin: jax.Array = eqx.field(converter=jnp.asarray)
    t_rx: jax.Array = eqx.field(converter=jnp.asarray)
    gamma_src_re: jax.Array = eqx.field(converter=jnp.asarray)
    gamma_src_im: jax.Array = eqx.field(converter=jnp.asarray)
    gamma_rec_re: jax.Array = eqx.field(converter=jnp.asarray)
    gamma_rec_im: jax.Array = eqx.field(converter=jnp.asarray)
    switch_key: str = eqx.field(static=True, default="receiver_input")

    def __check_init__(self):
        if self.gamma_src_re.ndim != 2:
            raise StateValidationError(
                f"gamma_src_re must be 2D (n_source, n_freq), got "
                f"ndim={self.gamma_src_re.ndim}."
            )
        if self.gamma_src_re.shape != self.gamma_src_im.shape:
            raise StateValidationError(
                f"gamma_src real/imaginary parts disagree: {self.gamma_src_re.shape} "
                f"vs {self.gamma_src_im.shape}."
            )
        if self.gamma_rec_re.shape != self.gamma_rec_im.shape:
            raise StateValidationError(
                f"gamma_rec real/imaginary parts disagree: {self.gamma_rec_re.shape} "
                f"vs {self.gamma_rec_im.shape}."
            )
        if self.gamma_rec_re.ndim != 1:
            raise StateValidationError(
                f"gamma_rec_re must be 1D (n_freq,), got ndim={self.gamma_rec_re.ndim}."
            )
        if self.gamma_src_re.shape[1] != self.gamma_rec_re.shape[0]:
            raise StateValidationError(
                f"gamma_src has n_freq={self.gamma_src_re.shape[1]} but gamma_rec "
                f"has n_freq={self.gamma_rec_re.shape[0]}."
            )
        # After the gamma checks, because they are what establish n_freq.
        n_freq = int(self.gamma_rec_re.shape[0])
        _check_temperature("t_unc", self.t_unc, n_freq)
        _check_temperature("t_cos", self.t_cos, n_freq)
        _check_temperature("t_sin", self.t_sin, n_freq)
        _check_temperature("t_rx", self.t_rx, n_freq)

    @property
    def n_source(self) -> int:
        """Number of switchable sources this operator carries a ``Gamma`` for."""
        return int(self.gamma_src_re.shape[0])

    def _source_index(self, state: State) -> jax.Array:
        """The per-sample source index, or all-zeros when there is one source."""
        n_time = state.data.shape[0]
        extra = {} if state.coords is None else state.coords.extra
        if self.switch_key not in extra:
            if self.n_source == 1:
                return jnp.zeros((n_time,), dtype=int)
            raise StateValidationError(
                f"NoiseWaveOperator carries {self.n_source} sources but "
                f"coords.extra[{self.switch_key!r}] is absent, so there is no way "
                "to tell which one is connected. Defaulting to the first would "
                "return a finite, correctly-shaped, wrong answer."
            )
        index = jnp.asarray(extra[self.switch_key])
        if index.ndim != 1 or index.shape[0] != n_time:
            raise StateValidationError(
                f"coords.extra[{self.switch_key!r}] must be (n_time,) = ({n_time},), "
                f"got shape {index.shape}."
            )
        return index.astype(int)

    def __call__(self, state: State) -> State:
        rcj = _rhino_cal_jax()
        if state.data is None or jnp.asarray(state.data).ndim != 2:
            got = None if state.data is None else jnp.asarray(state.data).shape
            raise StateValidationError(
                f"NoiseWaveOperator expects (n_time, n_freq) data; got {got}."
            )
        if state.data.shape[1] != self.gamma_rec_re.shape[0]:
            raise StateValidationError(
                f"data has n_freq={state.data.shape[1]} but gamma_rec has "
                f"n_freq={self.gamma_rec_re.shape[0]}."
            )

        cycle = rcj.SwitchCycle(
            source_index=self._source_index(state),
            labels=tuple(str(i) for i in range(self.n_source)),
        )
        coup = rcj.Couplings.from_stacked(
            cycle.gather(
                rcj.couplings(
                    self.gamma_src_re + 1j * self.gamma_src_im,
                    self.gamma_rec_re + 1j * self.gamma_rec_im,
                ).stacked
            )
        )
        return state.with_data(
            rcj.system_temperature(
                coup, t_src=state.data, t_unc=self.t_unc,
                t_cos=self.t_cos, t_sin=self.t_sin, t_rx=self.t_rx,
            )
        )
