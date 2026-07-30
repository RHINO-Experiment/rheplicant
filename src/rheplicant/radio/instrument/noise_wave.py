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

That is not a convenience. Each switch position contributes exactly one equation
per frequency channel, so with per-channel noise-wave temperatures the design
matrix has rank ``min(n_src, 3) * n_freq``. One load leaves it deficient by a
factor of three; three distinct loads make it square. Sharing a single ``Gamma``
across the cycle collapses every source onto the same row and gives that up
entirely — the fit then returns a finite, correctly-shaped, wholly prior-driven
answer. (Frequency structure in ``Gamma`` *does* identify **scalar** noise-wave
temperatures from a single load; it is the per-channel case, which is the
physical one, that needs the switch.)

``Gamma`` is stored as two real leaves rather than one complex leaf because
:func:`~rheplicant.inference.uncertainty.fisher_information` runs
``jax.jacfwd``, which refuses complex parameters.
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


class NoiseWaveOperator(AbstractOperator):
    """Apply reflection couplings and add the noise-wave temperatures.

    Attributes:
        t_unc: uncorrelated noise-wave temperature [K]; scalar or ``(n_freq,)``.
        t_cos: in-phase noise-wave temperature [K].
        t_sin: quadrature noise-wave temperature [K].
        t_rx: receiver offset temperature [K] (the draft's ``T_rx``; the numpy
            reference calls the same quantity ``t_0``).
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
