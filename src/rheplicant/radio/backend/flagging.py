"""FlaggingOperator — PLACEHOLDER RFI flagging.

Element: "We then apply various flagging, averaging, and calibration steps
that can correct for some of these contributions, but can also introduce
their own additional issues, e.g. if the models are slightly wrong/biased."

Real physics to come: MomentRFI-based flagging (as in the noise-wave GCR
draft, where flags inform the noise covariance). Flagging is a *data
processing* operator living in the same pipeline formalism — which is exactly
how "processing steps introduce their own issues" becomes modellable. The
placeholder thresholds the data and stores a boolean mask in ``state.aux``
(the traced side-channel), leaving the data itself untouched.
"""

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp

from rheplicant.core.errors import StateValidationError
from rheplicant.core.frozen import FrozenMapping
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State


class FlaggingOperator(AbstractOperator):
    """Store a threshold-based flag mask in ``state.aux["flags"]`` (placeholder).

    ``True`` marks a flagged (bad) sample. Data is not modified; downstream
    operators (averaging, likelihoods) decide how to use the mask.

    Attributes:
        threshold: flag samples with ``data > threshold`` (static
            configuration; thresholding is not differentiable anyway).
    """

    requires: ClassVar[tuple[str, ...]] = ("data",)
    provides: ClassVar[tuple[str, ...]] = ("aux.flags",)
    graph_node: ClassVar[str] = "flagging"

    threshold: float = eqx.field(static=True)

    def __call__(self, state: State) -> State:
        flags = state.data > self.threshold
        return state.replace(aux={**state.aux, "flags": flags})


class MomentRFIFlaggingOperator(AbstractOperator):
    """RFI flagging via MomentRFI's ``IterativeSurfaceFitter`` (host callback).

    The real flagger behind the placeholder above. Flagging is inherently
    non-differentiable (a boolean decision), so ``jax.pure_callback`` into the
    numpy MomentRFI package is the *permanent* right integration — not a
    stopgap. Jit-compatible; not vmappable/differentiable (by nature).

    Behaviour:
        * ``state.data`` must be a positive, linear-scale ``(n_time, n_freq)``
          waterfall (MomentRFI works on log10 internally).
        * Existing ``aux["flags"]`` are passed as MomentRFI's ``prior_mask``
          and included in the output — flaggers compose instead of clobbering.
        * The result is written back to ``aux["flags"]`` (True = flagged);
          consumed by ``MaskedGaussianLikelihood`` (noise covariance, per the
          GCR draft) and by ``SkySpaceFilter`` weighting.

    Requires the optional ``MomentRFI`` package (install it into the same
    environment); raises a helpful ImportError otherwise.

    Attributes:
        config: ``IterativeSurfaceFitter`` keyword arguments (static, hashable;
            e.g. ``{"sigma_threshold": 4.0, "degree_freq": 10}``).
            ``verbose`` is forced off.
        kernel_shapes: broad-round box-kernel shapes, e.g. ``((3, 3), (1, 9))``
            (static; empty runs round 0 only).
    """

    requires: ClassVar[tuple[str, ...]] = ("data",)
    provides: ClassVar[tuple[str, ...]] = ("aux.flags",)
    graph_node: ClassVar[str] = "flagging"

    config: FrozenMapping = eqx.field(
        static=True, converter=FrozenMapping, default_factory=FrozenMapping
    )
    kernel_shapes: tuple[tuple[int, int], ...] = eqx.field(static=True, default=())

    def __call__(self, state: State) -> State:
        if state.data.ndim != 2:
            raise StateValidationError(
                f"MomentRFI expects a 2D (n_time, n_freq) waterfall, got ndim={state.data.ndim}."
            )
        prior = state.aux.get("flags")
        if prior is None:
            prior = jnp.zeros(state.data.shape, dtype=bool)
        out_spec = jax.ShapeDtypeStruct(state.data.shape, jnp.bool_)
        flags = jax.pure_callback(self._host_fit, out_spec, state.data, prior)
        return state.replace(aux={**state.aux, "flags": flags})

    def _host_fit(self, waterfall, prior_mask):  # numpy land
        import numpy as np

        try:
            from MomentRFI import IterativeSurfaceFitter
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "MomentRFIFlaggingOperator needs the MomentRFI package installed "
                "in this environment (pip install -e <path-to-MomentRFI>). "
                "The threshold-based FlaggingOperator works without it."
            ) from exc

        fitter = IterativeSurfaceFitter(verbose=False, **dict(self.config))
        kernels = [np.ones(shape) for shape in self.kernel_shapes] or None
        mask = fitter.fit(
            np.asarray(waterfall), kernels=kernels, prior_mask=np.asarray(prior_mask)
        )
        return np.asarray(mask, dtype=bool)
