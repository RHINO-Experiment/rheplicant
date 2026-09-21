"""The drift model: what a chain's state does between epochs.

A linear-Gaussian transition and the Ornstein-Uhlenbeck process that is the
usual way to build one, plus the hyper-transition that lets a chain's own
parameters drift. Nothing here integrates anything -- it says what one step
is, and the recursion beside it says what a sequence of them comes to.
"""

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from rheplicant.core.errors import StateValidationError


class LinearGaussianTransition(eqx.Module):
    """``zeta_{e+1} = phi zeta_e + w``, ``w ~ N(0, diag(process_std)^2)``.

    Attributes:
        phi: ``(n, n)``. A full matrix, because a multi-component drift can
            rotate; the process and initial spreads are diagonal because a
            correlated *innovation* is a modelling claim nobody has made and a
            silently-accepted full covariance would need its own Cholesky
            refusal.
        process_std: ``(n,)``, strictly positive.
        initial_std: ``(n,)``, strictly positive -- ``sd(zeta_1)``.
        initial_mean: ``(n,)``. Zero unless declared.
        hyper: empty for a fixed transition. Present so that
            :class:`~rheplicant.inference.factorize.Factorization` can ask one
            question of either type.

    **Positivity is checked here and nowhere else, on purpose.** The rows this
    class contributes are what constrain every ``zeta_e``, so a strictly
    positive spread makes each marginalisation's block full-rank *by
    construction* -- and that is what lets the filter call
    :func:`~rheplicant.inference.sqrtinfo.marginalise_arrays` inside a
    ``lax.scan`` instead of the checked
    :func:`~rheplicant.inference.sqrtinfo.marginalise`, which concretises and
    therefore cannot be traced or differentiated. One eager check at
    declaration, not one traced check per epoch of a thousand.

    A traced spread is **not** checked, and cannot be: a
    :class:`HyperTransition` builds these blocks from theta, and under NUTS
    theta goes wherever it likes. Parameterise the builder so that positivity is
    structural -- return ``jnp.exp(log_sigma)``, never a raw sampled scale --
    which is why this class takes standard deviations rather than a covariance.
    """

    phi: jax.Array
    process_std: jax.Array
    initial_std: jax.Array
    initial_mean: jax.Array
    hyper: tuple[str, ...] = eqx.field(static=True, default=())

    def __init__(
        self,
        phi: Any,
        process_std: Any,
        initial_std: Any,
        initial_mean: Any = None,
        hyper: Sequence[str] = (),
    ):
        self.process_std = jnp.atleast_1d(jnp.asarray(process_std))
        self.initial_std = jnp.atleast_1d(jnp.asarray(initial_std))
        self.phi = jnp.atleast_2d(jnp.asarray(phi))
        self.initial_mean = (
            jnp.zeros_like(self.initial_std)
            if initial_mean is None
            else jnp.broadcast_to(jnp.atleast_1d(jnp.asarray(initial_mean)), self.initial_std.shape)
        )
        self.hyper = tuple(hyper)

    def __check_init__(self):
        width = int(self.process_std.shape[0])
        if self.phi.shape != (width, width):
            raise StateValidationError(
                f"phi is {self.phi.shape} but process_std has {width} "
                "component(s), so the transition maps the chain into a space of a "
                "different size. Broadcasting one into the other would build a "
                "chain nobody declared."
            )
        if self.initial_std.shape != (width,):
            raise StateValidationError(
                f"initial_std is {self.initial_std.shape} but process_std is "
                f"{self.process_std.shape}; they describe the same chain."
            )
        for name, spread in (
            ("process_std", self.process_std),
            ("initial_std", self.initial_std),
        ):
            # A traced spread cannot be judged here and must not be pretended
            # about -- see the class docstring.
            if isinstance(spread, jax.core.Tracer):
                continue
            # `not (... > 0)` rather than `... <= 0`, because every comparison
            # against NaN is False and the second form waves a NaN spread
            # through. `isfinite` is separate rather than folded in for the same
            # reason: `inf > 0` is True, and an infinite spread makes
            # `1 / process_std` zero, which is a transition row of zeros -- the
            # density comes back -inf from the log-determinant, a thousand epochs
            # after the declaration that caused it.
            if not bool(jnp.all(jnp.isfinite(spread) & (spread > 0.0))):
                raise StateValidationError(
                    f"{name} must be finite and strictly positive; got {spread}. "
                    "These rows are what constrain zeta at each marginalisation, so "
                    "a zero leaves the Gaussian integral over that epoch divergent "
                    "-- and inside a lax.scan finite arithmetic returns a large "
                    "plausible number for it rather than an infinity anyone would "
                    "notice. A chain that genuinely does not move is "
                    "process_std=1e-9, not 0.0; one that is effectively unlinked is "
                    "1e12, not inf, which would zero the transition rows and send "
                    "the whole campaign's density to -inf; and a quantity that is "
                    'constant across the campaign is scope="global".'
                )

    @property
    def width(self) -> int:
        """How many components the chain carries."""
        return int(self.process_std.shape[0])

    def at(self, values: dict[str, jax.Array]) -> "LinearGaussianTransition":
        """Itself. A fixed transition does not depend on theta -- that is the claim."""
        return self


def ornstein_uhlenbeck(
    tau: Any, sigma: Any, width: int = 1, hyper: Sequence[str] = ()
) -> LinearGaussianTransition:
    """A stationary OU chain: correlation time ``tau`` in epochs, spread ``sigma``.

    ``phi = exp(-1/tau)`` and ``process_std = sigma sqrt(1 - phi^2)``, so
    ``var(zeta_{e+1}) = phi^2 var + Q`` returns ``sigma^2`` when it starts there
    -- stationarity is arithmetic here, not an assumption, and
    ``tests/evidence/test_transition.py`` pins it.

    A **function**, not a class: section 11's sketch writes
    ``OrnsteinUhlenbeck(...)`` as though it were a type, but the type the filter
    consumes -- and the type a :class:`HyperTransition` builder must return --
    is :class:`LinearGaussianTransition`. An OU is a way of constructing one, and
    this package spells constructors in lower case.
    """
    decay = jnp.exp(-1.0 / jnp.asarray(tau))
    return LinearGaussianTransition(
        phi=decay * jnp.eye(width),
        process_std=jnp.broadcast_to(jnp.asarray(sigma) * jnp.sqrt(1.0 - decay**2), (width,)),
        initial_std=jnp.broadcast_to(jnp.asarray(sigma), (width,)),
        hyper=hyper,
    )


class HyperTransition(eqx.Module):
    """A transition whose blocks are functions of theta -- section 6's ``linked_hyper``.

    Attributes:
        build: ``{global latent: value} -> LinearGaussianTransition``. Static: it
            is code. Called **inside** the theta likelihood, so it must be
            traceable, and it must return blocks that are positive for every
            theta the sampler can reach -- ``exp`` of a sampled log-scale, not a
            sampled scale.
        hyper: which global latents ``build`` reads. Declared rather than
            inferred, because
            :class:`~rheplicant.inference.factorize.Factorization` checks them
            and a closure's free variables are not inspectable.
        width: how many components the chain carries. Declared for the same
            reason the shape of anything else is: the filter needs it before it
            has a value to look at.
    """

    build: Callable[[dict[str, jax.Array]], LinearGaussianTransition] = eqx.field(static=True)
    hyper: tuple[str, ...] = eqx.field(static=True)
    width: int = eqx.field(static=True)

    def at(self, values: dict[str, jax.Array]) -> LinearGaussianTransition:
        """Resolve the blocks at these latent values."""
        missing = [name for name in self.hyper if name not in values]
        if missing:
            raise StateValidationError(
                f"This transition is built from {list(self.hyper)}; no value was "
                f"given for {missing}. A linked_hyper chain is evaluated inside the "
                "theta likelihood precisely so those values are current -- "
                "resolving it against a default would be the compression-time "
                "pinning the sub-scope exists to prevent."
            )
        resolved = self.build({name: values[name] for name in self.hyper})
        if resolved.width != self.width:
            raise StateValidationError(
                f"This transition declares width {self.width} but its builder "
                f"returned {resolved.width}. The stored per-epoch blocks were "
                "shaped against the declared number and cannot be re-cut now."
            )
        return resolved
