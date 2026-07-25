"""Calibration: infer pipeline parameters from observed data.

Deliberately OUTSIDE the forward model — a calibrator consumes the
``forward(params)`` function built by :func:`~replicant.inference.forward.build_forward_fn`
and never reaches into operators. :class:`GradientCalibrator` is a minimal
working demonstration (fixed-step gradient descent, pure JAX); Bayesian
inference goes through :mod:`replicant.inference.numpyro_bridge`, uncertainty
forecasts through :mod:`replicant.inference.uncertainty` — all via the same seam.
"""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax

from replicant.core.errors import StateValidationError
from replicant.inference.likelihood import mean_squared_error


class GradientCalibrator(eqx.Module):
    """Fixed-step gradient descent on a forward model (minimal demonstrator).

    Attributes:
        learning_rate: step size (static configuration).
        n_steps: number of gradient steps (static configuration).
    """

    learning_rate: float = eqx.field(static=True, default=1e-2)
    n_steps: int = eqx.field(static=True, default=100)

    def __check_init__(self):
        if self.learning_rate <= 0:
            raise StateValidationError(f"learning_rate must be > 0, got {self.learning_rate}.")
        if not isinstance(self.n_steps, int) or self.n_steps < 1:
            raise StateValidationError(f"n_steps must be a positive int, got {self.n_steps!r}.")

    def fit(
        self,
        forward: Callable[[Any], jax.Array],
        params0: Any,
        observed: jax.Array,
        loss_fn: Callable[[jax.Array, jax.Array], jax.Array] = mean_squared_error,
    ) -> tuple[Any, jax.Array]:
        """Minimize ``loss_fn(forward(params), observed)`` from ``params0``.

        Returns:
            ``(params_fit, losses)``: the fitted parameter pytree and the
            per-step loss history, shape ``(n_steps,)``.
        """

        def loss(params: Any) -> jax.Array:
            return loss_fn(forward(params), observed)

        def step(params: Any, _: None) -> tuple[Any, jax.Array]:
            value, grads = jax.value_and_grad(loss)(params)
            params = jax.tree.map(lambda p, g: p - self.learning_rate * g, params, grads)
            return params, value

        params_fit, losses = jax.lax.scan(step, params0, None, length=self.n_steps)
        return params_fit, losses


class AdamCalibrator(eqx.Module):
    """Adam optimizer on a forward model (pure JAX — no optax dependency).

    Adaptive per-parameter step sizes make this the right tool where
    fixed-step gradient descent stalls or diverges — notably neural
    surrogate stages (:class:`~replicant.radio.surrogate.NeuralOperator`) and
    other poorly-conditioned parameter sets. Same interface as
    :class:`GradientCalibrator`.

    Attributes:
        learning_rate: Adam step size (static).
        n_steps: number of steps (static).
        beta1: first-moment decay (static).
        beta2: second-moment decay (static).
        eps: numerical floor (static).
    """

    learning_rate: float = eqx.field(static=True, default=1e-2)
    n_steps: int = eqx.field(static=True, default=1000)
    beta1: float = eqx.field(static=True, default=0.9)
    beta2: float = eqx.field(static=True, default=0.999)
    eps: float = eqx.field(static=True, default=1e-8)

    def __check_init__(self):
        if self.learning_rate <= 0:
            raise StateValidationError(f"learning_rate must be > 0, got {self.learning_rate}.")
        if not isinstance(self.n_steps, int) or self.n_steps < 1:
            raise StateValidationError(f"n_steps must be a positive int, got {self.n_steps!r}.")
        if not (0.0 <= self.beta1 < 1.0 and 0.0 <= self.beta2 < 1.0):
            raise StateValidationError(
                f"beta1/beta2 must be in [0, 1), got {self.beta1}, {self.beta2}."
            )

    def fit(
        self,
        forward: Callable[[Any], jax.Array],
        params0: Any,
        observed: jax.Array,
        loss_fn: Callable[[jax.Array, jax.Array], jax.Array] = mean_squared_error,
    ) -> tuple[Any, jax.Array]:
        """Minimize ``loss_fn(forward(params), observed)`` from ``params0``.

        Returns:
            ``(params_fit, losses)``: fitted parameters and per-step loss
            history, shape ``(n_steps,)``.
        """

        def loss(params: Any) -> jax.Array:
            return loss_fn(forward(params), observed)

        zeros = jax.tree.map(jax.numpy.zeros_like, params0)

        def step(carry: Any, index: jax.Array) -> tuple[Any, jax.Array]:
            params, m, v = carry
            value, grads = jax.value_and_grad(loss)(params)
            m = jax.tree.map(lambda a, g: self.beta1 * a + (1 - self.beta1) * g, m, grads)
            v = jax.tree.map(
                lambda a, g: self.beta2 * a + (1 - self.beta2) * g**2, v, grads
            )
            t = index + 1
            params = jax.tree.map(
                lambda p, mm, vv: p
                - self.learning_rate
                * (mm / (1 - self.beta1**t))
                / (jax.numpy.sqrt(vv / (1 - self.beta2**t)) + self.eps),
                params, m, v,
            )
            return (params, m, v), value

        (params_fit, _, _), losses = jax.lax.scan(
            step, (params0, zeros, zeros), jax.numpy.arange(self.n_steps)
        )
        return params_fit, losses
