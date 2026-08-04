"""Calibration: infer pipeline parameters from observed data.

Deliberately OUTSIDE the forward model — a calibrator consumes the
``forward(params)`` function built by :func:`~rheplicant.inference.forward.build_forward_fn`
and never reaches into operators. :class:`GradientCalibrator` is a minimal
working demonstration (fixed-step gradient descent, pure JAX); Bayesian
inference goes through :mod:`rheplicant.inference.numpyro_bridge`, uncertainty
forecasts through :mod:`rheplicant.inference.uncertainty` — all via the same seam.
"""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax

from rheplicant.core.errors import ParameterSpaceError, StateValidationError
from rheplicant.inference.likelihood import (
    MAXIMIZE,
    check_observed_shape,
    mean_squared_error,
    sense_of,
)


def _refuse_mis_shaped_observed(
    forward: Callable[[Any], jax.Array], params0: Any, observed: jax.Array
) -> None:
    """Entry guard shared by both calibrators.

    ``jax.eval_shape`` asks the forward model what it predicts without
    evaluating it — one abstract trace, the same instrument
    :meth:`~rheplicant.inference.parameters.ParameterSpace.validate` uses, and
    negligible against a fit. It belongs HERE and not inside ``step``: the
    optimizers run their loop under ``lax.scan``, so a check in the loss would
    be re-traced with every recompilation and, worse, would only refuse after
    the calibrator had already been asked to run.
    """
    prediction = jax.eval_shape(forward, params0)
    check_observed_shape(
        jax.numpy.shape(prediction), observed, predictor="this forward model"
    )


def _refuse_a_score_the_optimizer_would_walk_away_from(
    loss_fn: Callable[[jax.Array, jax.Array], jax.Array],
    forward: Callable[[Any], jax.Array],
    params0: Any,
    observed: jax.Array,
) -> None:
    """Refuse a scoring function whose optimum lies the wrong way.

    Both calibrators MINIMIZE. A log-density has the same signature as an
    error and must be MAXIMIZED, so passing one type-checks, runs, and
    descends a function unbounded below while the loss history — the only
    evidence a user has — looks like textbook convergence. Measured on a
    one-parameter gain fit with truth ``g = 1.0``::

        mean_squared_error       ->  g = +0.9999    loss  2499  ->  0.002617
        GaussianLikelihood(0.05) ->  g = -30.7349   loss -3.2e7 -> -1.3e11

    The check is in two parts, and the second is the one that matters.

    **Declared.** :func:`~rheplicant.inference.likelihood.sense_of` reads a
    ``sense`` attribute. Cheap, exact, and gives the best message — but it
    only sees objects that declare it, so on its own it is a whitelist, and a
    whitelist is wrong about precisely the code it has not met.

    **Measured.** A scoring function is evaluated where it is unambiguous: at
    the *perfect* prediction, ``loss_fn(observed, observed)``. An error attains
    its minimum there; a log-density attains its maximum. So if scoring the
    perfect prediction returns a value ABOVE the score at the starting
    parameters, the function increases toward the truth and a minimizer will
    run away from it. That holds for any callable, including a user's own
    likelihood class and a bare lambda, with no declaration at all.

    Costs one extra evaluation of ``forward`` and two of ``loss_fn`` on
    concrete arrays, once, at entry — negligible against a fit, and outside
    the ``lax.scan`` the optimizers run.

    Raises:
        ParameterSpaceError: if the sense is wrong by either test, or if the
            score at the starting parameters is not finite. The last is not
            scope creep: a non-finite score makes the comparison unable to
            judge, and NaN compares False against everything, so treating it
            as "cannot tell, proceed" would let the case this guard exists for
            through whenever it arrives with a NaN attached.
    """
    if sense_of(loss_fn) == MAXIMIZE:
        raise ParameterSpaceError(
            f"{type(loss_fn).__name__} declares sense={MAXIMIZE!r}: it is a "
            "log-density, and this calibrator minimizes. Minimizing a log-density "
            "walks away from the truth while reporting an improving loss. Pass "
            "`lambda p, o: -likelihood(p, o)`, or use a likelihood-aware route "
            "(numpyro_bridge, SamplingPlan)."
        )

    at_start = jax.numpy.asarray(loss_fn(forward(params0), observed))
    at_truth = jax.numpy.asarray(loss_fn(observed, observed))
    if not jax.numpy.isfinite(at_start) or not jax.numpy.isfinite(at_truth):
        raise ParameterSpaceError(
            f"the loss is not finite at entry (start={at_start}, perfect-fit="
            f"{at_truth}). A fit cannot begin from here, and the sense of the "
            "scoring function cannot be established either — a non-finite score "
            "compares False against everything."
        )
    if at_truth > at_start:
        raise ParameterSpaceError(
            f"{getattr(loss_fn, '__name__', type(loss_fn).__name__)} scores a "
            f"PERFECT prediction ({at_truth}) higher than the starting one "
            f"({at_start}), so it increases toward the truth and must be "
            "maximized — but this calibrator minimizes, and will walk away from "
            "the answer while the loss history improves. Negate it: "
            "`lambda p, o: -score(p, o)`."
        )


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

        Raises:
            ParameterSpaceError: if ``observed`` is not shaped exactly like
                ``forward(params0)``, or if ``loss_fn`` is a log-density rather
                than an error. Both minimize something other than what was
                asked and report a small, converged loss for it.
        """
        _refuse_mis_shaped_observed(forward, params0, observed)
        _refuse_a_score_the_optimizer_would_walk_away_from(
            loss_fn, forward, params0, observed
        )

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
    surrogate stages (:class:`~rheplicant.radio.surrogate.NeuralOperator`) and
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

        Raises:
            ParameterSpaceError: if ``observed`` is not shaped exactly like
                ``forward(params0)``, or if ``loss_fn`` is a log-density rather
                than an error — see :meth:`GradientCalibrator.fit`.
        """
        _refuse_mis_shaped_observed(forward, params0, observed)
        _refuse_a_score_the_optimizer_would_walk_away_from(
            loss_fn, forward, params0, observed
        )

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
