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
from bayesmith.errors import StructureError as _FarStructureError
from bayesmith.optimize import check_loss_sense as _far_check_loss_sense
from bayesmith.optimize import minimize as _far_minimize

from rheplicant.core.errors import ParameterSpaceError, StateValidationError
from rheplicant.inference.likelihood import (
    check_observed_shape,
    mean_squared_error,
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


#: The far side's remedy names its OWN routes, which do not exist here. The
#: wrapper below swaps this sentence for the near-side one; see
#: :func:`_refuse_a_score_the_optimizer_would_walk_away_from` for why the
#: substitution is asserted rather than trusted.
_FAR_REMEDY = (
    "Pass `lambda p, o: -score(p, o)`, or use a density-aware route "
    "(`fit`, `nuts`)."
)
_NEAR_REMEDY = (
    "Pass `lambda p, o: -likelihood(p, o)`, or use a likelihood-aware route "
    "(numpyro_bridge, SamplingPlan)."
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

    The detection is
    :func:`bayesmith.optimize.check_loss_sense`'s as of the Wave C
    ``calibrate`` switch — migration ledger **D11**, whose wording is "loss
    方向守卫随迁, ``test_loss_sense`` 经包装重放". This function is that
    wrapper. It is two tests over there, and both are worth naming because the
    second is the one that matters: a **declared** ``sense`` attribute, which
    is exact but only sees objects that carry one and so is a whitelist; and a
    **measured** comparison at the perfect prediction, ``loss_fn(observed,
    observed)``, which an error minimises and a log-density maximises. The
    second holds for any callable, declaration or not.

    **What the wrapper is for.** Two things do not survive the seam.

    The exception CLASS: the far side raises ``StructureError`` and this
    package's callers are pinned on ``ParameterSpaceError``. Iron law 1 keeps
    the class, so it is translated here.

    The REMEDY sentence: the far side ends with *"use a density-aware route
    (`fit`, `nuts`)"*, and neither name exists in rheplicant. Advice that
    points a user at a function their package does not have is worse than no
    advice, so :data:`_FAR_REMEDY` is swapped for :data:`_NEAR_REMEDY`.

    **The swap is asserted, not trusted.** A substitution keyed on the far
    side's exact wording stops applying the moment that wording changes, and
    it stops applying *silently* — the message would simply go back to naming
    `fit` and `nuts`. ``test_loss_sense.py`` asserts the near-side remedy is
    present, so a reworded upstream goes red here rather than shipping wrong
    advice.

    Raises:
        ParameterSpaceError: if the sense is wrong by either test, or if the
            score at the starting parameters is not finite. The last is not
            scope creep: a non-finite score makes the comparison unable to
            judge, and NaN compares False against everything, so treating it
            as "cannot tell, proceed" would let the case this guard exists for
            through whenever it arrives with a NaN attached.
    """
    try:
        _far_check_loss_sense(loss_fn, forward(params0), observed)
    except _FarStructureError as exc:
        raise ParameterSpaceError(str(exc).replace(_FAR_REMEDY, _NEAR_REMEDY)) from None


class GradientCalibrator(eqx.Module):
    """Fixed-step gradient descent on a forward model (minimal demonstrator).

    Attributes:
        learning_rate: step size (static configuration).
        n_steps: number of gradient steps (static configuration).
    """

    learning_rate: float = eqx.field(static=True, default=1e-2)
    n_steps: int = eqx.field(static=True, default=100)

    def __check_init__(self):
        if not self.learning_rate > 0:  # `not >` so a NaN rate is refused too
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
        far = _far_minimize(
            lambda params: loss_fn(forward(params), observed),
            params0,
            method="gradient",
            steps=self.n_steps,
            learning_rate=self.learning_rate,
        )
        return far.values, far.history


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
        if not self.learning_rate > 0:  # `not >` so a NaN rate is refused too
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
        far = _far_minimize(
            lambda params: loss_fn(forward(params), observed),
            params0,
            method="adam",
            steps=self.n_steps,
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            eps=self.eps,
        )
        return far.values, far.history
