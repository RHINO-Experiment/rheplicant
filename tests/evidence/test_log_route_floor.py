"""The log route against ``RadiometerNoise.floor``, measured on both sides (A5-3).

``RadiometerNoise`` declares ``sigma = f max(|mu|, floor)``. The log route
(:func:`~rheplicant.inference.loglinear.to_log_space`) replaces that with ``f``
on every sample, which is the declared likelihood only where the floor never
binds. Before the refusal, a ``log_conjugate`` estimate was bit-identical for
every floor, and ``auto_blocks`` routed a floored model to it.

The reference is a grid posterior of the DECLARED likelihood, log-determinant
included, in numpy float64: one latent, ``a = log gain``, prediction
``mu_j = exp(a) S_j`` with ``S`` spanning a decade so a floor can bind on part
of it. This is why the file lives in the x64 session: at ``f = 1e-4`` the
posterior sd of ``a`` is 1.25e-5, and float32 summation error alone exceeds
1e-3 of that.

``f = 1e-4`` rather than a real channel's 4.05e-3, because the first-order log
approximation has its own error, ``O(f)`` in units of the posterior sd --
measured on this fixture, 7.4e-3 sd at ``f = 4.05e-3`` and 1.8e-4 sd at
``f = 1e-4``. At the smaller ``f`` the comparison isolates the floor.

Measured before the refusal, log route minus grid mean in sd units, and the
relative error of the log route's sd:

==============  ==========  =========
floor           mean shift  sd rel
==============  ==========  =========
0               +1.8e-4     1.0e-8
1e-6 min(mu)    +1.8e-4     1.0e-8
min(mu)(1-d)    +1.8e-4     1.0e-8
min(mu)(1+d)    +1.9e-4     1.6e-5
max(mu)         -1.3e-1     5.3e-1
10 max(mu)      -1.3e-2     9.5e-1
==============  ==========  =========

The refusal is conservative for the first rows: a floor below every prediction
does not change the likelihood at the truth, but whether it binds depends on
the parameter values a solve visits, so the predicate reads the declaration
and refuses any ``floor > 0``.
"""

from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from rheplicant import Coordinates, State
from rheplicant.core.errors import ParameterSpaceError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.pipeline import Pipeline
from rheplicant.inference import (
    Bind,
    Block,
    Latent,
    ParameterSpace,
    SamplingPlan,
    auto_blocks,
)
from rheplicant.inference.engines import LOG_CONJUGATE
from rheplicant.inference.loglinear import log_route_refusal, to_log_space
from rheplicant.inference.noise import RadiometerNoise
from rheplicant.radio import GainOperator

N = 64
SKY = np.logspace(2.0, 3.0, N)
TRUE_A = float(np.log(1.5))
PRIOR_STD = 10.0
#: f = 1 / sqrt(1e6 * 100) = 1e-4.
CHANNEL_WIDTH, INTEGRATION_TIME = 1e6, 100.0
FRACTIONAL = 1.0 / np.sqrt(CHANNEL_WIDTH * INTEGRATION_TIME)
MU = np.exp(TRUE_A) * SKY
DELTA = 1e-3
#: Agreement demanded of a route that is taken, in units of the posterior sd.
TOLERANCE = 1e-3

FLOORS = {
    "zero": 0.0,
    "1e-6*min": 1e-6 * MU.min(),
    "min*(1-d)": MU.min() * (1 - DELTA),
    "min": MU.min(),
    "min*(1+d)": MU.min() * (1 + DELTA),
    "max": MU.max(),
    "10*max": 10 * MU.max(),
}


class Row(AbstractOperator):
    """``data[t, f] = row[f]`` for every ``t``."""

    requires: ClassVar[tuple[str, ...]] = ("coords.time", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data",)

    row: jax.Array

    def __call__(self, state):
        n_time = state.coords.time.shape[0]
        return state.with_data(jnp.broadcast_to(self.row, (n_time, self.row.shape[0])))


def _state():
    return State(
        coords=Coordinates(time=jnp.arange(1.0), freq=jnp.linspace(60e6, 85e6, N)),
        meta={"telescope": "RHINO", "obs_id": "a5-3"},
    )


def _pipeline():
    return Pipeline(
        Row(row=jnp.asarray(SKY)), GainOperator(gain=jnp.ones(1)), names=("sky", "gain")
    )


def _space():
    return ParameterSpace(
        latents=[
            Latent(
                "a",
                init=jnp.asarray([TRUE_A + 1e-3]),
                prior=dist.Normal(jnp.zeros(1), PRIOR_STD * jnp.ones(1)),
            )
        ],
        bindings=[Bind("a", into=lambda p: p["gain"].gain, fn=jnp.exp)],
    )


def _observed():
    rng = np.random.default_rng(0)
    return MU * (1.0 + FRACTIONAL * rng.standard_normal(N))


def _grid_posterior(d, floor):
    """Mean, sd and MAP of ``a`` under the declared likelihood, by grid."""

    def nll(a):
        mu = np.exp(a)[:, None] * SKY[None, :]
        sigma = FRACTIONAL * np.maximum(np.abs(mu), floor)
        chi = (d[None, :] - mu) / sigma
        return (0.5 * chi**2 + np.log(sigma)).sum(axis=1) + 0.5 * (a / PRIOR_STD) ** 2

    centre, width = float(np.log(np.mean(d / SKY))), 10 * FRACTIONAL / np.sqrt(N)
    for _ in range(3):  # recentre twice so the grid resolves the posterior
        grid = np.linspace(centre - 30 * width, centre + 30 * width, 20001)
        value = nll(grid)
        weight = np.exp(-(value - value.min()))
        weight /= weight.sum()
        mean = float(np.sum(weight * grid))
        sd = float(np.sqrt(np.sum(weight * (grid - mean) ** 2)))
        best = float(grid[np.argmin(value)])
        centre, width = mean, sd
    return mean, sd, best


def _noise(floor):
    return RadiometerNoise(
        channel_width=CHANNEL_WIDTH, integration_time=INTEGRATION_TIME, floor=float(floor)
    )


def _estimate(block, floor, **kwargs):
    return SamplingPlan(_space(), block).estimate(
        _pipeline(),
        _state(),
        jnp.asarray(_observed())[None, :],
        noise=_noise(floor),
        check_identifiability=False,
        **kwargs,
    )


def test_the_fixture_is_float64():
    assert jnp.asarray(_observed()).dtype == jnp.float64


@pytest.mark.parametrize("cell", list(FLOORS))
def test_the_log_route_is_refused_or_matches_the_declared_likelihood(cell):
    """Every floor cell: refused at partition AND at solve, or taken and right.

    ``zero`` is the only cell taken, and it is the negative control: it must
    stay accepted and agree with the grid to ``TOLERANCE`` sd in mean and sd.
    """
    floor = FLOORS[cell]
    blocks = auto_blocks(_space(), _pipeline(), _state(), noise=_noise(floor))
    if floor > 0.0:
        assert log_route_refusal(_noise(floor)) == "noise_neither"
        assert blocks[-1].engine is None  # the gradient block
        with pytest.raises(ParameterSpaceError, match="floor"):
            _estimate(Block("a", engine=LOG_CONJUGATE), floor)
        return

    assert blocks[-1].engine == LOG_CONJUGATE
    mean, sd, _ = _grid_posterior(_observed(), floor)
    estimate = float(_estimate(Block("a", engine=LOG_CONJUGATE), floor).values["a"][0])
    _, sigma = to_log_space(jnp.asarray(_observed()), _noise(floor))
    log_sd = 1.0 / np.sqrt(np.sum(1.0 / np.asarray(sigma) ** 2) + 1.0 / PRIOR_STD**2)
    assert abs(estimate - mean) < TOLERANCE * sd
    assert abs(log_sd - sd) < TOLERANCE * sd


@pytest.mark.parametrize("cell", ["max", "10*max"])
def test_where_the_floor_binds_the_old_route_was_wrong(cell):
    """The refusal is needed, not only conservative.

    The first-order log route in closed form -- ``y = log d + f^2 / 2``,
    precision ``n / f^2`` -- is what ``to_log_space`` fed the solver before the
    refusal. Where the floor binds it misses the declared posterior by far more
    than the tolerance, in mean or in sd.
    """
    d = _observed()
    mean, sd, _ = _grid_posterior(d, FLOORS[cell])
    precision = N / FRACTIONAL**2 + 1.0 / PRIOR_STD**2
    log_mean = np.sum(np.log(d / SKY) + 0.5 * FRACTIONAL**2) / FRACTIONAL**2 / precision
    log_sd = 1.0 / np.sqrt(precision)
    shift = max(abs(log_mean - mean) / sd, abs(log_sd - sd) / sd)
    assert shift > 100 * TOLERANCE


def test_both_engines_at_a_floor_equal_to_the_smallest_prediction():
    """The threshold itself, with the dispatcher bypassed on both sides.

    At ``floor = min(mu)`` the floor binds on one sample as soon as ``a`` drops
    below the truth. The log engine asked for by name refuses; the gradient
    engine, which evaluates the declared likelihood, reaches the grid's MAP.
    """
    floor = FLOORS["min"]
    with pytest.raises(ParameterSpaceError, match="floor"):
        _estimate(Block("a", engine=LOG_CONJUGATE), floor)

    _, sd, best = _grid_posterior(_observed(), floor)
    gradient = _estimate(Block("a", steps=2000), floor, max_iter=50)
    assert abs(float(gradient.values["a"][0]) - best) < TOLERANCE * sd
