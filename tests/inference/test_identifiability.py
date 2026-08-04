"""Tests for cross-block identifiability — the rank test that sees ACROSS blocks.

The motivating failure is an alternating solve over ``gain x T_ant``. Every
per-block guard this package ships passes at every sweep: ``check_linearity``
because each conditional genuinely is affine, the CG residual because it is
computed from the block, ``condition_estimate`` because it returns kappa of the
block's own normal operator. The joint model is nevertheless degenerate, and
the solve lands thousands of kelvin from the truth -- how many is the initial
offset carried along the null direction, measured across four decades in
``test_degenerate_partition.py``, which is also where the guards are shown
reading alike on the best and worst runs.

A block-local diagnostic cannot see that, by construction. A rank test over the
JOINT Jacobian can, and these tests pin that it does — including the part that
matters most, that it reports a DIFFERENT verdict for a good parameterization
than for a bad one.

Verified under BOTH the package's default precision and ``JAX_ENABLE_X64=1``:
the whole point of the diagnostic is a number at 1e-17, so a suite that only
held in one precision would not be testing the claim.
"""

import dataclasses
import os
import subprocess
import sys
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant import Coordinates, State
from rheplicant.core.errors import ParameterSpaceError, StateValidationError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.pipeline import Pipeline
from rheplicant.inference import Bind, Latent, ParameterSpace
from rheplicant.inference.identifiability import (
    DEFAULT_RANK_RTOL,
    identifiability,
)
from rheplicant.radio import GainOperator

N_TIME, N_FREQ = 8, 8
TONE_CHANNEL, TONE_KELVIN = 3, 5000.0


# --------------------------------------------------------------- test doubles --


class AntennaTemperature(AbstractOperator):
    """Write a full ``(n_time, n_freq)`` antenna temperature as the data.

    Test double, not physics: the package's ``SkyOperator`` carries a single
    scalar amplitude, and the whole question here is what happens when the
    antenna temperature is given one free parameter PER CELL instead.
    """

    requires: ClassVar[tuple[str, ...]] = ("coords.time", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data",)

    t_ant: jax.Array

    def __call__(self, state):
        return state.with_data(self.t_ant)


class CalibrationTone(AbstractOperator):
    """Add a KNOWN per-channel signal ahead of the gain.

    This is the injected calibration tone the hearing asked about: a reference
    of known amplitude that the gain multiplies along with the sky.
    """

    requires: ClassVar[tuple[str, ...]] = ("data",)
    provides: ClassVar[tuple[str, ...]] = ("data",)

    tone: jax.Array

    def __call__(self, state):
        return state.with_data(state.data + self.tone[None, :])


class SinglePrecisionSky(AbstractOperator):
    """A model that pins its own prediction to float32.

    Test double for the precision guard. Nothing in the package does this
    today, which is exactly why it has to be a test double.
    """

    provides: ClassVar[tuple[str, ...]] = ("data",)

    amplitude: jax.Array

    def __call__(self, state):
        n_time = state.coords.time.shape[0]
        n_freq = state.coords.freq.shape[0]
        return state.with_data(
            (self.amplitude * jnp.ones((n_time, n_freq))).astype(jnp.float32)
        )


# ------------------------------------------------------------------- fixtures --


def _poly_basis(n: int, degree: int) -> jax.Array:
    x = jnp.linspace(-1.0, 1.0, n)
    return jnp.stack([x**k for k in range(degree)], axis=1)


TIME_BASIS = _poly_basis(N_TIME, 3)
FREQ_BASIS = _poly_basis(N_FREQ, 3)
# Deliberately not symmetric in i<->j: a coefficient matrix that happened to be
# symmetric would leave a whole family of mistakes invisible.
COEFF0 = jnp.array([[3000.0, -180.0, 40.0], [120.0, 25.0, -8.0], [-45.0, 6.0, 2.0]])
T_ANT0 = TIME_BASIS @ COEFF0 @ FREQ_BASIS.T
GAIN0 = 1.5 + 0.05 * jnp.arange(N_TIME, dtype=float)


def make_pipeline(tone_kelvin: float) -> Pipeline:
    """``data[t, f] = gain[t] * (T_ant[t, f] + tone[f])``, tone known."""
    tone = jnp.zeros(N_FREQ).at[TONE_CHANNEL].set(tone_kelvin)
    return Pipeline(
        AntennaTemperature(t_ant=T_ANT0),
        CalibrationTone(tone=tone),
        GainOperator(gain=GAIN0),
        names=("t_ant", "tone", "gain"),
    )


def free_space() -> ParameterSpace:
    """One free antenna temperature per (time, frequency) cell."""
    return ParameterSpace(
        latents=[Latent("gain", init=GAIN0), Latent("t_ant", init=T_ANT0)],
        bindings=[
            Bind("gain", into=lambda p: p["gain"].gain),
            Bind("t_ant", into=lambda p: p["t_ant"].t_ant),
        ],
    )


def basis_space() -> ParameterSpace:
    """The same antenna temperature through a (3, 3) time x frequency basis."""
    return ParameterSpace(
        latents=[Latent("gain", init=GAIN0), Latent("t_coeff", init=COEFF0)],
        bindings=[
            Bind("gain", into=lambda p: p["gain"].gain),
            Bind(
                "t_coeff",
                into=lambda p: p["t_ant"].t_ant,
                fn=lambda c: TIME_BASIS @ c @ FREQ_BASIS.T,
            ),
        ],
    )


def make_state() -> State:
    return State(
        coords=Coordinates(
            time=jnp.arange(N_TIME, dtype=float),
            freq=jnp.linspace(60e6, 85e6, N_FREQ),
        ),
        meta={"telescope": "RHINO", "obs_id": "identifiability-000"},
    )


STATE = make_state()


@pytest.fixture
def state():
    return make_state()


# --------------------------------------------------------------- the headline --


class TestTheMotivatingCase:
    """The four-row table that changed the project's plan.

    A known calibration tone buys EXACTLY NOTHING against a free-per-cell
    antenna temperature and EVERYTHING against a frequency-smooth one. No
    other diagnostic in the package can say this.
    """

    def test_the_four_row_table(self, state):
        rows = {
            ("free", "on"): identifiability(free_space(), make_pipeline(TONE_KELVIN), state),
            ("free", "off"): identifiability(free_space(), make_pipeline(0.0), state),
            ("basis", "on"): identifiability(basis_space(), make_pipeline(TONE_KELVIN), state),
            ("basis", "off"): identifiability(basis_space(), make_pipeline(0.0), state),
        }
        table = {k: (r.n_par, r.rank, r.nullity) for k, r in rows.items()}
        assert table == {
            ("free", "on"): (72, 64, 8),
            ("free", "off"): (72, 64, 8),
            ("basis", "on"): (17, 17, 0),
            ("basis", "off"): (17, 16, 1),
        }, table

    def test_the_tone_buys_nothing_against_a_free_per_cell_temperature(self, state):
        """The free cell at the tone's channel absorbs the gain sample by sample,
        so nullity stays at n_time whether the tone is there or not."""
        on = identifiability(free_space(), make_pipeline(TONE_KELVIN), state)
        off = identifiability(free_space(), make_pipeline(0.0), state)
        assert on.nullity == off.nullity == N_TIME

    def test_the_tone_buys_everything_against_a_basis_temperature(self, state):
        """A delta at one channel is not in the span of three smooth frequency
        basis functions, so it cannot be reabsorbed: 1 -> 0."""
        off = identifiability(basis_space(), make_pipeline(0.0), state)
        on = identifiability(basis_space(), make_pipeline(TONE_KELVIN), state)
        assert (off.nullity, on.nullity) == (1, 0)

    def test_the_two_parameterizations_are_told_apart(self, state):
        """The discrimination, stated as a comparison rather than two numbers:
        with the tone on, one design is degenerate 8 ways and the other is not
        degenerate at all. A test that only reported one of them would pass on
        an implementation that returned a constant."""
        free = identifiability(free_space(), make_pipeline(TONE_KELVIN), state)
        basis = identifiability(basis_space(), make_pipeline(TONE_KELVIN), state)
        assert free.nullity > 0
        assert basis.nullity == 0
        assert free.nullity != basis.nullity

    def test_the_measured_spectra_straddle_the_threshold_by_many_decades(self, state):
        """The numbers the default rtol is chosen against.

        The weakest IDENTIFIED direction of the basis model sits at ~7e-2 of
        the largest; its null direction at ~7e-17. The default 1e-8 is roughly
        the geometric centre of that gap, so it is about as far from flipping
        either verdict as a single number can be.
        """
        on = identifiability(basis_space(), make_pipeline(TONE_KELVIN), state)
        off = identifiability(basis_space(), make_pipeline(0.0), state)
        weakest_identified = on.weakest_identified
        largest_null = float(off.singular_values[off.rank] / off.singular_values[0])
        assert weakest_identified > 1e-3
        assert largest_null < 1e-13
        assert largest_null < DEFAULT_RANK_RTOL < weakest_identified

    def test_the_free_model_reports_the_hearing_s_own_ratio(self, state):
        """``weakest_identified`` is s[rank-1]/s[0]: for the free model the
        Jacobian is surjective, so every computed singular value is identified
        and the number is 1/sqrt(2) exactly — the 7.071e-01 the hearing
        reported. Pinned because it is the one row where an implementation that
        confused "smallest computed" with "smallest identified" still looks
        right."""
        free = identifiability(free_space(), make_pipeline(TONE_KELVIN), state)
        assert free.weakest_identified == pytest.approx(1.0 / np.sqrt(2.0), rel=1e-6)
        # ... while the FULL spectrum, padded to n_par, ends in exact zeros,
        # because 72 parameters cannot be identified by 64 data points.
        assert free.singular_values.shape == (72,)
        assert float(free.singular_values[-1]) == 0.0
        assert free.n_data == 64


# ------------------------------------------------------- naming the null space --


class TestNamedNullDirections:
    """An unnamed null space tells a user they have a problem and nothing about
    which. The naming is the whole value."""

    def test_the_direction_names_both_latents_it_mixes(self, state):
        report = identifiability(basis_space(), make_pipeline(0.0), state)
        share = report.participation(0)
        assert set(share) == {"gain", "t_coeff"}
        # The bilinear degeneracy g -> a*g, T -> T/a puts comparable weight on
        # both. A direction that lived entirely in one latent would be a
        # different (and wrong) statement about the model.
        assert share["gain"] == pytest.approx(0.5, abs=0.05)
        assert share["t_coeff"] == pytest.approx(0.5, abs=0.05)
        assert sum(share.values()) == pytest.approx(1.0, abs=1e-6)

    def test_the_direction_is_shaped_like_the_latents(self, state):
        report = identifiability(basis_space(), make_pipeline(0.0), state)
        direction = report.direction(0)
        assert direction["gain"].shape == (N_TIME,)
        assert direction["t_coeff"].shape == (3, 3)

    def test_moving_along_the_direction_really_does_not_move_the_model(self, state):
        """The end-to-end statement, in the coordinates a caller acts in.

        ``direction`` is in RAW latent units, so adding a small multiple of it
        to the latents must leave the prediction unchanged to first order --
        while a random direction of the same size must not. This is what pins
        that the per-name split is assembled correctly: a direction whose gain
        and t_coeff halves were swapped or mis-scaled would fail here even
        though every shape still matched.
        """
        space = basis_space()
        pipeline = make_pipeline(0.0)
        forward, values0 = space.forward_fn(pipeline, state)
        report = identifiability(space, pipeline, state)
        direction = report.direction(0)

        step = 1e-3
        base = forward(values0)
        along = forward({k: values0[k] + step * jnp.asarray(direction[k]) for k in values0})
        key = jax.random.key(0)
        random = {
            k: values0[k]
            + step * jax.random.normal(jax.random.fold_in(key, i), jnp.shape(values0[k]))
            for i, k in enumerate(values0)
        }
        away = forward(random)

        moved_along = float(jnp.max(jnp.abs(along - base)))
        moved_away = float(jnp.max(jnp.abs(away - base)))
        assert moved_along < 1e-3 * moved_away, (moved_along, moved_away)

    def test_the_report_uses_DECLARATION_order_not_dict_order(self, state):
        """Flattening a ``{name: array}`` dict sorts the keys; a report that
        flattened one way and named the other would attribute a degeneracy to
        the wrong latent while every shape still checked out.

        The fixture is deliberately asymmetric: the degeneracy is between
        ``sky_scale`` and ``load_scale``, which sit at opposite ends of the
        declaration order and at DIFFERENT ends of the sorted order, while
        ``tone_amps`` — the one in the middle when sorted — carries none of it.
        """
        pipeline = Pipeline(
            AntennaTemperature(t_ant=T_ANT0),
            CalibrationTone(tone=jnp.zeros(N_FREQ)),
            names=("t_ant", "tone"),
        )
        space = ParameterSpace(
            latents=[
                Latent("sky_scale", init=jnp.array(1.0)),
                Latent("tone_amps", init=jnp.array([10.0, -4.0])),
                Latent("load_scale", init=jnp.array(0.0)),
            ],
            bindings=[
                Bind(
                    ("sky_scale", "load_scale"),
                    into=lambda p: p["t_ant"].t_ant,
                    fn=lambda s, load: (s + load) * T_ANT0,
                ),
                Bind(
                    "tone_amps",
                    into=lambda p: p["tone"].tone,
                    fn=lambda a: jnp.zeros(N_FREQ).at[2].set(a[0]).at[5].set(a[1]),
                ),
            ],
        )
        assert sorted(space.names) != list(space.names), "fixture must not be sort-stable"

        report = identifiability(space, pipeline, state)
        assert report.names == ("sky_scale", "tone_amps", "load_scale")
        assert report.nullity == 1
        share = report.participation(0)
        assert share["sky_scale"] == pytest.approx(0.5, abs=0.05)
        assert share["load_scale"] == pytest.approx(0.5, abs=0.05)
        assert share["tone_amps"] == pytest.approx(0.0, abs=1e-6)

    def test_an_out_of_range_direction_is_refused(self, state):
        """Indexing a JAX array out of range CLAMPS rather than raising, so
        ``direction(5)`` on a 1-dimensional null space would silently return
        direction 0 again."""
        report = identifiability(basis_space(), make_pipeline(0.0), state)
        assert report.nullity == 1
        with pytest.raises(StateValidationError, match="null direction"):
            report.direction(1)
        with pytest.raises(StateValidationError, match="null direction"):
            report.participation(-1)

    def test_a_fully_identified_model_has_no_directions_to_ask_for(self, state):
        report = identifiability(basis_space(), make_pipeline(TONE_KELVIN), state)
        assert report.nullity == 0
        assert report.null_space.shape == (0, 17)
        with pytest.raises(StateValidationError, match="null direction"):
            report.direction(0)

    def test_the_null_space_is_whole_when_there_are_fewer_data_than_parameters(self, state):
        """The headline case, and the one an SVD shortcut silently empties.

        The free-per-cell model has 64 data points against 72 parameters. Taken
        with ``full_matrices=False`` the SVD returns a (64, 72) ``Vh`` with no
        rows past index 64, so ``right[rank:]`` with ``rank = 64`` is EMPTY
        while ``nullity`` still reports 8 — and every direction the report
        names becomes unreachable. Every OTHER model in this file is
        over-determined, where the two spellings agree, so nothing else here
        can tell them apart. The shape is asserted against ``nullity`` and
        ``n_par`` rather than against literals alone, so the two records of one
        fact cannot drift apart quietly.
        """
        free = identifiability(free_space(), make_pipeline(TONE_KELVIN), state)
        assert free.n_data < free.n_par, (free.n_data, free.n_par)
        assert (free.n_data, free.n_par, free.nullity) == (64, 72, N_TIME)
        assert free.null_space.shape == (free.nullity, free.n_par) == (N_TIME, 72)

        # The LAST direction has to be reachable too, not just the first: a null
        # space truncated to any size short of `nullity` fails here.
        for index in (0, free.nullity - 1):
            direction = free.direction(index)
            assert direction["gain"].shape == (N_TIME,)
            assert direction["t_ant"].shape == (N_TIME, N_FREQ)
            assert np.all(np.isfinite(direction["gain"])), direction
            assert np.all(np.isfinite(direction["t_ant"])), direction

    def test_moving_along_a_free_per_cell_direction_does_not_move_the_model(self, state):
        """The end-to-end statement, on the UNDER-determined model.

        The same check ``test_moving_along_the_direction_really_does_not_move
        _the_model`` makes for the basis model, repeated where ``direction`` has
        to survive a null space the SVD only returns under
        ``full_matrices=True``. Measured at step 1e-3: **9.77e-4** along the
        null direction against **9.38** along a random one, four decades apart.
        """
        space = free_space()
        pipeline = make_pipeline(TONE_KELVIN)
        forward, values0 = space.forward_fn(pipeline, state)
        report = identifiability(space, pipeline, state)
        direction = report.direction(0)

        step = 1e-3
        base = forward(values0)
        along = forward({k: values0[k] + step * jnp.asarray(direction[k]) for k in values0})
        key = jax.random.key(0)
        random = {
            k: values0[k]
            + step * jax.random.normal(jax.random.fold_in(key, i), jnp.shape(values0[k]))
            for i, k in enumerate(values0)
        }
        away = forward(random)

        moved_along = float(jnp.max(jnp.abs(along - base)))
        moved_away = float(jnp.max(jnp.abs(away - base)))
        assert moved_along < 1e-3 * moved_away, (moved_along, moved_away)

    def test_a_report_whose_null_space_disagrees_with_its_nullity_is_refused(self, state):
        """``nullity`` and ``null_space`` are two records of one fact.

        ``_row``'s bounds check trusts the first; the lookup after it uses the
        second. A truncated null space satisfies the bounds check and then
        indexes off the end of the array, and numpy's bare ``IndexError`` names
        neither the cause nor the repair. Constructed here directly rather than
        by mutating the SVD, so the invariant is pinned as an invariant.
        """
        good = identifiability(free_space(), make_pipeline(TONE_KELVIN), state)

        # Too FEW rows — what full_matrices=False produces.
        truncated = dataclasses.replace(good, null_space=good.null_space[:0])
        assert (truncated.nullity, truncated.null_space.shape) == (N_TIME, (0, 72))
        with pytest.raises(StateValidationError, match="Inconsistent report"):
            truncated.direction(0)
        with pytest.raises(StateValidationError, match="Inconsistent report"):
            truncated.participation(0)

        # Too few COLUMNS — the other half of the shape, which would otherwise
        # reach `row / column_norms` and die on a numpy broadcast error instead.
        narrowed = dataclasses.replace(good, null_space=good.null_space[:, :5])
        assert (narrowed.nullity, narrowed.null_space.shape) == (N_TIME, (N_TIME, 5))
        with pytest.raises(StateValidationError, match="Inconsistent report"):
            narrowed.direction(0)

        # ... while the intact report is not disturbed by the same check.
        assert good.direction(0)["gain"].shape == (N_TIME,)
        assert good.participation(0)["gain"] == pytest.approx(0.5, abs=0.05)


# ------------------------------------------------- conditional blocks (Gibbs) --


class TestConditionalBlocks:
    """``names=`` is the Gibbs-block question: is THIS block identified, given
    the others held fixed?"""

    def test_every_block_is_identified_while_the_joint_is_not(self, state):
        """The measured failure, in one assertion.

        Each conditional has full rank — which is why every per-block guard
        passes, correctly — and the joint does not. A diagnostic that could
        only see one block at a time would report three clean bills of health
        on a model that is degenerate.
        """
        space, pipeline = basis_space(), make_pipeline(0.0)
        gain_only = identifiability(space, pipeline, state, names=("gain",))
        t_only = identifiability(space, pipeline, state, names=("t_coeff",))
        joint = identifiability(space, pipeline, state)

        assert gain_only.nullity == 0
        assert t_only.nullity == 0
        assert joint.nullity == 1

    def test_a_subset_reports_only_its_own_parameters(self, state):
        report = identifiability(
            basis_space(), make_pipeline(0.0), state, names=("t_coeff",)
        )
        assert report.names == ("t_coeff",)
        assert report.n_par == 9

    def test_names_may_be_given_in_any_order(self, state):
        report = identifiability(
            basis_space(), make_pipeline(0.0), state, names=("t_coeff", "gain")
        )
        assert report.names == ("t_coeff", "gain")
        assert report.nullity == 1
        share = report.participation(0)
        assert share["gain"] == pytest.approx(0.5, abs=0.05)

    def test_a_bare_string_is_one_name_not_four_characters(self, state):
        """``Bind`` takes "one or many" for its latents; so does this. Without
        the normalisation ``names="gain"`` iterates into ``('g','a','i','n')``
        and comes back as four undeclared latents."""
        one = identifiability(basis_space(), make_pipeline(0.0), state, names="gain")
        tupled = identifiability(
            basis_space(), make_pipeline(0.0), state, names=("gain",)
        )
        assert one.names == ("gain",) == tupled.names
        assert one.n_par == tupled.n_par == N_TIME

    def test_at_moves_the_evaluation_point(self, state):
        """Identifiability is a LOCAL property, so a Gibbs sweep has to ask it
        where the sampler currently is — not where the space was declared.

        With the gain conditioned to zero the antenna-temperature block stops
        reaching the data at all: every one of its nine parameters becomes a
        null direction. At the declared gain none of them is.
        """
        space, pipeline = basis_space(), make_pipeline(0.0)
        here = identifiability(space, pipeline, state, names=("t_coeff",))
        there = identifiability(
            space, pipeline, state, names=("t_coeff",), at={"gain": jnp.zeros(N_TIME)}
        )
        assert here.nullity == 0
        assert there.nullity == 9
        # Nothing at all is identified there, so the headline ratio has no
        # meaning and must say so rather than divide zero by zero.
        assert there.rank == 0
        assert there.weakest_identified == 0.0
        assert here.weakest_identified > 0.0

    def test_it_works_through_a_raw_bind_space(self, state):
        """``ParameterSpace.raw`` is the escape hatch, and it reaches the model
        by a different route (an opaque bind function rather than compiled
        ``Bind`` blocks). Everything here goes through ``forward_fn``, so it
        should not care — pinned rather than assumed, because "should not care"
        is how untested paths get described right up until they break.
        """
        pipeline = make_pipeline(0.0)
        space = ParameterSpace.raw(
            latents=[Latent("gain", init=GAIN0), Latent("t_coeff", init=COEFF0)],
            bind=lambda p, v: eqx.tree_at(
                lambda q: (q["gain"].gain, q["t_ant"].t_ant),
                p,
                (v["gain"], TIME_BASIS @ v["t_coeff"] @ FREQ_BASIS.T),
            ),
        )
        report = identifiability(space, pipeline, state)
        assert (report.n_par, report.rank, report.nullity) == (17, 16, 1)
        share = report.participation(0)
        assert share["gain"] == pytest.approx(0.5, abs=0.05)

    def test_at_rejects_an_unknown_name(self, state):
        with pytest.raises(ParameterSpaceError, match="not a latent"):
            identifiability(
                basis_space(), make_pipeline(0.0), state, at={"nope": jnp.array(1.0)}
            )


# ------------------------------------------------------- column normalisation --


class TestColumnNormalisation:
    """Without it the rank verdict reports UNITS rather than identifiability."""

    @staticmethod
    def _mixed_scale_model():
        """Two perfectly identified latents whose natural scales differ by 1e10.

        ``big`` drives a 1e5 K signal, ``small`` a 1e-5 K one, into DISJOINT
        frequency channels — so the two are exactly orthogonal and the model is
        unambiguously of rank 2.
        """
        pipeline = Pipeline(
            AntennaTemperature(t_ant=jnp.zeros((N_TIME, N_FREQ))),
            CalibrationTone(tone=jnp.zeros(N_FREQ)),
            names=("t_ant", "tone"),
        )
        space = ParameterSpace(
            latents=[Latent("big", init=jnp.array(1.0)), Latent("small", init=jnp.array(1.0))],
            bindings=[
                Bind(
                    ("big", "small"),
                    into=lambda p: p["tone"].tone,
                    fn=lambda b, s: jnp.zeros(N_FREQ).at[1].set(1e5 * b).at[6].set(1e-5 * s),
                ),
            ],
        )
        return space, pipeline

    def test_a_1e10_scale_gap_does_not_manufacture_a_null_direction(self, state):
        space, pipeline = self._mixed_scale_model()
        report = identifiability(space, pipeline, state)
        assert (report.n_par, report.rank, report.nullity) == (2, 2, 0)

    def test_the_same_model_without_normalisation_would_be_called_degenerate(
        self, state
    ):
        """The counterfactual, computed here rather than asserted by assertion.

        The RAW Jacobian's singular values differ by 1e-10, which is below the
        default rtol — so an implementation that skipped the column scaling
        would report nullity 1 for a model that has no null direction at all.
        """
        space, pipeline = self._mixed_scale_model()
        forward, values0 = space.forward_fn(pipeline, state)
        order = ("big", "small")

        def flat(x):
            return jnp.ravel(forward({**values0, **{n: x[i] for i, n in enumerate(order)}}))

        raw = jax.jacfwd(flat)(jnp.array([values0[n] for n in order]))
        raw_spectrum = jnp.linalg.svd(raw, compute_uv=False)
        raw_ratio = float(raw_spectrum[-1] / raw_spectrum[0])
        assert raw_ratio < DEFAULT_RANK_RTOL, raw_ratio

        report = identifiability(space, pipeline, state)
        assert report.nullity == 0
        assert report.weakest_identified > DEFAULT_RANK_RTOL

    def test_a_latent_with_no_first_order_effect_is_reported_not_NaN(self, state):
        """A zero Jacobian column is an exact null direction; dividing it by its
        own zero norm turns the whole report into NaN, and a NaN spectrum
        reports rank 0 for every model."""
        pipeline = Pipeline(
            AntennaTemperature(t_ant=jnp.zeros((N_TIME, N_FREQ))),
            CalibrationTone(tone=jnp.zeros(N_FREQ)),
            names=("t_ant", "tone"),
        )
        space = ParameterSpace(
            latents=[
                Latent("live", init=jnp.array(1.0)),
                # d(x^2)/dx = 0 at x = 0: bound, reached, and flat.
                Latent("flat", init=jnp.array(0.0)),
            ],
            bindings=[
                Bind(
                    ("live", "flat"),
                    into=lambda p: p["tone"].tone,
                    fn=lambda a, b: jnp.zeros(N_FREQ).at[1].set(a).at[6].set(b**2),
                ),
            ],
        )
        report = identifiability(space, pipeline, state)
        assert np.all(np.isfinite(report.singular_values))
        assert (report.n_par, report.rank, report.nullity) == (2, 1, 1)
        share = report.participation(0)
        assert share["flat"] == pytest.approx(1.0, abs=1e-9)
        assert share["live"] == pytest.approx(0.0, abs=1e-9)

    @staticmethod
    def _zero_column_model():
        """A live latent and a DEAD one, deliberately hard to confuse.

        ``live`` is ``(2,)`` and drives two channels at scales 1e5 apart;
        ``flat`` is ``(3,)`` and is bound through ``b**2`` at ``b = 0``, so all
        three of its Jacobian columns are exactly zero. Different shapes AND
        different scales AND different sizes, because this file has twice
        shipped a test blinded by a symmetric fixture — a direction that
        carried the wrong latent, or the right one mis-scaled, must not be able
        to pass here by coincidence.
        """
        pipeline = Pipeline(
            AntennaTemperature(t_ant=jnp.zeros((N_TIME, N_FREQ))),
            CalibrationTone(tone=jnp.zeros(N_FREQ)),
            names=("t_ant", "tone"),
        )
        space = ParameterSpace(
            latents=[
                Latent("live", init=jnp.array([1.0, 1.0])),
                Latent("flat", init=jnp.zeros(3)),
            ],
            bindings=[
                Bind(
                    ("live", "flat"),
                    into=lambda p: p["tone"].tone,
                    fn=lambda a, b: (
                        jnp.zeros(N_FREQ)
                        .at[1]
                        .set(1e3 * a[0])
                        .at[4]
                        .set(1e-2 * a[1])
                        .at[2]
                        .set(b[0] ** 2)
                        .at[5]
                        .set(b[1] ** 2)
                        .at[7]
                        .set(b[2] ** 2)
                    ),
                ),
            ],
        )
        return space, pipeline

    def test_a_direction_through_a_zero_column_is_finite_not_NaN(self, state):
        """What storing the SAFE column norms is actually for.

        ``column_norms`` keeps the GUARDED norms — 1.0 substituted for every
        exactly-zero column — and :meth:`direction` divides by them to undo the
        normalisation. Store the raw norms instead and that division is 1/0 on
        the supported entry and 0/0 on the rest, so every direction through a
        dead latent comes back as ``[nan, nan, inf]``. The unit-norm rescale
        cannot repair it either: ``norm`` is then NaN, ``NaN > 0.0`` is False,
        and the fallback quietly divides by 1.0 and preserves it.

        :meth:`participation` never touches ``column_norms``, so it stays
        correct and cannot see any of this — which is exactly why the existing
        zero-column test above, which only asks for shares, passes against a
        raw-norm store. Only ``direction`` can catch it.

        A latent the prediction does not depend on at all is an ordinary way to
        discover a modelling mistake, so this is the path a user hits on the
        day the diagnostic is doing its job.
        """
        space, pipeline = self._zero_column_model()
        report = identifiability(space, pipeline, state)
        assert (report.n_par, report.rank, report.nullity) == (5, 2, 3)

        # The stored norms are the safe ones: no zero survives to be divided by.
        assert np.all(report.column_norms > 0.0), report.column_norms
        assert report.column_norms.shape == (5,)
        # ... and they are the REAL norms where the column is live, not 1.0 for
        # everything — a store that substituted 1.0 unconditionally would undo
        # the column normalisation `direction` exists to reverse.
        assert report.column_norms[0] == pytest.approx(1e3 * np.sqrt(N_TIME), rel=1e-6)
        assert report.column_norms[1] == pytest.approx(1e-2 * np.sqrt(N_TIME), rel=1e-6)
        assert np.all(report.column_norms[2:] == 1.0)

        for index in range(report.nullity):
            direction = report.direction(index)
            assert direction["live"].shape == (2,)
            assert direction["flat"].shape == (3,)
            assert np.all(np.isfinite(direction["live"])), (index, direction)
            assert np.all(np.isfinite(direction["flat"])), (index, direction)
            # Documented contract: unit 2-norm over the flat vector.
            flat_vector = np.concatenate([direction["live"], direction["flat"]])
            assert float(np.linalg.norm(flat_vector)) == pytest.approx(1.0)
            # The degeneracy is the DEAD latent's, entirely — and the live
            # latent's two very differently scaled halves are both untouched.
            assert np.allclose(direction["live"], 0.0), (index, direction)
            assert float(np.linalg.norm(direction["flat"])) == pytest.approx(1.0)


# -------------------------------------------------------------- the threshold --


class TestRankThreshold:
    def test_the_threshold_is_exposed_and_reported(self, state):
        report = identifiability(basis_space(), make_pipeline(0.0), state)
        assert report.rtol == DEFAULT_RANK_RTOL
        assert report.threshold == pytest.approx(
            DEFAULT_RANK_RTOL * float(report.singular_values[0])
        )

    def test_raising_rtol_moves_the_verdict(self, state):
        """A silently chosen threshold that flips a verdict is the bug this
        package likes least, so the knob is real and its effect is pinned: the
        basis model's weakest identified direction sits at ~7e-2, and an rtol
        above that reclassifies it as null."""
        pipeline = make_pipeline(TONE_KELVIN)
        strict = identifiability(basis_space(), pipeline, state, rtol=1e-8)
        loose = identifiability(basis_space(), pipeline, state, rtol=0.5)
        assert strict.nullity == 0
        assert loose.nullity > 0
        assert loose.rtol == 0.5

    def test_the_suite_pins_this_constant_more_tightly_than_the_physics(self, state):
        """Where a retune of ``DEFAULT_RANK_RTOL`` will fail, and why — in one place.

        The numerically justified window is wide: every tolerance between the
        SVD's own noise floor (~1e-13) and the basis model's weakest identified
        direction (4.8e-5) returns the same verdict, 8.7 decades of freedom.
        The SUITE allows 2.5, because two counterfactuals elsewhere in this file
        are stated against the default rather than against a literal:

        * below **1.0e-10** the mixed-scale fixture's raw spectrum ratio stops
          falling under the tolerance, and ``test_the_same_model_without
          _normalisation_would_be_called_degenerate`` demonstrates nothing;
        * above **3.1168e-8** the basis model's float32 null direction falls
          under the tolerance, single precision gets the verdict RIGHT, and
          ``test_float32_would_have_got_the_verdict_wrong`` is simply false.

        Both are real claims about the constant rather than drift catchers, so
        the narrow window is kept rather than loosened. What is added is that it
        is *findable*: a retuner who reads only the constant's docstring sees
        8.7 decades of headroom and then gets two failures in unrelated classes.
        This test names the window, and the constant's docstring points here.
        """
        justified_low, justified_high = 1e-13, 4.822138e-05
        pinned_low, pinned_high = 1.0e-10, 3.116758e-08

        assert justified_low < pinned_low, "the suite's floor is above the noise floor"
        assert pinned_low < DEFAULT_RANK_RTOL < pinned_high, DEFAULT_RANK_RTOL
        assert pinned_high < justified_high, "the suite's ceiling is the tighter one"

        # The two ends are measured numbers, so pin them to the fixtures they
        # come from. A fixture that drifts is then caught here, with the window
        # in view, rather than as a puzzling failure two classes away.
        space, pipeline = TestColumnNormalisation._mixed_scale_model()
        forward, values0 = space.forward_fn(pipeline, state)
        order = ("big", "small")

        def flat(x):
            return jnp.ravel(forward({**values0, **{n: x[i] for i, n in enumerate(order)}}))

        raw = jax.jacfwd(flat)(jnp.array([values0[n] for n in order]))
        raw_spectrum = jnp.linalg.svd(raw, compute_uv=False)
        assert float(raw_spectrum[-1] / raw_spectrum[0]) == pytest.approx(pinned_low, rel=1e-3)

        # The upper end's own measurement lives in the float32 test, which has
        # to skip under x64; the number it straddles is this model's, and it is
        # precision-independent because the diagnostic forces float64.
        off = identifiability(basis_space(), make_pipeline(0.0), state)
        assert off.weakest_identified == pytest.approx(justified_high, rel=1e-3)


# ---------------------------------------------------------------- precision --


class TestPrecision:
    """The verdict is a number at 1e-17. float32 cannot represent that as a
    signal, so the diagnostic runs in float64 whatever the caller's precision."""

    def test_the_report_is_float64_whatever_the_caller_s_precision(self, state):
        report = identifiability(basis_space(), make_pipeline(0.0), state)
        assert report.singular_values.dtype == np.float64
        assert report.null_space.dtype == np.float64
        assert report.jacobian.dtype == np.float64

    def test_the_arrays_are_numpy_so_they_survive_leaving_the_context(self, state):
        """A float64 JAX array that escapes an x64 context truncates -- and
        warns -- the moment a default-precision caller touches it, throwing
        away exactly the precision this diagnostic went to trouble to get."""
        report = identifiability(basis_space(), make_pipeline(0.0), state)
        assert isinstance(report.singular_values, np.ndarray)
        assert isinstance(report.direction(0)["gain"], np.ndarray)
        # the operation that would have warned and truncated on a JAX array
        assert float(np.sum(report.singular_values**2)) > 0.0

    def test_float32_would_have_got_the_verdict_wrong(self, state):
        """Why x64 is not optional, measured on this exact model.

        Computed in single precision the null direction of the basis model
        surfaces at ~3e-8 of the largest singular value -- ABOVE the default
        1e-8 rtol -- so the same rank test would report the degenerate model as
        fully identified. In float64 the same number is ~7e-17.
        """
        if jax.config.read("jax_enable_x64"):
            pytest.skip("this is the single-precision floor")
        space, pipeline = basis_space(), make_pipeline(0.0)
        forward, values0 = space.forward_fn(pipeline, state)
        order = space.names
        sizes = [int(jnp.size(values0[n])) for n in order]

        def flat(x):
            out, off = {}, 0
            for name, size in zip(order, sizes, strict=True):
                out[name] = jnp.reshape(x[off : off + size], jnp.shape(values0[name]))
                off += size
            return jnp.ravel(forward(out))

        x0 = jnp.concatenate([jnp.ravel(values0[n]) for n in order])
        jac = jax.jacfwd(flat)(x0)
        assert jac.dtype == jnp.float32
        norms = jnp.linalg.norm(jac, axis=0)
        spectrum = jnp.linalg.svd(jac / norms, compute_uv=False)
        single = float(spectrum[-1] / spectrum[0])
        assert single > DEFAULT_RANK_RTOL, single
        assert int(jnp.sum(spectrum > DEFAULT_RANK_RTOL * spectrum[0])) == 17

        # ... and the function, run from this same float32 session, gets it right.
        report = identifiability(space, pipeline, state)
        assert report.nullity == 1
        assert float(report.singular_values[16] / report.singular_values[0]) < 1e-13

    def test_it_does_not_disturb_a_caller_who_has_not_enabled_x64(self, state):
        before = jax.config.read("jax_enable_x64")
        default_dtype = jnp.zeros(()).dtype
        identifiability(basis_space(), make_pipeline(0.0), state)
        assert jax.config.read("jax_enable_x64") is before
        assert jnp.zeros(()).dtype == default_dtype

    def test_the_flag_is_restored_even_when_the_analysis_raises(self, state):
        """A restore that is not in a ``finally`` leaves the whole process in
        x64 after one bad call, changing the dtype of every array the caller
        makes afterwards.

        The failure has to be raised from INSIDE the x64 block for this to test
        anything, so it uses the precision guard rather than a name guard —
        every name and dtype check runs before the context is entered, and a
        test built on one of those would pass against no ``finally`` at all.
        """
        before = jax.config.read("jax_enable_x64")
        default_dtype = jnp.zeros(()).dtype
        with pytest.raises(StateValidationError):
            identifiability(*self._single_precision_model(state), state)
        assert jax.config.read("jax_enable_x64") is before
        assert jnp.zeros(()).dtype == default_dtype

    @staticmethod
    def _single_precision_model(state):
        pipeline = Pipeline(SinglePrecisionSky(amplitude=jnp.array(3000.0)), names=("sky",))
        space = ParameterSpace.direct(
            "amplitude", init=jnp.array(3000.0), into=lambda p: p["sky"].amplitude
        )
        return space, pipeline

    def test_a_model_pinned_to_float32_is_refused(self, state):
        """A model that computes in single precision however the config is set
        cannot support a rank verdict at 1e-8: its own roundoff is larger."""
        space, pipeline = self._single_precision_model(state)
        with pytest.raises(StateValidationError, match="float32|single precision"):
            identifiability(space, pipeline, state)

    def test_x64_subprocess_gives_the_same_verdict(self):
        """The other precision, in a fresh interpreter because
        ``jax_enable_x64`` is process-global."""
        script = f"""
import sys
sys.path.insert(0, {os.path.dirname(__file__)!r})
import jax
assert jax.config.read("jax_enable_x64")
from test_identifiability import basis_space, free_space, make_pipeline, make_state
from rheplicant.inference.identifiability import identifiability

state = make_state()
table = {{}}
for label, space in (("free", free_space), ("basis", basis_space)):
    for tone, amp in (("on", 5000.0), ("off", 0.0)):
        r = identifiability(space(), make_pipeline(amp), state)
        table[(label, tone)] = (r.n_par, r.rank, r.nullity)
print(table)
assert table == {{
    ("free", "on"): (72, 64, 8),
    ("free", "off"): (72, 64, 8),
    ("basis", "on"): (17, 17, 0),
    ("basis", "off"): (17, 16, 1),
}}, table
assert jax.config.read("jax_enable_x64"), "x64 must survive the call"
"""
        env = {**os.environ, "JAX_ENABLE_X64": "1"}
        done = subprocess.run(
            [sys.executable, "-c", script], env=env, capture_output=True, text=True
        )
        assert done.returncode == 0, done.stdout + done.stderr
        # A subprocess that exited 0 without running the analysis would look
        # identical from here, so insist on seeing the table it printed.
        assert "(72, 64, 8)" in done.stdout, done.stdout


# ------------------------------------------------------------------- refusals --


class TestGuards:
    def test_an_unknown_latent_name_is_refused(self, state):
        with pytest.raises(ParameterSpaceError, match="not a latent"):
            identifiability(basis_space(), make_pipeline(0.0), state, names=("nope",))

    def test_a_repeated_latent_name_is_refused(self, state):
        """Two copies of one latent are exactly degenerate with each other, so
        a repeat manufactures a null direction that says nothing about the
        model."""
        with pytest.raises(ParameterSpaceError, match="more than once|repeated"):
            identifiability(
                basis_space(), make_pipeline(0.0), state, names=("gain", "gain")
            )

    def test_an_empty_selection_is_refused(self, state):
        """``names=()`` would report nullity 0 over nothing at all, which reads
        as a clean bill of health for an empty Gibbs block."""
        with pytest.raises(ParameterSpaceError, match="at least one"):
            identifiability(basis_space(), make_pipeline(0.0), state, names=())

    def test_a_complex_latent_is_refused(self, state):
        """Matched on a phrase only the COMPLEX branch uses.

        Both dtype branches raise ParameterSpaceError, and a complex dtype is
        also not floating — so deleting the complex branch drops the latent into
        the non-floating one, whose message embeds the string ``complex64``. A
        test matching ``"complex"`` therefore passed against no complex guard at
        all; it was caught by mutation and is pinned on ``R-linear`` instead.
        """
        pipeline = make_pipeline(0.0)
        space = ParameterSpace(
            latents=[
                Latent("gain", init=GAIN0),
                Latent("coeff", init=jnp.ones(2) + 0j),
            ],
            bindings=[
                Bind("gain", into=lambda p: p["gain"].gain),
                Bind(
                    "coeff",
                    into=lambda p: p["t_ant"].t_ant,
                    fn=lambda c: jnp.real(jnp.sum(c)) * jnp.ones((N_TIME, N_FREQ)),
                ),
            ],
        )
        with pytest.raises(ParameterSpaceError, match="R-linear but not C-linear"):
            identifiability(space, pipeline, state)

    def test_an_integer_latent_is_refused(self, state):
        pipeline = make_pipeline(0.0)
        space = ParameterSpace(
            latents=[
                Latent("gain", init=GAIN0),
                Latent("count", init=jnp.array([1, 2, 3])),
            ],
            bindings=[
                Bind("gain", into=lambda p: p["gain"].gain),
                Bind(
                    "count",
                    into=lambda p: p["t_ant"].t_ant,
                    fn=lambda n: jnp.sum(n).astype(jnp.float32)
                    * jnp.ones((N_TIME, N_FREQ)),
                ),
            ],
        )
        with pytest.raises(ParameterSpaceError, match="not a continuous parameter"):
            identifiability(space, pipeline, state)

    def test_a_complex_latent_that_is_not_selected_is_not_refused(self, state):
        """The dtype rules are about the parameters being DIFFERENTIATED, so a
        complex latent held fixed in another Gibbs block is no obstacle."""
        pipeline = make_pipeline(0.0)
        space = ParameterSpace(
            latents=[
                Latent("gain", init=GAIN0),
                Latent("coeff", init=jnp.ones(2) + 0j),
            ],
            bindings=[
                Bind("gain", into=lambda p: p["gain"].gain),
                Bind(
                    "coeff",
                    into=lambda p: p["t_ant"].t_ant,
                    fn=lambda c: jnp.real(jnp.sum(c)) * jnp.ones((N_TIME, N_FREQ)),
                ),
            ],
        )
        report = identifiability(space, pipeline, state, names=("gain",))
        assert report.n_par == N_TIME
        assert report.nullity == 0
