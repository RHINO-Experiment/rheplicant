"""inference.observed: simulation, file, several observations, and the seed."""

import re
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.config import ConfigError
from rheplicant.config.sections import observed as observed_module
from rheplicant.config.sections.noise import build_noise
from rheplicant.config.sections.observed import build_observed
from rheplicant.config.sections.parameters import parse_latents
from rheplicant.config.sections.transforms import build_space
from rheplicant.config.sections.twin import build_fit_twin
from rheplicant.inference import HomoscedasticNoise, RadiometerNoise
from tests.config.inference_helpers import MODEL, context, state, twin


def harness(model=None, seeds=None, base_dir=None):
    ctx = context(**({"seeds": seeds} if seeds else {}),
                  **({"base_dir": base_dir} if base_dir else {}))
    full = twin(model, ctx)
    space = build_space(
        parse_latents({"g": {"init": 1.0, "linear": True,
                             "into": "gain.gain"}}, ctx),
        None, None, fit_twin=full, replaced=(), context=ctx)
    return ctx, full, space


class _FakeObservation:
    integration_time_s = 2.0
    channel_width_hz = 3.125e6
    aux: dict = {}


def build(spec, *, ctx, full, space, noise=None):
    return build_observed(spec, twin=full, fit_twin=full, space=space,
                          noise=noise or build_noise(
                              None, observation=_FakeObservation(),
                              context=ctx),
                          state=state(), observation=_FakeObservation(),
                          context=ctx)


class TestSimulation:
    def test_at_binds_the_truth_before_evaluating(self):
        ctx, full, space = harness()
        observed = build({"from": "simulation",
                          "at": {"g": 2.0}}, ctx=ctx, full=full, space=space)
        reference = space.bind(full, {"g": jnp.asarray(2.0)})(state()).data
        assert observed.primary == "primary"
        assert jnp.allclose(observed.entries["primary"], reference)
        assert observed.at["primary"]["g"] == pytest.approx(2.0)

    def test_at_an_unknown_latent_is_refused_listing_the_names(self):
        ctx, full, space = harness()
        with pytest.raises(ConfigError, match="ghost"):
            build({"from": "simulation", "at": {"ghost": 2.0}},
                  ctx=ctx, full=full, space=space)

    def test_at_without_parameters_is_refused(self):
        ctx, full, _ = harness()
        with pytest.raises(ConfigError, match="parameters"):
            build({"from": "simulation", "at": {"g": 2.0}},
                  ctx=ctx, full=full, space=None)

    def test_twin_must_be_full_or_fit(self):
        ctx, full, space = harness()
        with pytest.raises(ConfigError, match="full"):
            build({"from": "simulation", "twin": "both"},
                  ctx=ctx, full=full, space=space)

    def test_twin_fit_evaluates_the_repaired_twin_and_records_it(self):
        # The fit twin here is DISTINGUISHABLE from the full one (a deeper
        # trough), so twin: fit reading the full twin anyway cannot pass.
        ctx, full, space = harness()
        deeper = {"depth": {"value": 0.9, "unit": "K"},
                  "centre": {"value": 75.0, "unit": "MHz"},
                  "width": {"value": 5.0, "unit": "MHz"}}
        fit, _ = build_fit_twin({"replace": {"global_signal": deeper}},
                                full, ctx)
        observed = build_observed(
            {"from": "simulation", "twin": "fit"}, twin=full, fit_twin=fit,
            space=space, noise=build_noise(None,
                                           observation=_FakeObservation(),
                                           context=ctx),
            state=state(), observation=_FakeObservation(), context=ctx)
        init = dict(space.initial_values())
        assert jnp.allclose(observed.entries["primary"],
                            space.bind(fit, init)(state()).data)
        assert not jnp.allclose(observed.entries["primary"],
                                space.bind(full, init)(state()).data)
        assert observed.records["primary"]["twin"] == "fit"

    def test_a_clean_simulation_is_evaluated_at_the_declared_init(self):
        # init g=1.0 differs from the twin's declared gain 1.1, so a build
        # that skips binding the initial values shows up here.
        ctx, full, space = harness()
        observed = build({"from": "simulation"}, ctx=ctx, full=full,
                         space=space)
        reference = space.bind(full, dict(space.initial_values()))(
            state()).data
        assert jnp.allclose(observed.entries["primary"], reference)
        assert not jnp.allclose(observed.entries["primary"],
                                full(state()).data)


class TestRealise:
    def spec(self, kind, seed="runtime.seeds.observed_noise", **extra):
        realise = {"kind": kind, **extra}
        if seed is not None:
            realise["seed"] = {"from": seed}
        return {"from": "simulation", "realise": realise}

    def test_homoscedastic_scatter_is_reproducible_from_its_named_seed(self):
        ctx, full, space = harness(seeds={"observed_noise": 7})
        one = build(self.spec("homoscedastic",
                              sigma={"value": 0.5, "unit": "K"}),
                    ctx=ctx, full=full, space=space)
        two = build(self.spec("homoscedastic",
                              sigma={"value": 0.5, "unit": "K"}),
                    ctx=ctx, full=full, space=space)
        clean = build({"from": "simulation"}, ctx=ctx, full=full, space=space)
        assert jnp.allclose(one.entries["primary"], two.entries["primary"])
        assert not jnp.allclose(one.entries["primary"],
                                clean.entries["primary"])

    def test_a_different_declared_seed_draws_a_different_scatter(self):
        # Reproducibility alone cannot see a draw that ignores the seed --
        # key(0) twice is also reproducible. Two declared values must differ.
        spec = self.spec("homoscedastic", sigma={"value": 0.5, "unit": "K"})
        ctx7, full7, space7 = harness(seeds={"observed_noise": 7})
        ctx8, full8, space8 = harness(seeds={"observed_noise": 8})
        seven = build(spec, ctx=ctx7, full=full7, space=space7)
        eight = build(spec, ctx=ctx8, full=full8, space=space8)
        assert not jnp.allclose(seven.entries["primary"],
                                eight.entries["primary"])

    def test_the_draw_is_the_packages_realise_at_the_recorded_seed(self):
        # The seam claim, pinned bitwise: the data is NoiseModel.realise at
        # jax.random.key(recorded seed), nothing hand-written beside it.
        ctx, full, space = harness(seeds={"observed_noise": 7})
        observed = build(self.spec("homoscedastic",
                                   sigma={"value": 0.5, "unit": "K"}),
                         ctx=ctx, full=full, space=space)
        clean = build({"from": "simulation"}, ctx=ctx, full=full, space=space)
        assert observed.records["primary"]["seed"] == 7
        expected = HomoscedasticNoise(jnp.asarray(0.5, dtype=jnp.float32)
                                      ).realise(clean.entries["primary"],
                                                key=jax.random.key(7))
        assert jnp.array_equal(observed.entries["primary"], expected)

    def test_an_undeclared_seed_name_derives_by_blake2s_not_luck(self):
        ctx, full, space = harness()   # runtime.seeds is empty; seed is set
        observed = build(self.spec("homoscedastic",
                                   sigma={"value": 0.5, "unit": "K"}),
                         ctx=ctx, full=full, space=space)
        assert observed.records["primary"]["seed"] is not None

    def test_the_seed_is_required_on_a_drawing_kind(self):
        ctx, full, space = harness()
        with pytest.raises(ConfigError, match="seed"):
            build(self.spec("homoscedastic", seed=None,
                            sigma={"value": 0.5, "unit": "K"}),
                  ctx=ctx, full=full, space=space)

    def test_radiometer_scatter_is_multiplicative(self):
        ctx, full, space = harness()
        observed = build(self.spec("radiometer"),
                         ctx=ctx, full=full, space=space)
        clean = build({"from": "simulation"}, ctx=ctx, full=full, space=space)
        ratio = observed.entries["primary"] / clean.entries["primary"]
        fractional = 1.0 / np.sqrt(3.125e6 * 2.0)
        assert float(jnp.max(jnp.abs(ratio - 1.0))) < 6 * fractional

    def test_radiometer_realise_is_the_multiplicative_form_exactly(self):
        # The trough prediction is negative throughout, so the additive
        # d + |d| f w form agrees with d(1 + f w) in ratio MAGNITUDE and
        # differs only in the sign of every perturbation -- which the ratio
        # test above cannot see. Bitwise equality with the package's own
        # realise is the check that can.
        ctx, full, space = harness(seeds={"observed_noise": 11})
        observed = build(self.spec("radiometer"),
                         ctx=ctx, full=full, space=space)
        clean = build({"from": "simulation"}, ctx=ctx, full=full, space=space)
        assert observed.records["primary"]["seed"] == 11
        expected = RadiometerNoise(3.125e6, 2.0).realise(
            clean.entries["primary"], key=jax.random.key(11))
        assert jnp.array_equal(observed.entries["primary"], expected)

    def test_from_model_draws_with_the_declared_noise_model(self):
        ctx, full, space = harness()
        noise = build_noise({"kind": "homoscedastic",
                             "sigma": {"value": 0.5, "unit": "K"}},
                            observation=_FakeObservation(), context=ctx)
        observed = build(self.spec("from_model"), ctx=ctx, full=full,
                         space=space, noise=noise)
        clean = build({"from": "simulation"}, ctx=ctx, full=full, space=space)
        assert not jnp.allclose(observed.entries["primary"],
                                clean.entries["primary"])

    def test_from_model_with_kind_none_is_refused_by_name(self):
        ctx, full, space = harness()
        with pytest.raises(ConfigError, match="from_model"):
            build(self.spec("from_model"), ctx=ctx, full=full, space=space)

    def test_kind_none_takes_no_seed_and_adds_nothing(self):
        ctx, full, space = harness()
        observed = build({"from": "simulation", "realise": {"kind": "none"}},
                         ctx=ctx, full=full, space=space)
        clean = build({"from": "simulation"}, ctx=ctx, full=full, space=space)
        assert jnp.allclose(observed.entries["primary"],
                            clean.entries["primary"])


class TestFileForm:
    def test_an_npz_lands_shape_checked(self, tmp_path):
        np.savez(tmp_path / "night1.npz",
                 waterfall=np.ones((16, 8), dtype=np.float32))
        ctx, full, space = harness(base_dir=str(tmp_path))
        observed = build({"file": {"path": "night1.npz", "format": "npz",
                                   "key": "waterfall"}},
                         ctx=ctx, full=full, space=space)
        assert observed.entries["primary"].shape == (16, 8)
        assert observed.records["primary"]["from"] == "file"

    def test_a_wrong_shape_is_refused_exactly(self, tmp_path):
        np.savez(tmp_path / "night1.npz",
                 waterfall=np.ones((8, 16), dtype=np.float32))
        ctx, full, space = harness(base_dir=str(tmp_path))
        with pytest.raises(ConfigError, match="16, 8"):
            build({"file": {"path": "night1.npz", "format": "npz",
                            "key": "waterfall"}},
                  ctx=ctx, full=full, space=space)


class TestTheFileIsMatchedAgainstThePrediction:
    """Check C11, fixed in place: the reference is what the model PRODUCES.

    ``sections/observed.py`` compared a file's shape against ``_shape(context)``
    -- the time and frequency GRIDS -- while its own message said
    *"broadcast-compatible is the dangerous case (check C11)"*.  Measured with
    ``averaging: {n_chunk: 4}`` on ``(16, 8)`` grids the prediction is
    ``(4, 8)``, so the shipped code accepted the ``(16, 8)`` file and refused
    the ``(4, 8)`` one, and at ``n_chunk: 16`` (prediction ``(1, 8)``) it
    accepted a ``(16, 8)`` file that then broadcasts silently -- the exact
    case the sentence claims to guard.

    **Every test in this class declares ``n_chunk > 1``, and that is the
    whole point.**  Without an ``averaging:`` node the prediction shape EQUALS
    the grids, so the right implementation and the wrong one agree on every
    document and no test can tell them apart.  Measured at ``e0e024a``: no
    test in ``tests/config/`` drove this branch with a reshaping model at all.
    """

    MODEL_N_CHUNK_4 = {**MODEL, "averaging": {"n_chunk": 4}}

    def _harness(self, tmp_path, model):
        ctx = context(base_dir=str(tmp_path))
        full = twin(model, ctx)
        return ctx, full

    def _build(self, ctx, full, name, *, fit_twin=None, space=None):
        return build_observed(
            {"file": {"path": name, "format": "npz", "key": "w"}},
            twin=full, fit_twin=full if fit_twin is None else fit_twin,
            space=space,
            noise=build_noise(None, observation=_FakeObservation(),
                              context=ctx),
            state=state(), observation=_FakeObservation(), context=ctx)

    def _file(self, tmp_path, shape):
        name = f"n{shape[0]}x{shape[1]}.npz"
        np.savez(tmp_path / name, w=np.ones(shape, dtype=np.float32))
        return name

    def test_n_chunk_4_refuses_the_grid_shaped_file_and_accepts_the_prediction(
            self, tmp_path):
        """§5's named box: the INVERSION, stated as an inversion.

        At ``e0e024a`` this document accepted ``(16, 8)`` and refused
        ``(4, 8)``.  Both halves are asserted here, so restoring the shipped
        comparison fails this test twice rather than once.
        """
        ctx, full = self._harness(tmp_path, self.MODEL_N_CHUNK_4)
        grid_shaped = self._file(tmp_path, (16, 8))
        predicted = self._file(tmp_path, (4, 8))

        # The half that used to be ACCEPTED.
        with pytest.raises(ConfigError) as caught:
            self._build(ctx, full, grid_shaped)
        assert "the file holds shape (16, 8)" in str(caught.value)

        # The half that used to be REFUSED.
        observed = self._build(ctx, full, predicted)
        assert observed.entries["primary"].shape == (4, 8)

    def test_n_chunk_16_no_longer_accepts_a_file_that_would_broadcast(
            self, tmp_path):
        """The case the shipped sentence names and the shipped code missed.

        Prediction ``(1, 8)``; a ``(16, 8)`` file is broadcast-compatible with
        it, which is exactly *"the dangerous case"*.
        """
        ctx, full = self._harness(tmp_path, {**MODEL,
                                             "averaging": {"n_chunk": 16}})
        with pytest.raises(ConfigError) as caught:
            self._build(ctx, full, self._file(tmp_path, (16, 8)))
        assert "predicts (1, 8)" in str(caught.value)
        assert self._build(
            ctx, full, self._file(tmp_path, (1, 8))
        ).entries["primary"].shape == (1, 8)

    def test_the_refusal_names_the_prediction_and_keeps_the_clause_that_was_right(
            self, tmp_path):
        """S1: the whole sentence, by equality, in both of its two forms.

        Named by ``test_config_preflight.py``'s ``_ASSEMBLED_ELSEWHERE`` as
        the pin that stands between the corrected message and a silent
        rewording, so it pins the WHOLE text and not a fragment.
        """
        ctx, full = self._harness(tmp_path, self.MODEL_N_CHUNK_4)
        with pytest.raises(ConfigError) as caught:
            self._build(ctx, full, self._file(tmp_path, (16, 8)))
        assert str(caught.value) == (
            "inference.observed.primary: the file holds shape (16, 8); this "
            "run's fit twin predicts (4, 8). Exactly -- broadcast-compatible "
            "is the dangerous case (check C11). The time and frequency grids "
            "are (16, 8): the model reshapes the prediction before the data "
            "is compared, so the grids are not the shape to match."
        )

        # And the form taken when the prediction IS the grids: no aside, and
        # the clause that was right is still the last thing said.
        plain_ctx, plain = self._harness(tmp_path, MODEL)
        with pytest.raises(ConfigError) as caught:
            self._build(plain_ctx, plain, self._file(tmp_path, (4, 8)))
        assert str(caught.value) == (
            "inference.observed.primary: the file holds shape (4, 8); this "
            "run's fit twin predicts (16, 8). Exactly -- broadcast-compatible "
            "is the dangerous case (check C11)."
        )

    def test_applying_the_refusals_own_advice_makes_the_document_build(
            self, tmp_path):
        """S4's second half: take the remedy the message names, and pass.

        The message names the shape to match; a file written at that shape
        must then be accepted, with no second refusal waiting behind it.
        """
        ctx, full = self._harness(tmp_path, self.MODEL_N_CHUNK_4)
        with pytest.raises(ConfigError) as caught:
            self._build(ctx, full, self._file(tmp_path, (16, 8)))
        wanted = re.search(r"predicts \((\d+), (\d+)\)", str(caught.value))
        shape = (int(wanted.group(1)), int(wanted.group(2)))
        assert shape == (4, 8)
        observed = self._build(ctx, full, self._file(tmp_path, shape))
        assert observed.entries["primary"].shape == shape

    def test_it_reads_the_fit_twin_so_inference_twin_replace_is_walked(
            self, tmp_path):
        """0.3 E.10, and the TRAP the task body names.

        ``inference.twin.replace`` reaches ``build_node_operator`` and can
        change ``averaging``'s ``n_chunk``, and the ``file:`` form carries no
        ``twin:`` key to disambiguate with.  Reading the FULL twin here would
        demand ``(4, 8)`` for a document whose fit twin predicts ``(2, 8)``.

        So this check DOES walk ``inference.twin.replace`` -- by construction,
        because ``build_fit_twin`` has already applied it to the twin handed
        in, not by a second traversal of the document.
        """
        ctx, full = self._harness(tmp_path, self.MODEL_N_CHUNK_4)
        fit, replaced = build_fit_twin(
            {"replace": {"averaging": {"n_chunk": 8}}}, full, ctx)
        assert replaced == ("averaging",)

        # The fit twin predicts (2, 8); the FULL twin predicts (4, 8).
        observed = self._build(ctx, full, self._file(tmp_path, (2, 8)),
                               fit_twin=fit)
        assert observed.entries["primary"].shape == (2, 8)
        with pytest.raises(ConfigError, match=r"predicts \(2, 8\)"):
            self._build(ctx, full, self._file(tmp_path, (4, 8)), fit_twin=fit)

    @pytest.mark.parametrize("predicted", [(4, 8), (16, 4), (4, 4)])
    def test_the_aside_fires_on_any_axis_the_model_reshapes(
            self, tmp_path, predicted):
        """The gate is ``wanted == grids``, not ``wanted[0] == grids[0]``.

        Every document this class can cheaply build reshapes the TIME axis
        (``averaging: {n_chunk: N}``), so the gate was only ever exercised
        where the axes agreed or the first one differed -- and reducing it to
        axis 0 survived the whole suite. A model that reshaped only frequency
        would then print the refusal without the one clause the correction
        added. There is no shipped node that reshapes frequency alone, so the
        prediction is substituted directly rather than modelled.
        """
        ctx, full = self._harness(tmp_path, MODEL)
        with mock.patch.object(observed_module, "_predicted_shape",
                               lambda *a, **k: predicted):
            with pytest.raises(ConfigError) as caught:
                self._build(ctx, full, self._file(tmp_path, (2, 2)))
        assert "The time and frequency grids are (16, 8):" in str(caught.value)

    def test_the_aside_stays_silent_when_the_model_reshapes_nothing(
            self, tmp_path):
        """The other side of the same gate, so it cannot be widened either."""
        ctx, full = self._harness(tmp_path, MODEL)
        with pytest.raises(ConfigError) as caught:
            self._build(ctx, full, self._file(tmp_path, (2, 2)))
        assert "The time and frequency grids are" not in str(caught.value)

    def test_the_simulation_branch_gains_no_check_of_its_own(self, tmp_path):
        """S3's twin, and the TRAP that says do NOT fix it symmetrically.

        On the simulation branch the prediction IS the array, so there are
        never two shapes to compare.  A symmetric "fix" would be a comparison
        of a value against itself: green forever, and it would refuse nothing.
        """
        ctx, full = self._harness(tmp_path, self.MODEL_N_CHUNK_4)
        observed = build_observed(
            {"from": "simulation"}, twin=full, fit_twin=full, space=None,
            noise=build_noise(None, observation=_FakeObservation(),
                              context=ctx),
            state=state(), observation=_FakeObservation(), context=ctx)
        # The grids say (16, 8) and this is accepted at (4, 8) with no
        # refusal anywhere -- which is the shipped behaviour, unchanged.
        assert observed.entries["primary"].shape == (4, 8)

    def test_the_prediction_is_taken_by_eval_shape_and_nothing_is_computed(
            self, tmp_path):
        """R9, as a MECHANISM pin rather than a wall-clock one.

        The plan's bound for reading the twin is "one ``jax.eval_shape`` and
        no forward pass".  A timing assertion on ~1 ms would be decoration on
        a loaded box (measured: 1.03 ms cold, 1.21 ms warm); counting the
        ``eval_shape`` calls cannot be satisfied by a real forward pass, and
        an implementation that evaluated the twin instead drops this to zero.
        """
        ctx, full = self._harness(tmp_path, self.MODEL_N_CHUNK_4)
        calls = []
        real = jax.eval_shape

        def counted(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        with mock.patch.object(observed_module.jax, "eval_shape", counted):
            self._build(ctx, full, self._file(tmp_path, (4, 8)))
        assert len(calls) == 1


class TestSeveralObservations:
    def test_named_entries_each_simulate_their_own_truth(self):
        ctx, full, space = harness()
        observed = build({"primary": {"from": "simulation", "at": {"g": 1.1}},
                          "second": {"from": "simulation", "at": {"g": 1.5}}},
                         ctx=ctx, full=full, space=space)
        assert set(observed.entries) == {"primary", "second"}
        assert observed.primary == "primary"
        assert not jnp.allclose(observed.entries["primary"],
                                observed.entries["second"])

    def test_each_entrys_at_is_its_own_record(self):
        # Task 6's truth derivation reads ObservedBuild.at per entry; a dict
        # shared across entries would hand it the LAST entry's truth for all.
        ctx, full, space = harness()
        observed = build({"primary": {"from": "simulation", "at": {"g": 1.1}},
                          "second": {"from": "simulation", "at": {"g": 1.5}},
                          "third": {"from": "simulation"}},
                         ctx=ctx, full=full, space=space)
        assert float(observed.at["primary"]["g"]) == pytest.approx(1.1)
        assert float(observed.at["second"]["g"]) == pytest.approx(1.5)
        assert observed.at["third"] == {}

    def test_without_a_primary_the_default_is_unresolved(self):
        ctx, full, space = harness()
        observed = build({"a": {"from": "simulation"},
                          "b": {"from": "simulation"}},
                         ctx=ctx, full=full, space=space)
        assert observed.primary is None

    def test_an_entry_name_colliding_with_the_grammar_is_refused(self):
        ctx, full, space = harness()
        with pytest.raises(ConfigError, match="realise"):
            build({"realise": {"from": "simulation"}},
                  ctx=ctx, full=full, space=space)

    def test_unknown_keys_in_a_simulation_spec_are_swept(self):
        ctx, full, space = harness()
        with pytest.raises(ConfigError, match="sigma"):
            build({"from": "simulation", "sigma": 0.5},
                  ctx=ctx, full=full, space=space)
