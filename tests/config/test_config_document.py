"""load_document / run_forward: the whole build order, both observation forms."""

import inspect

import jax.numpy as jnp
import pytest

import rheplicant.config.document as document_module
from rheplicant.config import ConfigError
from rheplicant.config.document import ConfiguredRun, load_document, run_forward
from rheplicant.config.findings import Report
from rheplicant.config.findings import warn as warn_finding
from rheplicant.config.preflight import CHECKS as TEXT_CHECKS
from rheplicant.config.preflight import register as register_text
from rheplicant.config.sections.runs import RunResult, RunSpec


def synthetic_document():
    return {
        "schema_version": 1,
        "runtime": {"seed": 20260806},
        "observation": {
            "meta": {"telescope": "RHINO"},
            "freq": {"grid": {"linspace": {"start": 60.0, "stop": 85.0,
                                           "num": 8, "endpoint": True},
                              "unit": "MHz"}},
            "time": {"grid": {"arange": {"start": 0.0, "step": 2.0, "num": 16},
                              "unit": "s"}},
            "environment": {"temperature": {"value": 280.0, "unit": "K"}},
        },
        "resources": {
            "arrays": {"flat": {"ones": ["n_freq"]}},
        },
        "model": {
            "global_signal": {"depth": {"value": 0.5, "unit": "K"},
                              "centre": {"value": 75.0, "unit": "MHz"},
                              "width": {"value": 5.0, "unit": "MHz"}},
            "gain": {"gain": {"value": 1.1, "unit": "dimensionless"}},
            "noise": {"type": "NoiseOperator",
                      "sigma": {"value": 0.05, "unit": "K"}},
        },
        "variants": {
            "unity_gain": {"model": {"gain": {"gain": {"value": 1.0,
                                                       "unit": "dimensionless"}}}},
        },
        "runs": [{"kind": "forward"}],
    }


class TestLoadDocument:
    def test_the_synthetic_document_loads(self):
        run = load_document(synthetic_document())
        assert isinstance(run, ConfiguredRun)
        assert run.state.data is None
        assert run.state.coords.freq.shape == (8,)
        assert float(run.state.env.temperature) == pytest.approx(280.0)
        assert run.state.meta["telescope"] == "RHINO"
        assert run.state.key is not None
        assert {"global_signal", "gain", "noise"} <= set(run.twin.lit)
        assert "resources.arrays.flat" in run.context.resources

    def test_schema_version_is_required_and_must_be_one(self):
        doc = synthetic_document()
        del doc["schema_version"]
        with pytest.raises(ConfigError, match="schema_version"):
            load_document(doc)
        with pytest.raises(ConfigError, match="schema_version"):
            load_document({**synthetic_document(), "schema_version": 2})

    def test_required_sections_are_named_when_missing(self):
        doc = synthetic_document()
        del doc["model"]
        with pytest.raises(ConfigError, match="model"):
            load_document(doc)

    def test_an_unknown_section_is_refused_listing_the_twelve(self):
        with pytest.raises(ConfigError, match="observations"):
            load_document({**synthetic_document(), "observations": {}})

    @pytest.mark.parametrize(
        "section", ["outputs", "defaults", "plugins"]
    )
    def test_not_yet_owned_sections_name_where_they_are_handled(self, section):
        """Not "which plan will add it" -- that named internal history in a
        message a user reads, and it had gone stale: the work it pointed at
        shipped as the command line these three are handled by.

        Matched against the loader's own routing table rather than a quoted
        sentence, so re-wording the prose is free and re-routing a section is
        not."""
        from rheplicant.config.preflight import _NOT_YET

        with pytest.raises(ConfigError, match="command line") as excinfo:
            load_document({**synthetic_document(), section: {}})
        assert _NOT_YET[section] in str(excinfo.value)

    def test_campaign_names_the_deferred_capability(self):
        with pytest.raises(ConfigError, match="capability 4"):
            load_document({**synthetic_document(), "campaign": {}})

    def test_a_variant_changes_the_built_twin(self):
        base = load_document(synthetic_document())
        unity = load_document(synthetic_document(), variant="unity_gain")
        assert float(base.twin["gain"].gain) == pytest.approx(1.1)
        assert float(unity.twin["gain"].gain) == pytest.approx(1.0)

    def test_an_inference_section_builds_on_the_run(self):
        # D-10 (Plan 3C Task 6): this sigma moves WITH synthetic_document()'s
        # model.noise.sigma edit above -- observed: declares no twin: of its
        # own, so it defaults to "full" (sections/observed.py's own default)
        # and the FULL twin, carrying model.noise, is what draws this data.
        # A numeric C18 refuses this document if the two disagree.
        doc = {**synthetic_document(),
               "inference": {
                   "twin": {"without": ["noise"]},
                   "parameters": {"g": {"init": 1.0, "linear": True,
                                        "into": "gain.gain"}},
                   "noise": {"kind": "homoscedastic",
                             "sigma": {"value": 0.05, "unit": "K"}},
                   "observed": {"from": "simulation", "at": {"g": 1.5}},
               }}
        run = load_document(doc)
        assert run.inference.space is not None
        assert "noise" not in run.inference.fit_twin.lit
        assert run.inference.observed.entries["primary"].shape == (16, 8)
        assert float(run.inference.truth["g"]) == pytest.approx(1.5)

    def test_a_variant_cannot_smuggle_a_refused_section_past_the_sweep(self):
        """The sweep runs on the MERGED document: variants apply first, so a
        patch injecting a not-yet-owned section is still refused."""
        doc = synthetic_document()
        doc["variants"]["sneaky"] = {"outputs": {}}
        with pytest.raises(ConfigError, match="command line"):
            load_document(doc, variant="sneaky")


class TestRunForward:
    def test_the_forward_run_produces_the_waterfall(self):
        out = run_forward(synthetic_document())
        assert out.data.shape == (16, 8)
        assert bool(jnp.all(jnp.isfinite(out.data)))

    def test_a_configured_run_is_accepted_directly(self):
        run = load_document(synthetic_document())
        out = run_forward(run)
        assert out.data.shape == (16, 8)

    def test_it_is_reproducible_from_the_seed(self):
        one = run_forward(synthetic_document())
        two = run_forward(synthetic_document())
        assert jnp.allclose(one.data, two.data)


class TestLaterErrorsCarryTheCompletedReport:
    """Plan 4A Task 10 (spec §8): after a boundary completes, a later
    builder/kind-parser ``ConfigError`` carries the accumulated report."""

    def test_a_builder_error_carries_the_completed_passes_report(
            self, monkeypatch):
        """Measured before this task: the raised error's ``report`` is None --
        the warning this document earned is computed and then dropped."""
        marker = warn_finding("C97", "runtime.seed",
                              "a note the document earned.")
        register_text("C97")(lambda document: (marker,))

        def explode(*args, **kwargs):
            raise ConfigError("the beam is gone.")

        monkeypatch.setattr(document_module, "build_resources", explode)
        try:
            with pytest.raises(ConfigError, match="the beam is gone") as got:
                load_document(synthetic_document())
        finally:
            TEXT_CHECKS.pop("C97", None)
        assert got.value.report is not None
        assert marker in got.value.report.findings


class TestThePublicSurfaceIsUnchanged:
    """Step 6's constants, recorded before the orchestration refactor."""

    def test_the_wrapper_signatures(self):
        for fn, names in ((load_document, ("document", "variant", "base_dir")),
                          (run_forward, ("run", "variant", "base_dir"))):
            parameters = list(inspect.signature(fn).parameters.values())
            assert tuple(one.name for one in parameters) == names
            assert parameters[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            assert all(one.kind is inspect.Parameter.KEYWORD_ONLY
                       for one in parameters[1:])
            assert all(one.default is None for one in parameters[1:])
        from rheplicant.config.sections.runs import run_document
        parameters = list(inspect.signature(run_document).parameters.values())
        assert tuple(one.name for one in parameters) == ("document",
                                                         "base_dir")

    def test_the_tuple_shapes(self):
        assert ConfiguredRun._fields == (
            "document", "runtime", "state", "twin", "inference", "resources",
            "context", "report")
        assert ConfiguredRun._field_defaults == {"report": Report()}
        assert RunSpec._fields == ("name", "kind", "variant", "on", "expect",
                                   "options", "reuse")
        assert RunSpec._field_defaults == {"reuse": None}
        assert RunResult._fields == ("name", "kind", "product", "error",
                                     "variant")
        assert RunResult._field_defaults == {"variant": None}


class TestIngestedDocuments:
    def make_document(self, tmp_path):
        pytest.importorskip("h5py", reason="h5py comes with rheplicant[rhino]")
        from tests.config.test_config_section_ingest import make_file

        make_file(tmp_path / "obs.hd5f")
        return {
            "schema_version": 1,
            "runtime": {"seed": 1},
            "observation": {
                "from_file": {"format": "rhino_hdf5", "path": "obs.hd5f",
                              "freq_unit": "MHz", "settle_seconds": 0.0},
                "switching": {"order": ["antenna", "internal_load",
                                        "heated_load"]},
            },
            "model": {"gain": {"gain": {"value": 2.0,
                                        "unit": "dimensionless"}}},
            "runs": [{"kind": "forward"}],
        }

    def test_the_recording_becomes_the_state(self, tmp_path):
        run = load_document(self.make_document(tmp_path),
                            base_dir=str(tmp_path))
        assert run.state.data.shape == (12, 3)
        assert "receiver_input" in run.state.coords.extra
        assert "flags" in run.state.aux
        assert run.state.meta["from_file/sha256"]
        assert float(run.state.coords.time[0]) == 0.0   # relative, not unix

    def test_a_transform_twin_runs_on_the_recording(self, tmp_path):
        out = run_forward(self.make_document(tmp_path),
                          base_dir=str(tmp_path))
        assert float(out.data[0, 0]) == pytest.approx(2.0)   # gain doubled the ones

    def test_a_source_twin_against_recorded_data_is_the_assemblys_refusal(
            self, tmp_path):
        from rheplicant.core.graph import AssemblyError

        doc = self.make_document(tmp_path)
        doc["model"] = {"global_signal": {"depth": {"value": 0.5, "unit": "K"},
                                          "centre": {"value": 75.0,
                                                     "unit": "MHz"},
                                          "width": {"value": 5.0,
                                                    "unit": "MHz"}}}
        run = load_document(doc, base_dir=str(tmp_path))
        with pytest.raises(AssemblyError, match="generates its own data"):
            run_forward(run)
