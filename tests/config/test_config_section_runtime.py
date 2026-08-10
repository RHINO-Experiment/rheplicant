"""runtime: -> facts. Recorded and checked here; applied by Plan 4's CLI."""

import jax
import pytest

from rheplicant.config import ConfigError
from rheplicant.config.sections.runtime import RuntimeFacts, build_runtime, state_key


class TestBuildRuntime:
    def test_defaults(self):
        facts = build_runtime({})
        assert facts == RuntimeFacts(
            jax_enable_x64=False, platform="auto", seed=None, seeds={}
        )
        assert facts.dtype == "float32"

    def test_the_full_section(self):
        facts = build_runtime(
            {"platform": "cpu", "seed": 20260806, "seeds": {"sample": 11, "gcr": 7}}
        )
        assert facts.platform == "cpu"
        assert facts.seed == 20260806
        assert facts.seeds == {"sample": 11, "gcr": 7}

    def test_an_unknown_key_is_refused(self):
        with pytest.raises(ConfigError, match=r"does not take \['sede'\]"):
            build_runtime({"sede": 1})

    def test_x64_required_by_is_emitted_not_written(self):
        with pytest.raises(ConfigError, match="emitted"):
            build_runtime({"x64_required_by": ["projector"]})

    def test_a_non_mapping_section_is_refused(self):
        with pytest.raises(ConfigError, match="mapping"):
            build_runtime([1])

    def test_platform_is_a_closed_table(self):
        with pytest.raises(ConfigError, match="platform"):
            build_runtime({"platform": "quantum"})

    def test_a_bool_seed_is_refused(self):
        with pytest.raises(ConfigError, match="seed"):
            build_runtime({"seed": True})

    def test_a_null_seed_is_legal_and_recorded(self):
        assert build_runtime({"seed": None}).seed is None

    def test_seeds_values_must_be_ints(self):
        with pytest.raises(ConfigError, match="seeds"):
            build_runtime({"seeds": {"sample": 1.5}})
        with pytest.raises(ConfigError, match="seeds"):
            build_runtime({"seeds": {"sample": True}})
        with pytest.raises(ConfigError, match="seeds"):
            build_runtime({"seeds": "sample"})

    def test_declaring_x64_in_a_float32_process_is_refused_up_front(self):
        """The suite runs float32; the refusal must name the fix and who
        applies it automatically (Plan 4's CLI)."""
        assert not jax.config.jax_enable_x64
        with pytest.raises(ConfigError) as excinfo:
            build_runtime({"jax_enable_x64": True})
        message = str(excinfo.value)
        assert "jax_enable_x64" in message
        assert "before any array exists" in message


class TestRuntimeFacts:
    def test_x64_facts_declare_float64(self):
        """Directly constructed (bypassing the process guard): the one reason
        .dtype exists is that x64 physics means float64 deliveries."""
        facts = RuntimeFacts(jax_enable_x64=True, platform="auto", seed=None,
                             seeds={})
        assert facts.dtype == "float64"


class TestStateKey:
    def test_a_seed_becomes_a_typed_prng_key(self):
        key = state_key(build_runtime({"seed": 3}))
        assert isinstance(key, jax.Array)
        assert jax.numpy.issubdtype(key.dtype, jax.dtypes.prng_key)

    def test_no_seed_means_no_key(self):
        assert state_key(build_runtime({})) is None
