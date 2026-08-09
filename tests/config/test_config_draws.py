"""Form 3: a drawn value, and the seed that makes it reproducible."""

import jax
import jax.numpy as jnp
import pytest

from rheplicant.config import ConfigError
from rheplicant.config.context import ResolutionContext
from rheplicant.config.draws import seed_for
from rheplicant.config.values import resolve_value


@pytest.fixture
def context():
    return ResolutionContext(
        freq=jnp.linspace(60e6, 85e6, 8),
        time=jnp.arange(64.0),
        dtype="float32",
        seed=0,
        seeds={"sky_structure": 7},
    )


class TestTheDraw:
    def test_normal_takes_a_shape_a_loc_and_a_scale(self, context):
        got = resolve_value(
            {
                "normal": {
                    "shape": ["n_freq"],
                    "loc": 100.0,
                    "scale": 20.0,
                    "seed": {"from": "runtime.seeds.sky_structure"},
                },
                "unit": "K",
            },
            context,
        )
        assert got.value.shape == (8,)
        assert got.source == "normal"

    def test_uniform(self, context):
        got = resolve_value(
            {
                "uniform": {
                    "shape": ["n_time"],
                    "low": 0.0,
                    "high": 1.0,
                    "seed": {"from": "runtime.seeds.sky_structure"},
                }
            },
            context,
        )
        assert got.value.shape == (64,)
        assert float(got.value.min()) >= 0.0 and float(got.value.max()) <= 1.0
        # Not decoration: `source` is what check A40 reads and what every
        # refusal downstream quotes, and a _draw that labels both forms with
        # the one its first test happened to use is invisible to everything
        # else here -- the numbers are right, only the account of them is not.
        assert got.source == "uniform"

    def test_a_two_dimensional_shape_keeps_its_order(self, context):
        """Catches a shape tuple built in the wrong order. (n_time, n_freq)
        and (n_freq, n_time) are both plausible TOD shapes, every one-symbol
        shape in this file is blind to the difference, and on a square grid
        nothing downstream would raise either."""
        got = resolve_value(
            {
                "normal": {
                    "shape": ["n_time", "n_freq"],
                    "seed": {"from": "runtime.seeds.sky_structure"},
                }
            },
            context,
        )
        assert got.value.shape == (64, 8)

    def test_the_same_seed_gives_the_same_draw(self, context):
        node = {
            "normal": {
                "shape": ["n_freq"],
                "loc": 0.0,
                "scale": 1.0,
                "seed": {"from": "runtime.seeds.sky_structure"},
            }
        }
        first = resolve_value(node, context).value
        second = resolve_value(node, context).value
        assert jnp.array_equal(first, second)

    def test_loc_and_scale_are_themselves_value_nodes(self, context):
        got = resolve_value(
            {
                "normal": {
                    "shape": ["n_freq"],
                    "loc": {"value": 100.0, "unit": "K"},
                    "scale": {"value": 20.0, "unit": "K"},
                    "seed": {"from": "runtime.seeds.sky_structure"},
                },
                "unit": "K",
            },
            context,
        )
        assert got.value.shape == (8,)


class TestTheSeedIsNamedNotWritten:
    def test_a_literal_seed_is_refused(self, context):
        """schema 2.1.4: seed: must NAME an entry of runtime.seeds, so every
        realisation in the run is enumerated in one place and lands in
        provenance.json. A literal here is a realisation nothing records."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value(
                {"normal": {"shape": ["n_freq"], "loc": 0.0, "scale": 1.0, "seed": 7}}, context
            )
        message = str(excinfo.value)
        assert "runtime.seeds" in message
        assert "provenance" in message

    def test_a_missing_seed_is_refused(self, context):
        with pytest.raises(ConfigError, match="seed"):
            resolve_value({"normal": {"shape": ["n_freq"], "loc": 0.0, "scale": 1.0}}, context)

    def test_a_named_seed_that_runtime_does_not_declare_is_derived_reproducibly(self, context):
        """schema 4.0 replaces v0's fold_in(key, hash(name)) -- Python's string
        hash is salted per process, so v0 was not reproducible across runs."""
        first = seed_for("jitter", context)
        second = seed_for("jitter", context)
        assert first == second
        assert first != seed_for("sky_structure", context)

    def test_the_derivation_does_not_use_pythons_salted_hash(self):
        """Run in a fresh interpreter with a different PYTHONHASHSEED: a
        salted hash would give a different answer."""
        import os
        import subprocess
        import sys

        script = (
            "from rheplicant.config.context import ResolutionContext;"
            "from rheplicant.config.draws import seed_for;"
            "print(seed_for('jitter', ResolutionContext(seed=0)))"
        )
        # PYTHONPATH is forwarded, not scrubbed with the rest of the
        # environment. The child has to import the SAME rheplicant the parent
        # did: with it dropped, a checkout under PYTHONPATH is invisible to the
        # child, which then imports whatever is installed in site-packages and
        # reports on a different copy of this function than the one under test
        # -- and reports PASS, because the installed copy is the correct one.
        # Measured: with the scrub, this test cannot see a hash() regression
        # introduced in the working tree.
        base = {"PATH": "/usr/bin:/bin"}
        if os.environ.get("PYTHONPATH"):
            base["PYTHONPATH"] = os.environ["PYTHONPATH"]
        outputs = {
            subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                check=True,
                env={**base, "PYTHONHASHSEED": str(salt)},
            ).stdout.strip()
            for salt in (0, 1, 12345)
        }
        assert len(outputs) == 1, outputs

    def test_a_run_with_no_seed_refuses_a_draw_and_names_the_key(self, context):
        import dataclasses

        seedless = dataclasses.replace(context, seed=None, seeds={})
        with pytest.raises(ConfigError) as excinfo:
            resolve_value(
                {
                    "normal": {
                        "shape": ["n_freq"],
                        "loc": 0.0,
                        "scale": 1.0,
                        "seed": {"from": "runtime.seeds.sky_structure"},
                    }
                },
                seedless,
            )
        message = str(excinfo.value)
        assert "runtime.seed" in message
        assert "null" in message  # the legal state it is being distinguished from

    def test_a_seed_from_outside_runtime_seeds_is_refused(self, context):
        """Catches a _seed_name that takes the {from: ...} shape as sufficient
        and never checks the namespace. runtime.seeds being the ONLY namespace
        is the whole mechanism: {from: observation.freq.grid} would otherwise
        name a seed nothing enumerates, which is a literal with extra steps."""
        with pytest.raises(ConfigError, match="runtime.seeds"):
            resolve_value(
                {
                    "normal": {
                        "shape": ["n_freq"],
                        "seed": {"from": "observation.freq.grid"},
                    }
                },
                context,
            )

    def test_a_seed_that_names_nothing_under_the_prefix_is_refused(self, context):
        """Catches dropping the empty-name half of the prefix guard. A bare
        'runtime.seeds.' passes startswith() and would resolve to the seed
        named '' -- a real, derivable, entirely unnameable entry."""
        with pytest.raises(ConfigError, match="runtime.seeds"):
            resolve_value(
                {"normal": {"shape": ["n_freq"], "seed": {"from": "runtime.seeds."}}}, context
            )


class TestTheReportedSeedIsTheSeedThatDrew:
    """The manifest records an integer; these say it is the RIGHT integer.

    Nothing else in this file compares the reported seed against the numbers
    that came out, so without these a derivation that is stable, unique per
    name and unrelated to :func:`seed_for` passes everything -- and the failure
    is a ``provenance.json`` that names a seed which reproduces a different
    sky. A different sky is still a plausible sky.
    """

    def _uniform(self, context, name, **bounds):
        return resolve_value(
            {
                "uniform": {
                    "shape": ["n_time"],
                    "seed": {"from": f"runtime.seeds.{name}"},
                    **bounds,
                }
            },
            context,
        ).value

    def test_a_declared_seed_draws_what_that_seed_draws(self, context):
        """Catches _key reading context.seeds by a different route than
        seed_for -- or not reading it at all, so the run reports 7 and draws
        from the root seed."""
        got = self._uniform(context, "sky_structure", low=0.0, high=1.0)
        key = jax.random.key(seed_for("sky_structure", context))
        assert jnp.array_equal(got, jax.random.uniform(key, (64,), dtype=context.dtype))

    def test_an_undeclared_name_draws_what_its_derived_seed_draws(self, context):
        """Catches the derived branch deriving its key independently of the
        number it reports -- fold_in(key(root), digest) is the natural way to
        write it and is exactly this bug: key(digest ^ root) is a different
        key, so the manifest's integer does not reproduce the array."""
        got = self._uniform(context, "jitter", low=0.0, high=1.0)
        key = jax.random.key(seed_for("jitter", context))
        assert jnp.array_equal(got, jax.random.uniform(key, (64,), dtype=context.dtype))

    def test_two_undeclared_names_do_not_draw_the_same_array(self, context):
        """Catches a _key that folds the root seed in and drops the name --
        jax.random.key(context.seed) -- under which every unnamed seed in a run
        is the same realisation and 'independent jitter' is one array."""
        assert not jnp.array_equal(
            self._uniform(context, "jitter", low=0.0, high=1.0),
            self._uniform(context, "gain_ripple", low=0.0, high=1.0),
        )


class TestTheParametersLandTheRightWayRound:
    def test_loc_is_the_centre_and_scale_is_the_width(self, context):
        """Catches loc and scale swapped in the normal construction. The two
        orders draw from N(100, 20) and N(20, 100): both finite, both the right
        shape, and the second is a 20 K sky with 100 K structure on it."""
        got = resolve_value(
            {
                "normal": {
                    "shape": ["n_time"],
                    "loc": 100.0,
                    "scale": 1.0,
                    "seed": {"from": "runtime.seeds.sky_structure"},
                }
            },
            context,
        ).value
        assert 99.0 < float(jnp.mean(got)) < 101.0
        assert float(jnp.std(got)) < 5.0

    def test_low_is_the_floor_and_high_is_the_ceiling(self, context):
        """Catches low and high swapped in the uniform construction. No range
        assertion can: minval + u*(maxval - minval) with the two exchanged is
        high - u*(high - low), which lands in [low, high] just the same and is
        the same distribution. Only u itself tells them apart, so this pins the
        draw against jax's own [0, 1) on the reported seed."""
        got = resolve_value(
            {
                "uniform": {
                    "shape": ["n_time"],
                    "low": 10.0,
                    "high": 20.0,
                    "seed": {"from": "runtime.seeds.sky_structure"},
                }
            },
            context,
        ).value
        key = jax.random.key(seed_for("sky_structure", context))
        unscaled = jax.random.uniform(key, (64,), dtype=context.dtype)
        assert jnp.allclose(got, 10.0 + 10.0 * unscaled, atol=1e-5)

    def test_loc_defaults_to_zero_and_scale_to_one(self, context):
        """Catches _resolve_operand's default returning 1.0 for loc or 0.0 for
        scale. Both are silent: a defaulted scale of 0 is a constant array of
        the right shape, and a defaulted loc of 1 is an offset sky."""
        seed = {"from": "runtime.seeds.sky_structure"}
        defaulted = resolve_value({"normal": {"shape": ["n_freq"], "seed": seed}}, context).value
        written = resolve_value(
            {"normal": {"shape": ["n_freq"], "loc": 0.0, "scale": 1.0, "seed": seed}}, context
        ).value
        assert jnp.array_equal(defaulted, written)


class TestTheSpecIsChecked:
    def test_an_unknown_key_is_refused(self, context):
        """Catches dropping the unknown-key check. 'stddev' for 'scale' is the
        obvious slip, and silently ignored it leaves scale at its default 1 --
        a draw of the declared shape with none of the declared width."""
        with pytest.raises(ConfigError, match="stddev"):
            resolve_value(
                {
                    "normal": {
                        "shape": ["n_freq"],
                        "stddev": 20.0,
                        "seed": {"from": "runtime.seeds.sky_structure"},
                    }
                },
                context,
            )

    def test_a_shape_is_required(self, context):
        """Catches a defaulted shape, which would draw a scalar and broadcast
        against anything it is later multiplied by."""
        with pytest.raises(ConfigError, match="shape"):
            resolve_value({"normal": {"seed": {"from": "runtime.seeds.sky_structure"}}}, context)

    def test_a_spec_that_is_not_a_mapping_is_refused(self, context):
        with pytest.raises(ConfigError, match="mapping"):
            resolve_value({"normal": ["n_freq"]}, context)

    def test_the_unit_converts_the_drawn_array(self, context):
        """Catches a _draw that records the unit and does not apply it --
        modifiers carries 'unit' either way and apply_modifiers deliberately
        passes it over, so nothing downstream would scale it later."""
        node = {
            "uniform": {
                "shape": ["n_time"],
                "low": 0.0,
                "high": 1.0,
                "seed": {"from": "runtime.seeds.sky_structure"},
            }
        }
        plain = resolve_value(node, context).value
        got = resolve_value({**node, "unit": "ms"}, context)
        assert got.unit.canonical == "s"
        assert jnp.allclose(got.value, plain * 1e-3, atol=1e-9)
