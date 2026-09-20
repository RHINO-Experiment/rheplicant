"""Pre-flight: ``engine: log_conjugate`` over a noise that has no log route.

A block declared ``engine: log_conjugate`` is solved against the log of the
data, and ``SamplingPlan`` gets there through
``inference.loglinear.to_log_space``, which refuses every noise that
``log_route_refusal`` names: an additive noise (``noise_additive``) and, since
T-002 G4 (branch ``t002-g4-numerics``, A5-3), a ``RadiometerNoise`` with a
declared floor (``noise_neither``). Measured before this check: such a
document passed ``rheplicant validate`` with exit 0 and ``rheplicant run``
failed with exit 1 and a traceback, because pre-flight never read the noise
for a log route.

The config layer may not import ``rheplicant.inference``, so the predicate is
restated over the noise TEXT (``fitting._log_route_refusal_text``). What keeps
the restatement honest is :class:`TestTheTextVerdictIsLogRouteRefusals`, which
builds the package's own noise model for each row of a table and compares the
two verdicts.

**Which verdicts come from G4.** On a branch without G4's change,
``log_route_refusal`` returns ``None`` for a floored ``RadiometerNoise``; the
rows marked ``G4`` below are ``xfail(strict=True)`` there, detected by
``"noise_neither" not in LOG_ROUTE_REFUSALS``, and run as ordinary
assertions once G4 is merged.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import yaml

from tests.config.preflight_helpers import preflight_document
from tests.config.test_config_exits_plan import _radiometer, document

#: A ``RadiometerNoise`` the log route accepts before any floor is added:
#: ``f = 1 / sqrt(3.5714286 MHz * 2 s) = 3.7e-4``, far under the 0.06 limit.
RADIOMETER = _radiometer(True)
WIDTH_HZ, TAU_S = 3.5714286e6, 2.0

#: A non-linear latent: ``engine: log_conjugate`` on a ``linear: true`` one is
#: a different refusal (``plan.py``'s own), and not this file's subject.
NONLINEAR = {
    "w": {
        "init": 5.0,
        "into": "global_signal.width",
        "prior": {"normal": {"loc": 5.0, "scale": 1.0}},
    }
}


def _before_g4() -> bool:
    """True on a branch without G4's floor refusal in the package."""
    from rheplicant.inference.loglinear import LOG_ROUTE_REFUSALS

    return "noise_neither" not in LOG_ROUTE_REFUSALS


#: The mark for a verdict G4's change decides. Strict, so the day G4 is
#: merged and the marker is still active (which it cannot be: the condition
#: reads the package) an unexpected pass is loud.
G4 = pytest.mark.xfail(
    _before_g4(),
    strict=True,
    reason="log_route_refusal refuses a floor > 0 only after T-002 G4 "
    "(t002-g4-numerics, A5-3) is merged",
)


def _document(noise, *, engine="log_conjugate", warm=False):
    run = {"name": "fit", "kind": "plan.estimate", "blocks": [{"names": ["w"], "engine": engine}]}
    if warm:
        run = {
            "name": "fit",
            "kind": "plan.sample",
            "n_sweeps": 12,
            "blocks": [{"names": ["w"]}],
            "warm_start": {
                "kind": "plan.estimate",
                "move": ["w"],
                "blocks": [{"names": ["w"], "engine": engine}],
            },
        }
    return preflight_document(inference={"parameters": NONLINEAR, "noise": noise}, runs=[run])


def _found(document):
    from rheplicant.config.preflight.fitting import _blocks

    return list(_blocks(document))


def _floored(value, unit="K"):
    return {**RADIOMETER, "floor": {"value": value, "unit": unit}}


class TestTheCheck:
    def test_a_floored_radiometer_refuses_a_log_conjugate_block(self):
        found = _found(_document(_floored(1.0)))
        assert len(found) == 1, found
        finding = found[0]
        assert finding.check == "A19"
        assert finding.where == "runs[0].blocks[0]"
        message = finding.message
        assert message.startswith("runs['fit']: blocks[0] asks for engine: log_conjugate")
        assert "inference.noise.floor declares 1.0 K" in message
        assert "noise_neither" in message
        assert message.endswith("(check A19).")

    @pytest.mark.parametrize(
        "noise, kind",
        [
            ({"kind": "homoscedastic", "sigma": {"value": 0.05, "unit": "K"}}, "homoscedastic"),
            (
                {
                    "kind": "radiometer_frozen",
                    "source": "observed",
                    "channel_width": RADIOMETER["channel_width"],
                    "integration_time": RADIOMETER["integration_time"],
                },
                "radiometer_frozen",
            ),
        ],
    )
    def test_an_additive_noise_refuses_a_log_conjugate_block(self, noise, kind):
        found = _found(_document(noise))
        assert [one.check for one in found] == ["A19"], found
        assert f"inference.noise is kind: {kind}" in found[0].message
        assert "noise_additive" in found[0].message

    @pytest.mark.parametrize(
        "noise",
        [
            RADIOMETER,
            _floored(0),
            _floored(0.0, "mK"),
            # -273.15 celsius is 0 K: the unit is APPLIED, affine offset included
            _floored(-273.15, "celsius"),
        ],
        ids=["no-floor", "zero-K", "zero-mK", "zero-kelvin-in-celsius"],
    )
    def test_a_radiometer_with_no_floor_is_accepted(self, noise):
        # The anti-vacuity half: a check that refused every log_conjugate
        # block would pass every test above.
        assert _found(_document(noise)) == []

    def test_the_unit_is_applied_before_the_sign_is_read(self):
        # 0 celsius is 273.15 K. Reading the number and dropping the unit
        # would accept a floor the package refuses.
        found = _found(_document(_floored(0.0, "celsius")))
        assert [one.where for one in found] == ["runs[0].blocks[0]"]

    @pytest.mark.parametrize("engine", ["gradient", "conjugate"])
    def test_only_a_log_conjugate_block_is_asked(self, engine):
        # A floored or additive noise is legal for every other engine.
        document = _document(_floored(1.0), engine=engine)
        if engine == "conjugate":
            document["inference"]["parameters"]["w"]["linear"] = True
        assert _found(document) == []

    def test_the_warm_start_block_is_asked_too(self):
        # The warm estimate is a SamplingPlan of its own over the same noise.
        found = _found(_document(_floored(1.0), warm=True))
        assert [one.where for one in found] == ["runs[0].warm_start.blocks[0]"]

    def test_it_speaks_even_when_the_partition_is_wrong(self):
        # It reads the block's engine and the noise, no latent, so a document
        # with both faults hears about both in one round trip, as it does for
        # the engine enum. Kills gating it on the partition.
        document = _document(_floored(1.0))
        document["runs"][0]["blocks"][0]["names"] = ["w", "ghost"]
        checks = sorted(one.check for one in _found(document))
        assert "A19" in checks and "A16" in checks, checks

    @pytest.mark.parametrize(
        "floor",
        [
            {"ref": "resources.arrays.floor"},
            {"value": "one", "unit": "K"},
            {"value": True, "unit": "K"},
        ],
        ids=["ref", "string", "bool"],
    )
    def test_a_floor_the_text_cannot_read_stands_down(self, floor):
        # "Cannot tell" is not a verdict; build_noise and the package still
        # decide these at P2/P3.
        assert _found(_document({**RADIOMETER, "floor": floor})) == []

    @pytest.mark.parametrize("noise", [None, {"kind": "none"}, "nope"])
    def test_a_missing_or_malformed_noise_stands_down(self, noise):
        # Other checks and build_noise own these; a second voice would give
        # one fault two refusals.
        document = _document(RADIOMETER)
        if noise is None:
            document["inference"].pop("noise")
        else:
            document["inference"]["noise"] = noise
        assert _found(document) == []


class TestTheCommandLine:
    """``rheplicant validate`` exits 2 on the document that used to exit 0."""

    @staticmethod
    def _write(tmp_path, noise):
        doc = document(
            {
                "kind": "plan.estimate",
                "blocks": [{"names": ["g"], "engine": "log_conjugate"}],
                "check_identifiability": False,
            }
        )
        doc["inference"]["parameters"]["g"].pop("linear")
        doc["inference"]["noise"] = noise
        path = tmp_path / "doc.yaml"
        path.write_text(yaml.safe_dump(doc))
        return str(path)

    def test_validate_refuses_a_floored_log_conjugate_block(self, tmp_path, capsys):
        from _rheplicant_bootstrap.cli import main

        assert main(["validate", self._write(tmp_path, _floored(1.0))]) == 2
        said = capsys.readouterr()
        assert "noise_neither" in said.out + said.err
        assert "check A19" in said.out + said.err

    def test_validate_refuses_an_additive_noise_under_log_conjugate(self, tmp_path):
        from _rheplicant_bootstrap.cli import main

        noise = {"kind": "homoscedastic", "sigma": {"value": 0.05, "unit": "K"}}
        assert main(["validate", self._write(tmp_path, noise)]) == 2

    def test_validate_accepts_the_same_document_without_the_floor(self, tmp_path):
        from _rheplicant_bootstrap.cli import main

        assert main(["validate", self._write(tmp_path, RADIOMETER)]) == 0


def _package_radiometer(floor=0.0):
    from rheplicant.inference.noise import RadiometerNoise

    return RadiometerNoise(WIDTH_HZ, TAU_S, floor)


def _package_flagged(floor=0.0):
    import jax.numpy as jnp

    from rheplicant.inference.noise import FlaggedNoise

    return FlaggedNoise(_package_radiometer(floor), jnp.zeros((2, 2), bool))


def _package_homoscedastic():
    from rheplicant.inference.noise import HomoscedasticNoise

    return HomoscedasticNoise(0.05)


def _package_decided_sigma():
    # radiometer_frozen hands the plan a DECIDED sigma array, and a plan wraps
    # a bare sigma as HomoscedasticNoise (``uncertainty.as_noise_model``).
    import jax.numpy as jnp

    from rheplicant.inference.uncertainty import as_noise_model

    return as_noise_model(jnp.full((2, 2), 0.05))


def _package_nan_floor():
    # RadiometerNoise refuses a NaN floor at construction, so the package's
    # PREDICATE is asked with the two attributes it reads. G4 writes the
    # comparison `not floor <= 0` so that a NaN is refused and not routed.
    return SimpleNamespace(fractional=1.0 / math.sqrt(WIDTH_HZ * TAU_S), floor=float("nan"))


#: (config text, the package's model for the same declaration). The
#: ``G4``-marked rows are the ones G4's change decides.
_TABLE = [
    pytest.param(RADIOMETER, _package_radiometer, id="fractional-only-no-floor"),
    pytest.param(_floored(0.0), lambda: _package_radiometer(0.0), id="floor-0"),
    pytest.param(_floored(1.0), lambda: _package_radiometer(1.0), id="floor-positive", marks=G4),
    pytest.param(_floored(float("nan")), _package_nan_floor, id="floor-nan", marks=G4),
    pytest.param(
        {"kind": "homoscedastic", "sigma": {"value": 0.05, "unit": "K"}},
        _package_homoscedastic,
        id="additive-only",
    ),
    pytest.param(
        {
            "kind": "radiometer_frozen",
            "source": "observed",
            "channel_width": RADIOMETER["channel_width"],
            "integration_time": RADIOMETER["integration_time"],
        },
        _package_decided_sigma,
        id="additive-decided-sigma",
    ),
    pytest.param(
        {**RADIOMETER, "flags": {"from": "observation"}}, _package_flagged, id="flagged-fractional"
    ),
    pytest.param(
        {**_floored(1.0), "flags": {"from": "observation"}},
        lambda: _package_flagged(1.0),
        id="both-fractional-and-floor",
        marks=G4,
    ),
]


class TestTheTextVerdictIsLogRouteRefusals:
    @pytest.mark.parametrize("text, model", _TABLE)
    def test_the_two_verdicts_agree(self, text, model):
        from rheplicant.config.preflight.fitting import _log_route_refusal_text
        from rheplicant.inference.loglinear import log_route_refusal

        assert _log_route_refusal_text(text) == log_route_refusal(model())

    @pytest.mark.parametrize(
        "reason",
        [
            "noise_additive",
            pytest.param("noise_neither", marks=G4),
        ],
    )
    def test_every_word_the_text_can_say_is_a_package_reason(self, reason):
        from rheplicant.config.preflight.fitting import _LOG_ROUTE_REASONS
        from rheplicant.inference.loglinear import LOG_ROUTE_REFUSALS

        assert reason in _LOG_ROUTE_REASONS
        assert reason in LOG_ROUTE_REFUSALS

    def test_the_text_cannot_reach_fractional_too_large(self):
        """The one package verdict this check does not restate, recorded.

        ``f = 1 / sqrt(channel_width * integration_time)`` needs both values
        resolved, and ``{from: observation}`` (their default) reads the
        observation's own grid: not text. A document whose ``f`` is above
        0.06 still reaches ``to_log_space`` and is refused there at P3.
        """
        from rheplicant.config.preflight.fitting import _log_route_refusal_text
        from rheplicant.inference.loglinear import log_route_refusal
        from rheplicant.inference.noise import RadiometerNoise

        text = {
            **RADIOMETER,
            "channel_width": {"value": 100.0, "unit": "Hz"},
            "integration_time": {"value": 1.0, "unit": "s"},
        }
        assert _log_route_refusal_text(text) is None
        assert log_route_refusal(RadiometerNoise(100.0, 1.0)) == "fractional_too_large"
