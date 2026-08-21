"""Form 6: destination-preserving derivations, one per package function."""

import jax.numpy as jnp
import pytest

from rheplicant.config import ConfigError
from rheplicant.config.context import ResolutionContext
from rheplicant.config.derive import _DERIVATIONS, DERIVATIONS
from rheplicant.config.values import resolve_value


@pytest.fixture
def context():
    return ResolutionContext(
        freq=jnp.linspace(60e6, 85e6, 8), time=jnp.arange(0.0, 128.0, 2.0), dtype="float32"
    )


class TestChannelSpacing:
    def test_it_measures_the_grid_rather_than_asking_the_user_to_divide(self, context):
        """CWCalibrationOperator floors line_width at MIN_WIDTH_IN_CHANNELS *
        median(|diff(freq)|) and check A13 forces the key to be written with
        no default. v0 then handed the user the arithmetic -- 25e6/(8-1) =
        806451.6129032258 -- which the code declines to guess. Rounding it to
        0.8 MHz is refused at trace time; rounding the other way mis-sizes the
        protection mask.

        Also catches the swap: reading context.time here gives 2.0.
        """
        got = resolve_value({"from": "channel_spacing"}, context)
        assert float(got.value) == pytest.approx(25e6 / 7)
        assert got.unit.canonical == "Hz"
        # Catches the two entries' units swapped, and catches `source` left as
        # the derivation name rather than the form key -- check A40 reads it,
        # and every refusal downstream quotes it.
        assert got.source == "from"

    def test_the_times_multiplier(self, context):
        """Catches `times:` ignored (1x), applied as a division (0.25x), or
        added rather than multiplied."""
        got = resolve_value({"from": "channel_spacing", "times": 4.0}, context)
        assert float(got.value) == pytest.approx(4.0 * 25e6 / 7)

    def test_the_gap_is_the_median_and_not_the_mean(self):
        """Catches `mean` in place of `median`. The plan's fixtures are a
        linspace and an arange, on which the two differ by one float32 ulp and
        both pass a rel-1e-6 assertion -- so on a uniform grid this mutation
        is invisible. It is visible on the grid the median was chosen for: a
        band with a flagged block dropped out of it. Here the mean of the gaps
        is 5 MHz, which is a spacing no adjacent pair of channels has.
        """
        gappy = ResolutionContext(freq=jnp.asarray([0.0, 1e6, 2e6, 3e6, 20e6]))
        got = resolve_value({"from": "channel_spacing"}, gappy)
        assert float(got.value) == pytest.approx(1e6)

    def test_a_descending_axis_still_has_a_positive_spacing(self):
        """Catches `jnp.diff` without `jnp.abs`. A descending grid is legal
        throughout this package; unsigned, its spacing comes back negative,
        and a negative spacing sits below every floor it is checked against --
        so check A13 reads as comfortably satisfied.
        """
        descending = ResolutionContext(freq=jnp.linspace(85e6, 60e6, 8))
        got = resolve_value({"from": "channel_spacing"}, descending)
        assert float(got.value) == pytest.approx(25e6 / 7)

    def test_an_undeclared_frequency_axis_is_refused_by_name(self):
        """Catches a refusal that names the wrong axis -- the message is the
        only thing that says WHICH grid the document forgot, and there are two.
        """
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"from": "channel_spacing"}, ResolutionContext(freq=None))
        assert "frequency" in str(excinfo.value)

    def test_a_single_sample_axis_has_no_spacing(self):
        """Catches `jnp.median` over an empty array, which returns nan rather
        than raising: nan defeats every comparison-based check downstream, so
        a one-channel grid would resolve to a line_width floor that passes.
        """
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"from": "channel_spacing"}, ResolutionContext(freq=jnp.asarray([60e6])))
        assert "1 sample(s)" in str(excinfo.value)


class TestSampleCadence:
    def test_it_measures_the_time_axis(self, context):
        """Catches the swap: reading context.freq here gives 3.57e6, and the
        fixture's two axes have different lengths and different spacings so
        neither direction of the swap can hide."""
        got = resolve_value({"from": "sample_cadence"}, context)
        assert float(got.value) == pytest.approx(2.0)
        assert got.unit.canonical == "s"

    def test_the_times_multiplier(self, context):
        """Catches `times:` honoured by channel_spacing and dropped here --
        the registry is two entries, and a per-entry mutation needs a
        per-entry test."""
        got = resolve_value({"from": "sample_cadence", "times": 0.5}, context)
        assert float(got.value) == pytest.approx(1.0)

    def test_an_undeclared_time_axis_is_refused_by_name(self):
        """Catches the axis names hard-coded to 'frequency' for both entries."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"from": "sample_cadence"}, ResolutionContext(time=None))
        message = str(excinfo.value)
        assert "time" in message
        assert "frequency" not in message


class TestBasisMatrix:
    def test_n_comes_from_the_grid_and_is_never_written(self, context):
        """radio/t_sys.py: 'a basis built for another band would return a
        smooth, plausible, wrong temperature'. Taking n from the grid makes
        that structurally impossible, which is why a file route is refused."""
        got = resolve_value(
            {"from": "basis_matrix", "kind": "legendre", "n_basis": 3, "axis": "freq"}, context
        )
        assert got.value.shape == (8, 3)
        assert got.unit.canonical == "dimensionless"

    def test_axis_time_takes_n_from_the_time_grid(self, context):
        """Catches `axis:` read but ignored, with context.freq taken always.
        The freq-only test above passes under that mutation; this one does not,
        because the fixture's grids are 8 and 64 samples long."""
        got = resolve_value(
            {"from": "basis_matrix", "kind": "legendre", "n_basis": 3, "axis": "time"}, context
        )
        assert got.value.shape == (64, 3)

    def test_an_unknown_kind_is_refused_by_the_package_s_own_guard(self, context):
        """Catches a wrapper that restates core/basis.py:206 instead of
        quoting it -- a second copy of the alphabet in this layer goes stale
        the first time BASIS_KINDS gains an entry."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value(
                {"from": "basis_matrix", "kind": "chebyshev", "n_basis": 3, "axis": "freq"}, context
            )
        message = str(excinfo.value)
        for kind in ("legendre", "polynomial", "fourier"):
            assert kind in message, kind

    def test_more_functions_than_samples_is_refused_as_a_document_error(self, context):
        """Catches the package guard being let through unwrapped. n_basis: 16
        against an 8-channel grid is a document error however deep in core it
        is noticed, and a loader that catches ConfigError to say 'this document
        is wrong' would otherwise crash with a StateValidationError that reads
        as a package bug. The quoted message has to survive the wrapping."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value(
                {"from": "basis_matrix", "kind": "legendre", "n_basis": 16, "axis": "freq"}, context
            )
        message = str(excinfo.value)
        assert "null space" in message
        assert "n is 8" in message

    def test_writing_n_is_refused_and_the_reason_is_named(self, context):
        """Catches the `'n' in node` refusal removed -- a written n is then
        simply ignored, and the document says one thing while the run does
        another. Catches, too, `n` being registered as an ordinary argument:
        the generic 'does not take' refusal would fire first and name no
        reason."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value(
                {"from": "basis_matrix", "kind": "legendre", "n_basis": 3, "n": 16, "axis": "freq"},
                context,
            )
        message = str(excinfo.value)
        assert "'n'" in message or '"n"' in message
        assert "grid" in message

    def test_n_is_not_advertised_as_an_argument(self, context):
        """The other half of the pair: routing `n` to the function must not
        put it in the list of arguments the derivation takes, which is the
        list a reader copies from."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"from": "basis_matrix", "nonsense": 1}, context)
        assert "its arguments are ['axis', 'kind', 'n_basis']" in str(excinfo.value)

    def test_an_axis_outside_the_run_is_refused(self, context):
        """Catches `.get(axis)` defaulting to a grid rather than to None."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value(
                {"from": "basis_matrix", "kind": "legendre", "n_basis": 3, "axis": "none"}, context
            )
        assert "supplies n" in str(excinfo.value)

    @pytest.mark.parametrize("missing", ["kind", "n_basis"])
    def test_a_missing_required_argument_is_refused_in_this_layer(self, context, missing):
        """Catches the required-key check removed. The key then reaches
        `node[...]` inside the package guard, whose wrapping turns the
        resulting KeyError into a ConfigError that happens to quote the same
        key -- so an assertion on the key alone passes under the mutation
        (measured: M21 survived it). The message is the whole point of the
        check: 'basis_matrix: kind is required' names the document, where
        'KeyError' names Python and blames the function."""
        node = {"from": "basis_matrix", "kind": "legendre", "n_basis": 3, "axis": "freq"}
        del node[missing]
        with pytest.raises(ConfigError) as excinfo:
            resolve_value(node, context)
        assert str(excinfo.value) == f"basis_matrix: {missing!r} is required."


class TestUnitMeanFree:
    def test_it_calls_the_package_function(self, context):
        """receiver.py:103 returns (bandpass / mean(bandpass))[:-1] -- one
        element SHORTER, because the dropped element is the constraint that
        pins the mean at one. Catches the derivation returning its input
        unchanged (which gives [1, 2, 3]), normalising without dropping (which
        gives three elements), or dividing by something other than the mean.
        """
        got = resolve_value(
            {
                "from": "unit_mean_free",
                "bandpass": {"list": [1.0, 2.0, 3.0], "unit": "dimensionless"},
            },
            context,
        )
        assert got.value.shape == (2,)
        assert got.value == pytest.approx(jnp.asarray([0.5, 1.0]))
        assert got.unit.canonical == "dimensionless"

    def test_a_bandpass_that_is_not_one_dimensional_is_refused(self, context):
        """Catches the package guard being let through unwrapped, and catches
        the bandpass node resolved but not passed on."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value(
                {"from": "unit_mean_free", "bandpass": {"zeros": [2, 2]}},
                context,
            )
        message = str(excinfo.value)
        assert "ndim=2" in message
        assert "(2, 2)" in message

    def test_the_bandpass_is_required(self, context):
        """Catches `node.get("bandpass")`, which would hand None to
        jnp.asarray and fail somewhere with no mention of the key."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"from": "unit_mean_free"}, context)
        assert "bandpass" in str(excinfo.value)


class TestTheRegistry:
    def test_an_unknown_derivation_is_refused_and_the_registered_ones_are_listed(self, context):
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"from": "beam_solid_angle"}, context)
        message = str(excinfo.value)
        assert "beam_solid_angle" in message
        for name in DERIVATIONS:
            assert name in message, name

    def test_the_listing_is_read_live_rather_than_snapshot(self, context):
        """Catches a module-level `_KNOWN = sorted(_DERIVATIONS)` computed
        beside the refusal instead of inside it. Every derivation in this
        module registers at import, so a snapshot is indistinguishable from
        the live table today -- and goes short the moment Plans 1B and 2
        register into it, at which point the message names a set the loader
        does not have."""
        _DERIVATIONS["probe_only"] = (None, frozenset(), frozenset())
        try:
            with pytest.raises(ConfigError) as excinfo:
                resolve_value({"from": "beam_solid_angle"}, context)
            assert "probe_only" in str(excinfo.value)
        finally:
            del _DERIVATIONS["probe_only"]

    def test_an_unknown_argument_is_refused(self, context):
        """Catches _from's unknown-argument check removed: `factor:` is then
        silently dropped and the run differs from the document by exactly the
        thing that key was there to say."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"from": "channel_spacing", "factor": 2.0}, context)
        assert "factor" in str(excinfo.value)
        assert "times" in str(excinfo.value)

    def test_a_modifier_is_legitimate_beside_a_derivation(self, context):
        """The complement: catches _from subtracting only its own arguments
        and refusing every modifier, which would make scale:, unit: and as:
        unwritable on form 6 alone."""
        got = resolve_value({"from": "channel_spacing", "scale": 2.0}, context)
        assert float(got.value) == pytest.approx(2.0 * 25e6 / 7)
