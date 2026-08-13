"""The transform registry, and bindings -> ParameterSpace."""

import sys

import jax
import jax.numpy as jnp
import pytest

from rheplicant.config import ConfigError
from rheplicant.config.sections.parameters import parse_latents
from rheplicant.config.sections.transforms import build_space, parse_transform
from tests.config.inference_helpers import context, twin


def space_for(parameters, bindings=None, joint_prior=None, replaced=(),
              model=None):
    fit = twin(model)
    return build_space(parse_latents(parameters, context()), bindings,
                       joint_prior, fit_twin=fit, replaced=replaced,
                       context=context()), fit


class TestRegistry:
    def test_named_transforms_map_to_the_documented_callables(self):
        exp, fan = parse_transform("exp", context(), where="t")
        assert float(exp(jnp.asarray(0.0))) == pytest.approx(1.0)
        assert fan is None or fan == "broadcast"

    def test_log_and_sum_are_the_documented_callables(self):
        log, _ = parse_transform("log", context(), where="t")
        assert float(log(jnp.exp(jnp.asarray(1.0)))) == pytest.approx(1.0)
        total, _ = parse_transform("sum", context(), where="t")
        assert float(total(jnp.asarray([1.0, 2.0, 3.0]))) == pytest.approx(6.0)

    def test_split_rows_is_the_distribute_transform(self):
        fn, fan = parse_transform("split_rows", context(), where="t")
        produced = fn(jnp.asarray([1.0, 2.0]))
        assert isinstance(produced, tuple) and len(produced) == 2
        assert fan == "distribute"

    def test_unit_mean_bandpass_is_the_packages_own(self):
        from rheplicant.radio.instrument.receiver import unit_mean_bandpass

        fn, _ = parse_transform("unit_mean_bandpass", context(), where="t")
        free = jnp.full((7,), 1.0)
        assert jnp.allclose(fn(free), unit_mean_bandpass(free))

    def test_affine_takes_scale_and_offset(self):
        fn, _ = parse_transform({"affine": {"scale": 2.0, "offset": 1.0}},
                                context(), where="t")
        assert float(fn(jnp.asarray(3.0))) == pytest.approx(7.0)

    def test_affine_defaults_to_unit_scale_and_zero_offset(self):
        fn, _ = parse_transform({"affine": {"offset": 1.0}}, context(),
                                where="t")
        assert float(fn(jnp.asarray(3.0))) == pytest.approx(4.0)
        fn, _ = parse_transform({"affine": {"scale": 2.0}}, context(),
                                where="t")
        assert float(fn(jnp.asarray(3.0))) == pytest.approx(6.0)

    def test_matmul_applies_a_declared_design(self):
        fn, _ = parse_transform(
            {"matmul": {"design": {"ones": ["n_freq", 2]}}},
            context(), where="t")
        out = fn(jnp.asarray([1.0, 2.0]))
        assert out.shape == (8,)
        assert float(out[0]) == pytest.approx(3.0)

    def test_log_link_basis_is_exp_of_a_basis_expansion(self):
        fn, _ = parse_transform(
            {"log_link_basis": {"kind": "legendre", "n_basis": 3}},
            context(), where="t")
        out = fn(jnp.zeros(3))
        assert out.shape == (8,)
        assert jnp.allclose(out, 1.0)

    def test_log_link_basis_axis_time_reads_the_time_grid(self):
        fn, _ = parse_transform(
            {"log_link_basis": {"kind": "legendre", "n_basis": 3,
                                "axis": "time"}},
            context(), where="t")
        assert fn(jnp.zeros(3)).shape == (16,)

    def test_basis_expand_reads_a_declared_basis_resource(self):
        from rheplicant.config.resources import build_resources

        ctx = context()
        built = build_resources(
            {"bases": {"smooth": {"time": {"kind": "legendre", "n_basis": 2},
                                  "freq": {"kind": "legendre",
                                           "n_basis": 3}}}}, ctx)
        ctx = context(resources=dict(built.resources))
        fn, _ = parse_transform(
            {"basis_expand": {"basis": {"ref": "resources.bases.smooth"}}},
            ctx, where="t")
        assert fn(jnp.zeros((2, 3))).shape == (16, 8)

    def test_basis_expand_refuses_a_ref_that_is_not_a_basis(self):
        ctx = context(resources={"resources.bases.flat": jnp.zeros(3)})
        with pytest.raises(ConfigError, match="not SeparableBasis"):
            parse_transform(
                {"basis_expand": {"basis": {"ref": "resources.bases.flat"}}},
                ctx, where="t")

    def test_python_requires_a_declared_fan(self):
        with pytest.raises(ConfigError, match="fan"):
            parse_transform({"python": "jax.numpy:exp"}, context(), where="t")

    def test_an_unknown_transform_is_refused_listing_the_registry(self):
        with pytest.raises(ConfigError, match="sinh"):
            parse_transform("sinh", context(), where="t")


def _ltj():
    return pytest.importorskip("limtod_jax", reason="limTOD[jax] not installed")


class TestBeamAnalysis:
    """``{beam_analysis: {...}}`` -> ``jax.vmap(limtod_jax.map2alm_iter)``.

    The transform exists to make a beam-map gradient the quantity the user
    meant.  A DriftScanProjector's only non-static field is ``beam_alms``
    (radio/sky/driftscan.py:189-203), so without this the only binding a
    document can write is ``into: ...projector.beam_alms``, whose gradient is
    d(chi2)/d(alm) -- finite, correctly shaped, and a different quantity in a
    different basis from the d(chi2)/d(map) the user was computing by hand.

    NSIDE/LMAX are not free: s2fft's healpix transform, under map2alm_iter,
    needs ``lmax >= 2 * nside - 1``.  Swept rather than inferred -- see
    ``TestBeamAnalysisBandLimit``, which re-measures the cliff at four nside
    and pins both of its sides.
    """

    NSIDE = 4
    LMAX = 7
    N_PIX = 12 * 4 ** 2               # 192
    N_ALM = (7 + 1) * (7 + 2) // 2    # 36, the healpy packing

    def _spec(self, **extra):
        return {"beam_analysis": {"nside": self.NSIDE, "lmax": self.LMAX,
                                  **extra}}

    def _maps(self):
        return jax.random.normal(jax.random.key(0), (3, self.N_PIX))

    def test_beam_analysis_is_the_vmapped_true_alm_transform(self):
        ltj = _ltj()
        fn, fan = parse_transform(self._spec(), context(), where="t")
        assert fan == "broadcast"
        maps = self._maps()
        alms = fn(maps)
        assert alms.shape == (3, self.N_ALM)
        assert jnp.iscomplexobj(alms)
        for index in range(3):
            reference = ltj.map2alm_iter(maps[index], nside=self.NSIDE,
                                         lmax=self.LMAX)
            assert jnp.allclose(alms[index], reference, atol=1e-6)

    def test_a_gradient_flows_through_to_the_beam_maps(self):
        """The whole point of the transform, and the only test that sees it.

        Every other assertion here is on a forward value, and a forward
        value survives ``jax.lax.stop_gradient`` around the call completely
        untouched: right shape, right dtype, matching map2alm_iter to 1e-6,
        still per-frequency, still refusing everything it should.  What a
        binding through that transform would hand the user is a silently
        ZERO d(chi2)/d(map) -- finite, correctly shaped, and the exact wrong
        answer this transform exists to prevent, in the one place it exists
        to prevent it.

        Pinned by VALUE rather than by being non-zero: a forward difference
        on a single pixel agrees with the autodiff to 0.35% here (measured,
        float32, eps=1e-2), so a gradient that is merely finite and
        differently scaled fails too.
        """
        _ltj()
        fn, _ = parse_transform(self._spec(), context(), where="t")
        maps = self._maps()

        def power(sample):
            return jnp.sum(jnp.abs(fn(sample)) ** 2)

        gradient = jax.grad(power)(maps)
        assert gradient.shape == maps.shape == (3, self.N_PIX)
        assert jnp.all(jnp.isfinite(gradient))
        assert jnp.any(gradient != 0.0)
        eps = 1e-2
        moved = maps.at[1, 5].add(eps)
        difference = float((power(moved) - power(maps)) / eps)
        assert float(gradient[1, 5]) == pytest.approx(difference, rel=0.05)

    def test_the_gradient_stays_on_its_own_frequency(self):
        """The vmap is load-bearing in the backward pass as well.

        d(row 1's power)/d(maps) must be exactly zero on rows 0 and 2: the
        frequency axis is a batch axis, not a contraction, so no other
        frequency's map can move this frequency's alms.
        """
        _ltj()
        fn, _ = parse_transform(self._spec(), context(), where="t")
        maps = self._maps()
        gradient = jax.grad(
            lambda sample: jnp.sum(jnp.abs(fn(sample)[1]) ** 2))(maps)
        assert jnp.all(gradient[0] == 0.0)
        assert jnp.all(gradient[2] == 0.0)
        assert jnp.any(gradient[1] != 0.0)

    def test_each_frequency_gets_its_own_transform(self):
        """Each frequency is transformed on its own, as its own map.

        map2alm_iter is a single-map function; the frequency axis is the
        vmap's, exactly as radio/sky/driftscan.py:300-302 writes it.  Scaling
        one row must scale that row's alms and leave the others alone -- an
        implementation that flattens the stack, or that transforms row 0 and
        broadcasts it, disagrees on both counts.

        It does NOT distinguish ``jax.vmap`` from a Python loop over the
        rows: that spelling is numerically identical and still
        differentiable, and only unrolls n_freq times under jit.  Nothing
        here catches that, and asserting the callable is literally a vmap
        would be brittle.
        """
        _ltj()
        fn, _ = parse_transform(self._spec(), context(), where="t")
        maps = self._maps()
        alms = fn(maps)
        assert alms.shape[0] == maps.shape[0] == 3
        assert not jnp.allclose(alms[0], alms[1])
        scaled = fn(maps.at[1].set(2.0 * maps[1]))
        assert jnp.allclose(scaled[0], alms[0], atol=1e-6)
        assert jnp.allclose(scaled[1], 2.0 * alms[1], atol=1e-5)
        assert jnp.allclose(scaled[2], alms[2], atol=1e-6)

    def test_the_transform_is_map2alm_iter_and_not_map2alm_quad(self):
        """The two are not interchangeable: they differ by npix/4pi.

        map2alm_iter returns true (healpy-convention) alms and is what the
        BEAM needs (from_beam_maps, driftscan.py:301); map2alm_quad returns
        quadrature alms and is what the SKY needs (sky_to_alms, :605-606).
        Both have the same shape and the same dtype, so only the numbers can
        tell them apart -- picking the visible one "silently rescales the
        beam by npix/4pi" (driftscan.py:269).  The ratio is pinned against
        npix/4pi itself rather than a wide band, so the assertion identifies
        the factor instead of merely noticing that two arrays differ:
        measured 15.065 here against npix/4pi = 15.279 at nside=4.
        """
        ltj = _ltj()
        fn, _ = parse_transform(self._spec(), context(), where="t")
        maps = self._maps()
        alms = fn(maps)
        quad = ltj.map2alm_quad(maps[0], nside=self.NSIDE, lmax=self.LMAX)
        assert quad.shape == alms[0].shape and quad.dtype == alms[0].dtype
        assert not jnp.allclose(alms[0], quad, atol=1e-4)
        ratio = float(jnp.max(jnp.abs(quad)) / jnp.max(jnp.abs(alms[0])))
        expected = self.N_PIX / (4.0 * float(jnp.pi))
        assert abs(ratio - expected) / expected < 0.05, (ratio, expected)

    def test_a_declared_iterations_reaches_the_call(self):
        """The Jacobi refinement count is visible in the numbers.

        One iteration and the package's three differ by ~4.2% of the largest
        coefficient at this resolution (measured), so a config that sweeps
        the key up and drops it fails here rather than passing on shape alone.
        """
        ltj = _ltj()
        fn, _ = parse_transform(self._spec(iterations=1), context(),
                                where="t")
        maps = self._maps()
        once = fn(maps)[0]
        assert jnp.allclose(
            once,
            ltj.map2alm_iter(maps[0], nside=self.NSIDE, lmax=self.LMAX,
                             iterations=1),
            atol=1e-6)
        assert not jnp.allclose(
            once,
            ltj.map2alm_iter(maps[0], nside=self.NSIDE, lmax=self.LMAX),
            atol=1e-4)

    def test_a_declared_zero_iterations_is_not_read_as_absent(self):
        """0 is a legal refinement count, and falsy.

        A helper that carries the key with ``if body.get(key)`` rather than
        ``if key in body`` drops this one and silently runs the package's
        three; measured on this array, the two differ by 13.0% of the
        largest coefficient (12.1% by norm), so the 1e-4 tolerance below has
        three orders of magnitude of headroom.
        """
        ltj = _ltj()
        fn, _ = parse_transform(self._spec(iterations=0), context(),
                                where="t")
        never = fn(self._maps())[0]
        maps = self._maps()
        assert jnp.allclose(
            never,
            ltj.map2alm_iter(maps[0], nside=self.NSIDE, lmax=self.LMAX,
                             iterations=0),
            atol=1e-6)
        assert not jnp.allclose(
            never,
            ltj.map2alm_iter(maps[0], nside=self.NSIDE, lmax=self.LMAX),
            atol=1e-4)

    def test_an_undeclared_iterations_leaves_the_packages_own(self):
        """Config keys never restate a package default (plan §0)."""
        ltj = _ltj()
        fn, _ = parse_transform(self._spec(), context(), where="t")
        maps = self._maps()
        assert jnp.allclose(
            fn(maps)[0],
            ltj.map2alm_iter(maps[0], nside=self.NSIDE, lmax=self.LMAX),
            atol=1e-6)

    def test_beam_analysis_is_a_mapping_head_the_gate_can_see(self):
        """It must be in _MAPPING, or the head gate swallows it.

        Deleting the Plan-2C placeholder without adding the name to _MAPPING
        leaves ``{beam_analysis: {...}}`` failing the head gate with "a
        mapping transform names exactly one of [...]".  Paired with a stray
        sibling key it must instead take the shared mapping-form path and be
        refused for standing with company -- so the assertion below names the
        head too, which the head gate's own message never does.
        """
        with pytest.raises(ConfigError) as excinfo:
            parse_transform({"beam_analysis": {"nside": self.NSIDE,
                                               "lmax": self.LMAX},
                             "fan": "broadcast"}, context(), where="t")
        message = str(excinfo.value)
        assert "beam_analysis: stands alone" in message
        assert "names exactly one of" not in message

    def test_a_non_mapping_body_is_refused_by_the_shared_preamble(self):
        with pytest.raises(ConfigError, match=r"t\.beam_analysis: is a "
                                              r"mapping"):
            parse_transform({"beam_analysis": 4}, context(), where="t")

    def test_nside_and_lmax_are_required(self):
        with pytest.raises(ConfigError, match="requires nside: and lmax:"):
            parse_transform({"beam_analysis": {"nside": self.NSIDE}},
                            context(), where="t")

    def test_a_missing_nside_is_refused_too(self):
        """The twin of the test above; a guard closed on one key only is the
        recurring shape this suite keeps catching."""
        with pytest.raises(ConfigError, match="requires nside: and lmax:"):
            parse_transform({"beam_analysis": {"lmax": self.LMAX}},
                            context(), where="t")

    def test_a_non_integer_nside_is_refused(self):
        with pytest.raises(ConfigError, match=r"nside: is an integer >= 1"):
            parse_transform({"beam_analysis": {"nside": 4.5,
                                               "lmax": self.LMAX}},
                            context(), where="t")

    def test_a_bool_is_not_an_integer_here(self):
        with pytest.raises(ConfigError, match=r"nside: is an integer >= 1"):
            parse_transform({"beam_analysis": {"nside": True,
                                               "lmax": self.LMAX}},
                            context(), where="t")

    def test_a_zero_nside_is_refused_by_the_grammar(self):
        with pytest.raises(ConfigError, match=r"nside: is an integer >= 1"):
            parse_transform({"beam_analysis": {"nside": 0, "lmax": 7}},
                            context(), where="t")

    def test_a_negative_lmax_is_refused_by_its_own_leg(self):
        """The lmax leg of the compound number check, which nside's tests
        leave untouched -- and it must be lmax: that is named, not nside:."""
        with pytest.raises(ConfigError, match=r"lmax: is an integer >= 0"):
            parse_transform({"beam_analysis": {"nside": 4, "lmax": -1}},
                            context(), where="t")

    def test_a_non_integer_lmax_is_refused(self):
        with pytest.raises(ConfigError, match=r"lmax: is an integer >= 0"):
            parse_transform({"beam_analysis": {"nside": 4, "lmax": 7.0}},
                            context(), where="t")

    def test_a_negative_iterations_is_refused_by_its_own_leg(self):
        """The third leg, which neither of the other two exercises."""
        with pytest.raises(ConfigError,
                           match=r"iterations: is an integer >= 0"):
            parse_transform(self._spec(iterations=-1), context(), where="t")

    def test_a_non_integer_iterations_is_refused(self):
        with pytest.raises(ConfigError,
                           match=r"iterations: is an integer >= 0"):
            parse_transform(self._spec(iterations=1.5), context(), where="t")

    def test_an_unknown_key_inside_beam_analysis_is_refused(self):
        """npol is a real map2alm_iter keyword this transform does not take;
        an unknown-key sweep that only refuses nonsense would miss it.  The
        label is pinned too: ``does not take`` is the shared helper's wording
        and every mapping head raises it."""
        with pytest.raises(ConfigError,
                           match=r"beam_analysis: does not take \['npol'\]"):
            parse_transform(self._spec(npol=2), context(), where="t")

    def test_a_malformed_spec_is_refused_without_reaching_limtod(
            self, monkeypatch):
        """The grammar is checked before ``import limtod_jax``, deliberately.

        The plan's Step 9.4 imports limTOD as the helper's first statement,
        which would make every refusal below raise ``ImportError`` instead
        on an install that lacks it -- and none of this class's refusal
        tests carries an ``importorskip``, so the plan's own suite requires
        the later import.  limTOD is installed here, so only a stubbed-out
        ``sys.modules`` entry can tell the two orderings apart.

        The last clause is what stops this passing vacuously: a well-formed
        spec MUST still reach the import, or the test would also pass on a
        helper that never touched limTOD at all.
        """
        monkeypatch.setitem(sys.modules, "limtod_jax", None)
        with pytest.raises(ConfigError, match="requires nside: and lmax:"):
            parse_transform({"beam_analysis": {"nside": self.NSIDE}},
                            context(), where="t")
        with pytest.raises(ConfigError, match=r"nside: is an integer >= 1"):
            parse_transform({"beam_analysis": {"nside": 4.5, "lmax": 7}},
                            context(), where="t")
        with pytest.raises(ConfigError, match=r"lmax >= 7"):
            parse_transform({"beam_analysis": {"nside": 4, "lmax": 6}},
                            context(), where="t")
        with pytest.raises(ConfigError, match="nside >= 2"):
            parse_transform({"beam_analysis": {"nside": 1, "lmax": 4}},
                            context(), where="t")
        with pytest.raises(ImportError):
            parse_transform(self._spec(), context(), where="t")


class TestBeamAnalysisBandLimit:
    """The band limit is the feature, so it is measured on both sides.

    s2fft's healpix transform, under ``map2alm_iter``, needs
    ``lmax >= 2 * nside - 1``.  Swept, not inferred: the executor re-measured
    the full range at nside 1, 2, 3, 4, 5, 6, 7, 8, 9, 12 and 16 and the
    lower edge is exactly ``2 * nside - 1`` at every one of them from nside=2
    up (powers of two and not), with NO upper edge -- at nside=4 every lmax
    from 7 to 89 returns alms.  An INEQUALITY, so the layer compares with
    ``<`` and not ``!=``.

    Every test here is parametrized over nside, because a boundary test
    written at one nside cannot tell ``lmax < 2 * nside - 1`` from
    ``lmax < nside`` or from a hard-coded ``lmax < 15``.
    """

    NSIDES = (2, 4, 8, 16)

    @staticmethod
    def _spec(nside, lmax):
        return {"beam_analysis": {"nside": nside, "lmax": lmax}}

    @pytest.mark.parametrize("nside", NSIDES)
    def test_the_package_cliff_is_at_two_nside_minus_one(self, nside):
        """Boundary check against the package itself, not against the layer.

        This is the measurement the refusal encodes, kept executable so the
        constant cannot drift away from limTOD underneath it: one below the
        floor the package raises, at the floor it returns the alm vector the
        healpy packing predicts.
        """
        ltj = _ltj()
        floor = 2 * nside - 1
        maps = jax.random.normal(jax.random.key(2), (12 * nside ** 2,))
        with pytest.raises(Exception):  # noqa: B017 - text is not pinned
            ltj.map2alm_iter(maps, nside=nside, lmax=floor - 1)
        at_floor = ltj.map2alm_iter(maps, nside=nside, lmax=floor)
        assert at_floor.shape == ((floor + 1) * (floor + 2) // 2,)

    @pytest.mark.parametrize("nside", NSIDES)
    def test_an_lmax_below_this_nsides_floor_is_refused(self, nside):
        """s2fft's own failure names neither key, and this layer's job is to.

        Below the limit map2alm_iter fails inside s2fft's healpix transform
        with a bare shape error -- ``TypeError: Cannot concatenate arrays
        with shapes that differ...`` just below the floor, ``ValueError: All
        input arrays must have the same shape`` further below -- which
        mentions neither nside: nor lmax: and gives the reader nothing to
        change.  The refusal names both, and the minimum lmax THIS nside
        admits, which is the number the user needs; asserting that computed
        number is what stops a hard-coded floor passing.
        """
        floor = 2 * nside - 1
        with pytest.raises(ConfigError) as excinfo:
            parse_transform(self._spec(nside, floor - 1), context(),
                            where="t")
        message = str(excinfo.value)
        assert "nside" in message
        assert "lmax" in message
        assert f"lmax >= {floor}" in message, message

    @pytest.mark.parametrize("nside", NSIDES)
    def test_an_lmax_far_below_the_floor_is_refused_the_same_way(self, nside):
        """The other side of the package's two failure texts.

        nside=8/lmax=7 raises ValueError inside s2fft while nside=8/lmax=14
        raises TypeError; the layer must refuse both before either is
        reached, so the refusal is not accidentally tuned to the near edge.
        """
        floor = 2 * nside - 1
        with pytest.raises(ConfigError) as excinfo:
            parse_transform(self._spec(nside, 0), context(), where="t")
        assert f"lmax >= {floor}" in str(excinfo.value)

    @pytest.mark.parametrize("nside", NSIDES)
    def test_this_nsides_floor_is_accepted(self, nside):
        """The rule is ``>=``, so the floor itself is legal.

        An ``lmax <= floor`` mutation and an off-by-one floor both die here.
        """
        _ltj()
        fn, fan = parse_transform(self._spec(nside, 2 * nside - 1), context(),
                                  where="t")
        assert fan == "broadcast" and fn is not None

    @pytest.mark.parametrize("nside", NSIDES)
    def test_an_lmax_above_the_band_limit_is_legal(self, nside):
        """The rule is an INEQUALITY, and this is the test that says so.

        A first version of the Executor's note behind this refusal recorded
        ``lmax == 2 * nside - 1`` -- an equality inferred from four
        confirming probe pairs rather than swept -- and an equality check
        here would reject these documents, which s2fft accepts.  A refusal
        that rejects valid documents is a bug, not a nuisance, so this test
        is not optional decoration.  It runs the transform rather than only
        parsing it, so the layer's "legal" and the package's agree.
        """
        ltj = _ltj()
        floor = 2 * nside - 1
        for lmax in (floor + 1, floor + 9):
            fn, fan = parse_transform(self._spec(nside, lmax), context(),
                                      where="t")
            assert fan == "broadcast"
            maps = jax.random.normal(jax.random.key(1), (2, 12 * nside ** 2))
            alms = fn(maps)
            assert alms.shape == (2, (lmax + 1) * (lmax + 2) // 2)
            assert jnp.allclose(
                alms[0], ltj.map2alm_iter(maps[0], nside=nside, lmax=lmax),
                atol=1e-6)

    def test_the_documented_pair_from_the_executors_note(self):
        """nside=8/lmax=20 -> a (231,) alm vector, the note's own example."""
        _ltj()
        fn, _ = parse_transform(self._spec(8, 20), context(), where="t")
        maps = jax.random.normal(jax.random.key(1), (2, 12 * 8 ** 2))
        assert fn(maps).shape == (2, 231)

    def test_nside_one_has_no_legal_lmax_in_the_package(self):
        """Measured, and the reason the layer refuses nside=1 outright.

        The band limit would otherwise admit ``lmax >= 1`` at nside=1, but
        the package has no working lmax there at all: every lmax from 0 to 11
        fails inside s2fft with ``ValueError: Need at least one array to
        stack``.  Without this the refusal's promise -- "this nside admits
        lmax >= N" -- would be false exactly where a user would act on it.
        """
        ltj = _ltj()
        maps = jax.random.normal(jax.random.key(3), (12,))
        for lmax in range(0, 6):
            with pytest.raises(Exception):  # noqa: B017 - text is not pinned
                ltj.map2alm_iter(maps, nside=1, lmax=lmax)

    def test_nside_one_is_refused_in_the_layers_own_voice(self):
        with pytest.raises(ConfigError, match="nside >= 2"):
            parse_transform(self._spec(1, 4), context(), where="t")

    def test_nside_one_is_refused_before_the_band_limit_speaks(self):
        """At nside=1 the band-limit message would name a floor of 1 and be
        wrong, so the nside guard must come first even when lmax is below
        that floor."""
        with pytest.raises(ConfigError) as excinfo:
            parse_transform(self._spec(1, 0), context(), where="t")
        message = str(excinfo.value)
        assert "nside >= 2" in message
        assert "lmax >= 1" not in message


class TestBuildSpace:
    def test_the_into_sugar_builds_a_working_space(self):
        space, fit = space_for(
            {"g": {"init": 1.0, "linear": True, "into": "gain.gain"}})
        bound = space.bind(fit, {"g": jnp.asarray(2.0)})
        assert float(bound["gain"].gain) == pytest.approx(2.0)

    def test_a_transform_travels_into_the_bind(self):
        space, fit = space_for(
            {"log_g": {"init": 0.0, "into": "gain.gain",
                       "transform": "exp"}})
        bound = space.bind(fit, {"log_g": jnp.asarray(0.0)})
        assert float(bound["gain"].gain) == pytest.approx(1.0)

    def test_split_rows_distributes_over_two_leaves(self):
        space, fit = space_for(
            {"pair": {"init": {"list": [0.25, 2.0]},
                      "into": ["global_signal.depth", "gain.gain"],
                      "transform": "split_rows"}})
        bound = space.bind(fit, {"pair": jnp.asarray([0.25, 2.0])})
        assert float(bound["global_signal"].depth) == pytest.approx(0.25)
        assert float(bound["gain"].gain) == pytest.approx(2.0)

    def test_a_bindings_entry_spells_the_same_thing_longhand(self):
        space, fit = space_for(
            {"g": {"init": 1.0}},
            bindings=[{"latents": ["g"], "into": "gain.gain"}])
        bound = space.bind(fit, {"g": jnp.asarray(3.0)})
        assert float(bound["gain"].gain) == pytest.approx(3.0)

    def test_a_bindings_entry_joins_two_latents_through_python(self):
        space, fit = space_for(
            {"a": {"init": 1.0}, "b": {"init": 2.0}},
            bindings=[{"latents": ["a", "b"], "into": "gain.gain",
                       "transform": {"python": "jax.numpy:add",
                                     "fan": "broadcast"}}])
        bound = space.bind(fit, {"a": jnp.asarray(2.0),
                                 "b": jnp.asarray(3.0)})
        assert float(bound["gain"].gain) == pytest.approx(5.0)

    def test_latents_keep_declaration_order_not_sorted_order(self):
        space, _ = space_for(
            {"z": {"init": 1.0, "into": "gain.gain"},
             "a": {"init": 1.0, "into": "global_signal.depth"}})
        assert space.names == ("z", "a")

    def test_a_binding_into_an_aliased_node_is_refused_up_front(self):
        class _Forked:
            aliased = ("gain",)

        with pytest.raises(ConfigError, match="more than one place"):
            build_space(parse_latents({"g": {"init": 1.0,
                                             "into": "gain.gain"}},
                                      context()),
                        None, None, fit_twin=_Forked(), replaced=(),
                        context=context())

    def test_into_sugar_and_a_bindings_entry_are_mutually_exclusive(self):
        with pytest.raises(ConfigError, match="mutually exclusive"):
            space_for({"g": {"init": 1.0, "into": "gain.gain"}},
                      bindings=[{"latents": ["g"],
                                 "into": "global_signal.depth"}])

    def test_two_bindings_into_one_leaf_are_refused(self):
        with pytest.raises(ConfigError, match="gain"):
            space_for({"a": {"init": 1.0, "into": "gain.gain"},
                       "b": {"init": 1.0, "into": "gain.gain"}})

    def test_a_binding_into_a_replaced_node_is_check_b8(self):
        with pytest.raises(ConfigError, match="replace"):
            space_for({"g": {"init": 1.0, "into": "gain.gain"}},
                      replaced=("gain",))

    def test_a_bindings_entry_naming_an_undeclared_latent_is_refused(self):
        with pytest.raises(ConfigError, match="ghost"):
            space_for({"g": {"init": 1.0, "into": "gain.gain"}},
                      bindings=[{"latents": ["ghost"],
                                 "into": "global_signal.depth"}])

    def test_a_declared_fan_conflicting_with_the_registry_is_refused(self):
        with pytest.raises(ConfigError, match="distribute"):
            space_for({"pair": {"init": {"list": [1.0, 2.0]},
                                "into": ["global_signal.depth", "gain.gain"],
                                "transform": "split_rows",
                                "fan": "broadcast"}})

    def test_a_transform_without_into_is_refused(self):
        with pytest.raises(ConfigError, match="into"):
            space_for({"g": {"init": 1.0, "transform": "exp"}})

    def test_no_parameters_means_no_space(self):
        fit = twin()
        assert build_space(None, None, None, fit_twin=fit, replaced=(),
                           context=context()) is None

    def test_bindings_without_parameters_are_refused(self):
        with pytest.raises(ConfigError, match="parameters"):
            build_space(None, [{"latents": ["g"], "into": "gain.gain"}],
                        None, fit_twin=twin(), replaced=(),
                        context=context())


class TestJointPrior:
    def test_jeffreys_lands_on_the_space(self):
        from rheplicant.inference import JeffreysPrior

        space, _ = space_for(
            {"a": {"init": 1.0, "into": "gain.gain"},
             "b": {"init": 1.0, "into": "global_signal.depth"}},
            joint_prior={"jeffreys": {"over": ["a", "b"]}})
        assert isinstance(space.joint_prior, JeffreysPrior)
        assert space.joint_prior.over == ("a", "b")

    def test_rank_rtol_travels_onto_the_prior(self):
        space, _ = space_for(
            {"a": {"init": 1.0, "into": "gain.gain"}},
            joint_prior={"jeffreys": {"over": ["a"], "rank_rtol": 0.25}})
        assert space.joint_prior.rank_rtol == pytest.approx(0.25)

    def test_only_jeffreys_exists(self):
        with pytest.raises(ConfigError, match="jeffreys"):
            space_for({"a": {"init": 1.0, "into": "gain.gain"}},
                      joint_prior={"reference": {"over": ["a"]}})

    def test_joint_prior_without_parameters_is_refused(self):
        with pytest.raises(ConfigError, match="parameters"):
            build_space(None, None, {"jeffreys": {"over": ["a"]}},
                        fit_twin=twin(), replaced=(), context=context())
