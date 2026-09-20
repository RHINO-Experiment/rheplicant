"""The convergence certificates, on plain callables.

Unit tests for :mod:`rheplicant.inference.certify`. Every objective here is
written in the test and every distance is checked against dense linear
algebra, because the module under test knows nothing about models, plans or
parameter names and neither does this file: the only import from the package
is the module itself.

The acceptance tests that drive the same machinery through a real plan, on
models whose MAP is known in closed form, are in
``test_estimate_reaches_map.py``, ``test_estimate_large_n_float32.py`` and
``test_plan.py``.

Distances are in units of the curvature's standard deviation, which for a
negative log posterior is the posterior sigma. The module's default threshold
stands for 0.1 of them, ``2 * 0.005`` in the units of the objective.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.inference import certify

#: ``2 * gap_tol`` for the 0.1 the callers of this module ask for.
LIMIT = 0.1


@pytest.fixture(scope="module", autouse=True)
def _float64():
    """Module-scoped and restored: the flag is process-global."""
    was = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", was)


def _quadratic(seed, n=30, condition=1e6, distance=0.105, spectrum=None):
    """A rotated SPD Hessian, each coordinate rescaled by up to e^3, and a
    point ``distance`` curvature sigma from its minimum (just outside 0.1)."""
    rng = np.random.default_rng(seed)
    rotation, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigenvalues = np.logspace(0.0, np.log10(condition), n) if spectrum is None else spectrum
    scale = np.exp(rng.uniform(-3.0, 3.0, n))
    hessian = (rotation * eigenvalues) @ rotation.T * np.outer(scale, scale)
    minimum, direction = rng.standard_normal(n), rng.standard_normal(n)
    reach = np.sqrt(abs(direction @ hessian @ direction))
    return hessian, minimum, minimum + distance * direction / reach


def _objective(hessian, minimum):
    """``f(x) = (x - m)^T H (x - m) / 2`` over ``{"x": array}``."""
    matrix, centre = jnp.asarray(hessian), jnp.asarray(minimum)

    def objective(values):
        offset = values["x"] - centre
        return 0.5 * offset @ matrix @ offset

    return objective


def _measure(hessian, minimum, point, **options):
    """:func:`certify.decrement` at ``point``, and the exact answer."""
    objective = _objective(hessian, minimum)
    measured = certify.decrement(objective, {"x": jnp.asarray(point)}, **options)
    slope = hessian @ (point - minimum)
    return measured, slope @ np.linalg.solve(hessian, slope)


def _floor_of(hessian):
    """The smallest eigenvalue: what a caller of the iterative path must know."""
    return float(np.linalg.eigvalsh(hessian)[0])


class TestTheNewtonDecrement:
    """``certify.decrement`` against dense algebra on a known quadratic."""

    @pytest.mark.parametrize("path", ["dense", "conjugate_gradients"])
    @pytest.mark.parametrize("seed", range(4))
    def test_it_is_g_h_inverse_g_within_the_bound_it_reports(
        self, monkeypatch, path, seed
    ):
        """30 parameters at condition number 1e6 before a scaling of up to
        e^3 each, so the scaling the dense path undoes is real. The estimate
        is a lower bound on the true decrement and the reported distance an
        upper one, on both paths; the dense path measures the curvature floor
        the bound needs, the iterative one is given it."""
        floor = None
        hessian, minimum, point = _quadratic(seed)
        if path == "conjugate_gradients":
            monkeypatch.setattr(certify, "DENSE_MAX", 0)
            floor = _floor_of(hessian)
        measured, exact = _measure(hessian, minimum, point, floor=floor)
        assert exact == pytest.approx(0.105**2, rel=1e-6)
        assert measured.status == certify.CONVERGED
        assert measured.lambda2 <= exact * (1 + 1e-7)
        assert exact <= measured.distance**2
        assert not measured.certifies(LIMIT), "0.105 sigma is outside 0.1"
        if path == "dense":
            root = np.sqrt(np.diag(hessian))
            scaled = np.linalg.eigvalsh(hessian / np.outer(root, root))
            assert measured.kappa == pytest.approx(scaled[-1] / scaled[0], rel=1e-6)
            assert measured.products == 31 and measured.dense
            assert abs(measured.lambda2 - exact) <= 1e-10 * exact
        else:
            assert not measured.dense
            assert measured.products <= 4 * 30 + 20 + 1

    def test_a_point_inside_the_threshold_certifies(self):
        """The twin of the case above: the same fixture at 0.05 sigma."""
        hessian, minimum, point = _quadratic(0, distance=0.05)
        measured, exact = _measure(hessian, minimum, point)
        assert measured.estimate == pytest.approx(0.05, rel=1e-6)
        assert measured.certifies(LIMIT)
        assert math.sqrt(exact) <= measured.distance <= LIMIT

    @pytest.mark.parametrize("path", ["dense", "conjugate_gradients"])
    def test_negative_curvature_is_not_a_minimum(self, monkeypatch, path):
        if path == "conjugate_gradients":
            monkeypatch.setattr(certify, "DENSE_MAX", 0)
        spectrum = np.concatenate([[-1.0], np.logspace(0.0, 2.0, 7)])
        hessian, minimum, point = _quadratic(3, n=8, spectrum=spectrum, distance=1e-3)
        measured, _ = _measure(hessian, minimum, point)
        assert measured.status == certify.NONCONVEX
        assert not measured.certifies(LIMIT)

    def test_an_iteration_that_does_not_reach_its_residual_certifies_nothing(
        self, monkeypatch
    ):
        monkeypatch.setattr(certify, "DENSE_MAX", 0)
        monkeypatch.setattr(certify, "MAXITER", 3)
        hessian, minimum, point = _quadratic(0, distance=1e-3)
        measured, _ = _measure(hessian, minimum, point, floor=_floor_of(hessian))
        assert measured.status == certify.UNREACHED
        assert measured.estimate < LIMIT and not measured.certifies(LIMIT)

    def test_a_complex_parameter_enters_as_its_real_and_imaginary_parts(self):
        """``f = Re(d^H H d) / 2`` for Hermitian ``H = A + iB`` is the real
        quadratic form of ``[[A, -B], [B, A]]`` in ``(Re, Im)``, and that is
        the Hessian the decrement must measure: 2n real degrees of freedom,
        not n."""
        n, rng = 5, np.random.default_rng(4)
        symmetric = rng.standard_normal((n, n))
        symmetric = symmetric @ symmetric.T + n * np.eye(n)
        skew = rng.standard_normal((n, n))
        skew = skew - skew.T
        hermitian = jnp.asarray(symmetric + 1j * skew)
        real_form = np.block([[symmetric, -skew], [skew, symmetric]])
        assert np.all(np.linalg.eigvalsh(real_form) > 0), "the fixture must be convex"
        minimum = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        offset = rng.standard_normal(n) + 1j * rng.standard_normal(n)

        def objective(values):
            gap = values["z"] - jnp.asarray(minimum)
            return 0.5 * jnp.real(jnp.conj(gap) @ (hermitian @ gap))

        point = minimum + 0.01 * offset
        measured = certify.decrement(objective, {"z": jnp.asarray(point)})
        parts = np.concatenate([(point - minimum).real, (point - minimum).imag])
        slope = real_form @ parts
        exact = slope @ np.linalg.solve(real_form, slope)
        assert measured.status == certify.CONVERGED
        assert measured.products == 2 * n + 1, "one product per REAL degree of freedom"
        assert measured.lambda2 == pytest.approx(exact, rel=1e-10)
        assert measured.residual < 1e-10

    def test_an_early_stop_cannot_turn_a_refusal_into_a_certificate(self, monkeypatch):
        """The guarantee, made visible by stopping the solve early.

        At ``rtol = 0.1`` on a condition number of 1e4 the conjugate
        gradients leave the decrement 2 to 8 per cent low (measured), so a
        point 0.102 curvature sigma from the minimum — outside the 0.1 the
        threshold stands for — has estimates that fall inside it. The
        residual is what refuses them: it is recomputed from the iterate
        rather than carried from the iteration, and the bound it gives covers
        the true decrement at every seed here.
        """
        monkeypatch.setattr(certify, "DENSE_MAX", 0)
        monkeypatch.setattr(certify, "RTOL", {4: 0.1, 8: 0.1})
        inside = 0
        for seed in range(6):
            hessian, minimum, point = _quadratic(seed, condition=1e4, distance=0.102)
            # the floor a caller proves: without one the probe would refuse
            # this conditioning outright, and the early stop would not be
            # exercised at all
            measured, exact = _measure(hessian, minimum, point,
                                       floor=_floor_of(hessian))
            assert measured.status == certify.CONVERGED, seed
            assert measured.reach > 0.0, "the residual must reach somewhere"
            assert measured.lambda2 < exact * (1 - 1e-6), "the solve must stop short"
            assert exact <= measured.distance**2, seed
            assert not measured.certifies(LIMIT), seed
            inside += 0.0 < measured.estimate <= LIMIT
        assert inside, "no seed's estimate fell inside the threshold: vacuous"

    @pytest.mark.parametrize(
        "lambda2, reach, kappa, dtype, status, certified",
        [
            # reach 1e-3 on an estimate of 0.09899: the bound is 0.09949
            (0.0098, 1e-3, 1e4, jnp.float64, certify.CONVERGED, True),
            # the same estimate with a residual ten times larger: 0.1045
            (0.0099, 1e-2, 1e4, jnp.float64, certify.CONVERGED, False),
            # a residual that bounds nothing at all
            (1e-6, float("inf"), 1e4, jnp.float64, certify.CONVERGED, False),
            # 4-byte floats at kappa 1e7: eps * kappa = 1.19, all rounding
            (1e-6, 0.0, 1e7, jnp.float32, certify.CONVERGED, False),
            # a solve that did not reach its residual, or met non-positive
            # curvature, certifies nothing however small its estimate
            (1e-6, 0.0, 1.0, jnp.float64, certify.UNREACHED, False),
            (1e-6, 0.0, 1.0, jnp.float64, certify.NONCONVEX, False),
            (float("nan"), 0.0, 1.0, jnp.float64, certify.CONVERGED, False),
        ],
    )
    def test_it_passes_only_on_the_upper_bound(
        self, lambda2, reach, kappa, dtype, status, certified
    ):
        """The rule on hand-made programs, at the 0.1 threshold: an estimate
        inside it whose residual reaches outside it is refused."""

        def program(values):
            return (jnp.asarray(lambda2, dtype), jnp.asarray(0.0, dtype),
                    jnp.asarray(reach, dtype), 5, status, jnp.asarray(kappa, dtype))

        measured = certify.decrement(None, {}, program=program)
        assert measured.certifies(LIMIT) is certified
        assert measured.products == 5 and measured.status == status

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    @pytest.mark.parametrize("tiny", [1e-6, 1e-8])
    def test_an_eigenvalue_the_iteration_barely_reaches_is_not_certified_away(
        self, monkeypatch, dtype, tiny
    ):
        """The third review's HIGH, above :data:`certify.DENSE_MAX`.

        300 parameters, the spectrum clustered at 1 with one eigenvalue at
        ``tiny``, and the gradient's component along that direction just
        under the iteration's own residual: the conjugate gradients stop
        with a residual that looks small and an error that is not, and a
        condition number read off their Lanczos matrix says 1.2 where the
        truth is 1e8. Reported that way, a bound of 0.04998 stood for a true
        0.0673 (float32, measured). The floor the bound divides by is now
        the probe's Ritz interval, which reaches below the cluster or says it
        cannot: the distance reported must cover the truth, whether or not
        the point certifies.
        """
        monkeypatch.setattr(certify, "DENSE_MAX", 0)  # 300 parameters are dense now
        n, rng = 300, np.random.default_rng(2)
        rtol = certify.RTOL[jnp.finfo(dtype).dtype.itemsize]
        spectrum = np.concatenate([1.0 + 0.01 * rng.standard_normal(n - 1), [tiny]])
        rotation, _ = np.linalg.qr(rng.standard_normal((n, n)))
        hessian = (rotation * spectrum) @ rotation.T
        hessian = 0.5 * (hessian + hessian.T)
        bulk = rotation[:, : n - 1] @ rng.standard_normal(n - 1)
        bulk = bulk / np.linalg.norm(bulk) * 0.05
        slope = bulk + 0.9 * rtol * np.linalg.norm(bulk) * rotation[:, -1]
        point = np.linalg.solve(hessian, slope)
        exact = float(slope @ point)
        matrix = jnp.asarray(hessian, dtype)

        def objective(values):
            return 0.5 * jnp.sum(values["x"] * (matrix @ values["x"]))

        measured = certify.decrement(objective, {"x": jnp.asarray(point, dtype)})
        assert measured.lambda2 <= exact * (1 + 1e-6), "the estimate is a lower bound"
        assert math.sqrt(exact) <= measured.distance, (
            f"{measured} reported a bound below the true {math.sqrt(exact):.4g}"
        )
        assert not measured.dense and measured.products >= certify.PROBE_STEPS
        if dtype is jnp.float64:
            # and not by refusing: the probe reaches under the cluster here,
            # so the bound is a number, and it equals the true distance
            assert measured.status == certify.CONVERGED
            assert measured.distance == pytest.approx(math.sqrt(exact), rel=1e-3)
        else:
            # in float32 that spectrum is numerically singular; the probe
            # cannot put its interval above zero and the point is refused
            assert not measured.certifies(LIMIT)

    @pytest.mark.parametrize("path", ["dense", "conjugate_gradients"])
    def test_the_two_paths_agree_on_what_is_not_a_minimum(self, monkeypatch, path):
        """The third review's second HIGH. Above :data:`certify.DENSE_MAX` the
        Hessian is not formed, and a zero gradient used to be read as a
        converged solve whatever the curvature, so a saddle certified at
        distance zero while the dense path refused the same matrix. Both
        paths now refuse an indefinite Hessian, at a saddle or away from it,
        and a singular one whose gradient happens to avoid the null
        direction."""
        n = 300
        if path == "conjugate_gradients":
            monkeypatch.setattr(certify, "DENSE_MAX", 0)
        rng = np.random.default_rng(1)
        rotation, _ = np.linalg.qr(rng.standard_normal((n, n)))
        spectrum = np.linspace(1.0, 10.0, n)

        def refuses(eigenvalues, point):
            hessian = (rotation * eigenvalues) @ rotation.T
            measured, _ = _measure(hessian, np.zeros(n), point)
            return measured

        indefinite, singular = spectrum.copy(), spectrum.copy()
        indefinite[0], singular[0] = -1.0, 0.0
        point = rng.standard_normal(n) * 1e-3
        assert refuses(spectrum, point).certifies(LIMIT), "the fixture must be usable"
        for eigenvalues, where in ((indefinite, point), (indefinite, np.zeros(n)),
                                   (singular, point)):
            measured = refuses(eigenvalues, where)
            assert measured.status == certify.NONCONVEX, (path, measured)
            assert not measured.certifies(LIMIT)

    def test_a_verified_floor_makes_a_large_model_cheap_to_certify(self):
        """What a caller's ``floor`` buys above :data:`certify.DENSE_MAX`: the
        iteration stops as soon as its bound is inside the limit, instead of
        probing the spectrum and driving the residual down. Measured on 2000
        parameters: two products against 35."""
        n, rng = 2000, np.random.default_rng(5)
        spectrum = np.linspace(1.0, 50.0, n)
        matrix = jnp.asarray(np.diag(spectrum))

        def objective(values):
            return 0.5 * jnp.sum(values["x"] * (matrix @ values["x"]))

        point = rng.standard_normal(n)
        point = point / np.sqrt(point @ (spectrum * point)) * 0.05
        at = {"x": jnp.asarray(point)}
        floored = certify.decrement(objective, at, floor=1.0, limit=LIMIT)
        probed = certify.decrement(objective, at, limit=LIMIT)
        assert floored.certifies(LIMIT) and probed.certifies(LIMIT)
        assert floored.products < probed.products
        assert floored.products <= 4 and probed.products >= certify.PROBE_STEPS

    def test_a_program_is_built_once_and_reused(self):
        """The program is the expensive half: a caller holds it across points."""
        hessian, minimum, point = _quadratic(1, n=6, condition=1e2, distance=0.05)
        objective = _objective(hessian, minimum)
        program = certify.decrement_program(objective, {"x": jnp.asarray(point)})
        first = certify.decrement(objective, {"x": jnp.asarray(point)}, program=program)
        second = certify.decrement(
            objective, {"x": jnp.asarray(minimum)}, program=program
        )
        assert first.estimate == pytest.approx(0.05, rel=1e-6)
        assert second.estimate == 0.0 and second.certifies(LIMIT)


class TestThePolish:
    """``certify.polish``: Steihaug-truncated CG and an Armijo search.

    Each case is one a review measured going wrong, or one a mutant of the
    acceptance tests survived.
    """

    @staticmethod
    def _polish(objective, x, iterations=3, dtype=None):
        start = {"x": jnp.asarray(x) if dtype is None else jnp.asarray(x, dtype)}
        point, kept = certify.polish(objective, start, iterations=iterations)
        return float(point["x"]), int(kept)

    def test_negative_curvature_is_descended_and_never_climbed(self):
        # -cos(x) at x = 2: plain Newton heads past the maximum at pi to 4.19,
        # where the objective is HIGHER (0.42 -> 0.50). Steihaug stops at the
        # negative curvature and takes the gradient direction, so the three
        # iterations go downhill, towards the minimum at 0.
        def objective(x):
            return -jnp.cos(x["x"])

        polished, kept = self._polish(objective, 2.0)
        assert abs(polished) < 2.0 and -np.cos(polished) < -np.cos(2.0)
        assert kept == 3
        assert abs(self._polish(objective, 0.5)[0]) < 1e-6  # the convex basin

    def test_a_saddle_is_not_where_the_step_goes(self):
        # f = (x y - 1)**2 / 2 + (x**2 + y**2) / 200 at (0.8, -0.3): the full
        # Newton step of plain CG lands near the stationary point at the
        # origin, a saddle. Steihaug with Armijo must leave f lower and away
        # from it.
        def objective(v):
            x, y = v["x"], v["y"]
            return 0.5 * (x * y - 1.0) ** 2 + (x**2 + y**2) / 200.0

        start = {"x": jnp.asarray(0.8), "y": jnp.asarray(-0.3)}
        out, _ = certify.polish(objective, start)
        assert float(objective(out)) < float(objective(start))
        assert float(jnp.hypot(out["x"], out["y"])) > 0.3

    def test_a_step_the_objective_cannot_resolve_is_refused(self):
        # The review's case: a tail of 1e-3 * logcosh(y) on an offset of 1e7,
        # in a 4-byte float. No change of y moves the sum, so no step can
        # satisfy Armijo's decrease; ``after <= before`` accepted a jump from
        # y = 3 to y = -97.9.
        def objective(x):
            y = x["x"]
            tail = jnp.abs(y) + jnp.log1p(jnp.exp(-2.0 * jnp.abs(y))) - jnp.log(2.0)
            return jnp.asarray(1e7, jnp.float32) + 1e-3 * tail

        assert self._polish(objective, 3.0, dtype=jnp.float32) == (3.0, 0)

    def test_a_non_finite_trial_is_backtracked_not_taken(self):
        # x**2 above 0.25, -inf below: from x = 1 the full Newton step lands on
        # 0, where the objective is -inf and "lower"; the finite test refuses
        # it and one halving lands on 0.5.
        def objective(x):
            y = x["x"]
            return jnp.where(y > 0.25, y**2, -jnp.inf)

        assert self._polish(objective, 1.0, iterations=1) == (0.5, 1)

    def test_a_complex_parameter_set_is_returned_unchanged(self):
        def objective(x):
            return jnp.real(jnp.conj(x["z"]) * x["z"])

        start = {"z": jnp.asarray(1.0 + 2.0j)}
        point, kept = certify.polish(objective, start)
        assert complex(point["z"]) == 1.0 + 2.0j and int(kept) == 0


class TestTheGapPreScreen:
    """``certify.gap_step`` on hand-made decreases, tol = 0.005.

    The pre-screen picks the iterations at which the decrement is computed;
    it certifies nothing, and an iteration it passes is only a candidate.
    """

    @staticmethod
    def _run(decreases, resolution=1e-9, tol=0.005):
        state, out = certify.GapState(), []
        for decrease in decreases:
            state, passed, gap, rho = certify.gap_step(state, decrease, resolution, tol)
            out.append((passed, gap, rho))
        return out

    def test_a_geometric_run_passes_once_its_tail_is_inside_tol(self):
        # D[k] = 0.5**k: the gap after D is (D + resolution) / (1 - 0.5).
        out = self._run([0.5**k for k in range(1, 12)])
        for (passed, gap, rho), k in zip(out[1:], range(2, 12), strict=True):
            assert rho == pytest.approx(0.5)
            assert gap == pytest.approx(2 * (0.5**k + 1e-9), rel=1e-12)
            assert passed is (gap <= 0.005)
        assert [passed for passed, _, _ in out].index(True) == 8  # 2**-9

    def test_a_slow_contraction_is_not_passed_by_a_small_step(self):
        # D = 1e-4 per iteration at rho = 0.999: the tail is 0.1, 20x tol.
        out = self._run([1e-4 * 0.999**k for k in range(10)])
        assert not any(passed for passed, _, _ in out)

    def test_a_rise_passes_nothing_and_forgets_the_contraction(self):
        out = self._run([1e-3, 5e-4, -1e-3, 1e-6])
        assert out[2] == (False, None, None)
        assert out[3][0] is False and out[3][2] is None

    def test_a_stall_below_the_resolution_is_passed_to_the_decrement(self):
        """A long valley stalled on rounding: resolvable decreases contracting
        at 0.9999, then nothing the arithmetic can see. The screen cannot tell
        that from a converged run, so it passes it, and the decrement decides
        (the acceptance tests measure it refusing such stalls)."""
        out = self._run([1e-3 * 0.9999**k for k in range(5)] + [0.0] * 5,
                        resolution=1e-8)
        assert not any(passed for passed, _, _ in out[:5])
        assert all(passed for passed, _, _ in out[5:])

    def test_a_run_that_never_moved_uses_a_contraction_of_zero(self):
        out = self._run([0.0, 0.0, 0.0], resolution=1e-9)
        assert [passed for passed, _, _ in out] == [True, True, True]
        assert out[-1][2] == 0.0


class TestTheChangeTest:
    """``certify.settled``, ``effective_tol``, ``change_program`` and the floor."""

    def test_the_stop_rule_counts_changes_in_both_directions(self):
        """A rise beyond the tolerance is not settled, which a one-sided test
        on the decrease would have called settled."""
        tol = 1e-6
        assert certify.settled([10.0, 10.0, 10.0], tol)
        assert certify.settled([10.0, 10.0 + 5e-6, 10.0], tol)
        assert not certify.settled([10.0, 10.5, 11.0], tol)       # rising
        assert not certify.settled([11.0, 10.5, 10.0], tol)       # falling
        assert not certify.settled([10.0, 10.0, 11.0], tol)       # one change only
        assert not certify.settled([10.0, 10.0], tol)             # one change is not two

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_the_tolerance_is_floored_at_the_dtypes_resolution(self, dtype):
        eps = float(jnp.finfo(dtype).eps)
        objective = jnp.asarray(1.0, dtype)
        assert certify.effective_tol(1e-8, objective) == max(
            1e-8, certify.OBJECTIVE_FLOOR_EPS * eps
        )
        assert certify.effective_tol(1e-3, objective) == 1e-3

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_an_inner_solve_is_tightened_down_to_the_dtypes_floor(self, dtype):
        floor = certify.solve_tol_floor(jnp.asarray(1.0, dtype))
        assert floor == certify.SOLVE_TOL_FLOOR[jnp.finfo(dtype).dtype.itemsize]
        assert floor > float(jnp.finfo(dtype).eps)

    def test_the_change_is_summed_term_by_term_and_carries_its_rounding(self):
        """The difference of the terms, not of their totals: a constant
        common to both evaluations cancels exactly, however large."""
        change = certify.change_program()
        offset = 1e8
        terms0 = {"a": jnp.asarray([offset, 1.0]), "b": jnp.asarray([2.0])}
        terms1 = {"a": jnp.asarray([offset, 0.5]), "b": jnp.asarray([1.5])}
        scales = {"a": jnp.asarray([offset, 1.0]), "b": jnp.asarray([2.0])}
        decrease, resolution = change(terms0, scales, terms1, scales)
        assert float(decrease) == pytest.approx(1.0, rel=1e-12)
        spread = np.sqrt((2 * offset) ** 2 + 2.0**2 + 4.0**2)
        eps = float(jnp.finfo(jnp.float64).eps)
        assert float(resolution) == pytest.approx(
            certify.RESOLUTION_EPS * eps * spread, rel=1e-12
        )

    def test_the_resolution_follows_the_working_precision(self):
        change = certify.change_program()
        terms = {"a": jnp.asarray([1.0], jnp.float32)}
        scales = {"a": jnp.asarray([1.0], jnp.float32)}
        _, coarse = change(terms, scales, terms, scales)
        wide = {"a": jnp.asarray([1.0], jnp.float64)}
        _, fine = change(wide, wide, wide, wide)
        assert float(coarse) / float(fine) == pytest.approx(
            float(jnp.finfo(jnp.float32).eps) / float(jnp.finfo(jnp.float64).eps),
            rel=1e-6,
        )
