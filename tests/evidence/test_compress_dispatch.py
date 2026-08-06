"""Routing, and accumulating T1 in coefficient space.

T1's `info` is a `SqrtInfo` over the basis coefficients and T2's is over theta.
They are quadratic forms in different vectors, so the memory carries two
accumulators and never one sum of the two. The refusal is checked here twice:
once at the router, and once at `SqrtInfo.combine`, which refuses a mixture by
name one level down whatever the router does.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.core.errors import StateValidationError
from rheplicant.inference import Bind, Latent, ParameterSpace
from rheplicant.inference.compress import compress, compress_reduced_basis
from rheplicant.inference.compressed import QuadraticLikelihood, ReducedBasisLikelihood
from rheplicant.inference.factorize import Factorization
from rheplicant.inference.memory import BayesMemory
from rheplicant.inference.reduced_basis import build_reduced_basis, score_directions
from rheplicant.inference.sqrtinfo import SqrtInfo
from tests.evidence import rhino_bank

#: The whitened bank's measured numerical rank. `build_reduced_basis` refuses
#: anything above it, so this is not a taste.
N_BASIS = 6


@pytest.fixture(scope="module")
def basis():
    return build_reduced_basis(
        rhino_bank.space(),
        rhino_bank.pipeline(),
        rhino_bank.state(),
        noise=rhino_bank.noise(),
        bank=rhino_bank.bank(jax.random.key(40)),
        n_basis=N_BASIS,
        support=rhino_bank.support(),
    )


def _memory():
    return BayesMemory(Factorization(rhino_bank.space()))


def _epoch(basis, index):
    return compress_reduced_basis(
        basis,
        observed=rhino_bank.observed(jax.random.fold_in(jax.random.key(41), index)),
        noise=rhino_bank.noise(),
        epoch_id=f"n{index}",
    )


def _linear_term(epoch_id="lin0", key=47):
    """A T2 term over the SAME four latents, from their score directions.

    A linear model built out of `dmu/dtheta` at the truth: the tangent model
    every tier agrees with to first order, which is what makes summing it with a
    T1 term a meaningful thing to check rather than an arithmetic accident.
    """
    scores = score_directions(rhino_bank.space(), rhino_bank.pipeline(), rhino_bank.state())
    design = {name: scores[name].T for name in rhino_bank.TRUTH}
    return compress(
        design=design,
        observed=rhino_bank.observed(jax.random.key(key)),
        noise_std=np.asarray(1.0 / rhino_bank.weight()),
        shapes={name: () for name in design},
        epoch_id=epoch_id,
    )


# ------------------------------------------------------------------ routing --


def test_a_fully_linear_model_routes_to_the_quadratic_tier():
    design = jax.random.normal(jax.random.key(42), (30, 2))
    term = compress(
        design={"x": design},
        observed=jax.random.normal(jax.random.key(43), (30,)),
        noise_std=0.1,
        shapes={"x": (2,)},
        epoch_id="e0",
    )
    assert isinstance(term, QuadraticLikelihood)
    assert term.exact is True


def test_a_basis_routes_to_the_reduced_tier(basis):
    term = compress(
        basis=basis,
        observed=rhino_bank.observed(jax.random.key(44)),
        noise=rhino_bank.noise(),
        epoch_id="e0",
    )
    assert isinstance(term, ReducedBasisLikelihood)
    assert term.exact is False


def test_the_dispatcher_returns_what_the_tier_returns_and_invents_nothing(basis):
    """Routing is a choice of function, not a computation of its own."""
    direct = compress_reduced_basis(
        basis,
        observed=rhino_bank.observed(jax.random.key(44)),
        noise=rhino_bank.noise(),
        epoch_id="e0",
    )
    routed = compress(
        basis=basis,
        observed=rhino_bank.observed(jax.random.key(44)),
        noise=rhino_bank.noise(),
        epoch_id="e0",
    )
    _, values = rhino_bank.forward()
    assert float(routed(values)) == float(direct(values))


def test_supplying_neither_a_design_nor_a_basis_is_refused():
    with pytest.raises(StateValidationError, match="T3"):
        compress(observed=jnp.zeros(4), noise=rhino_bank.noise(), epoch_id="e0")


def test_supplying_both_is_refused_rather_than_silently_preferred():
    with pytest.raises(StateValidationError, match="both"):
        compress(
            design={"x": jnp.zeros((4, 1))},
            basis=object(),
            observed=jnp.zeros(4),
            noise_std=0.1,
            shapes={"x": (1,)},
            epoch_id="e0",
        )


# ------------------------------------------------------------ accumulating --


def test_a_memory_accumulates_t1_terms_and_sums_their_densities(basis):
    memory = _memory()
    terms = [_epoch(basis, index) for index in range(4)]
    for term in terms:
        memory = memory.remember(term)
    _, values = rhino_bank.forward()
    assert float(memory.log_likelihood(values)) == pytest.approx(
        sum(float(term(values)) for term in terms), rel=1e-9
    )


def test_accumulating_t1_is_order_invariant(basis):
    terms = [_epoch(basis, index) for index in range(4)]
    forward_order = _memory()
    for term in terms:
        forward_order = forward_order.remember(term)
    reverse_order = _memory()
    for term in reversed(terms):
        reverse_order = reverse_order.remember(term)
    _, values = rhino_bank.forward()
    assert float(forward_order.log_likelihood(values)) == pytest.approx(
        float(reverse_order.log_likelihood(values)), rel=1e-10
    )


def test_a_term_built_on_a_different_basis_is_refused(basis):
    other = build_reduced_basis(
        rhino_bank.space(),
        rhino_bank.pipeline(),
        rhino_bank.state(),
        noise=rhino_bank.noise(),
        bank=rhino_bank.bank(jax.random.key(45)),
        n_basis=N_BASIS,
        support=rhino_bank.support(),
    )
    assert other.fingerprint() != basis.fingerprint()
    memory = _memory().remember(_epoch(basis, 0))
    stranger = compress_reduced_basis(
        other,
        observed=rhino_bank.observed(jax.random.key(46)),
        noise=rhino_bank.noise(),
        epoch_id="n1",
    )
    with pytest.raises(StateValidationError, match="dictionary"):
        memory.remember(stranger)


def test_the_two_accumulators_cannot_be_summed_even_if_the_router_were_wrong(basis):
    """The refusal one level down, which does not depend on the routing.

    `SqrtInfo.combine` compares names, and a T1 term's are `(COEFFICIENTS,)`
    while the theta accumulator's are the campaign's latents. So a mis-routed T1
    term does not quietly add a quadratic in `c` to a quadratic in `theta`; it
    stops. Checked here so the router is defence in depth rather than the only
    defence.
    """
    memory = _memory().remember(_epoch(basis, 0))
    assert memory.coefficients is not None
    with pytest.raises(StateValidationError, match="different latents"):
        SqrtInfo.combine(memory.accumulated, memory.coefficients)


def test_a_memory_holding_one_of_each_tier_sums_both(basis):
    """Different epochs may be compressed differently; the densities still add."""
    memory = _memory().remember(_epoch(basis, 0))
    _, values = rhino_bank.forward()
    only_t1 = float(memory.log_likelihood(values))

    linear = _linear_term()
    memory = memory.remember(linear)
    assert float(memory.log_likelihood(values)) == pytest.approx(
        only_t1 + float(linear(values)), rel=1e-9
    )


def test_a_t2_term_over_the_wrong_latents_is_still_refused(basis):
    """The relaxation the T1 route needs must not loosen the T2 route.

    Matched on `memory accumulates`, which only the memory's own guard says.
    `SqrtInfo.combine` refuses this too, one line later and in different words
    -- so a test matching the shared phrase "different latents" would pass with
    the guard deleted, and the caller would read a message about two quadratic
    forms instead of one naming the campaign's declared latents.
    """
    memory = _memory().remember(_epoch(basis, 0))
    design = jax.random.normal(jax.random.key(48), (30, 2))
    stranger = compress(
        design={"elsewhere": design},
        observed=jax.random.normal(jax.random.key(49), (30,)),
        noise_std=0.1,
        shapes={"elsewhere": (2,)},
        epoch_id="lin1",
    )
    with pytest.raises(StateValidationError, match="memory accumulates"):
        memory.remember(stranger)


def _space(names):
    latents = tuple(
        Latent(
            name,
            init=jnp.asarray(rhino_bank.TRUTH[name]),
            prior=rhino_bank._Normal(rhino_bank.TRUTH[name], rhino_bank.PRIOR_STD[name]),
        )
        for name in names
    )
    return ParameterSpace(
        latents=latents,
        bindings=tuple(
            Bind(latent.name, into=lambda p, n=latent.name: getattr(p, n))
            for latent in latents
        ),
    )


def test_a_t1_term_expanding_in_an_undeclared_latent_is_refused(basis):
    """The coefficient map would be evaluated at values the sampler never draws.

    A T1 term's columns are coefficients, so the equality check the T2 route
    uses cannot apply. What is checked instead is the set of latents the
    coefficient map CONSUMES, and this memory does not declare `running`.
    """
    memory = BayesMemory(Factorization(_space(("t21_depth", "index", "amplitude"))))
    with pytest.raises(StateValidationError, match="running"):
        memory.remember(_epoch(basis, 0))


def test_a_t1_term_expanding_in_a_SUBSET_is_accepted():
    """The nearest legitimate case: a guard that over-refuses is its own bug.

    A basis seeded on two of the four latents says nothing about the other two.
    That is a rank-deficient epoch, which is the normal case for one night and
    exactly what the square-root form exists to represent -- not an error.
    """
    partial = build_reduced_basis(
        rhino_bank.space(),
        rhino_bank.pipeline(),
        rhino_bank.state(),
        noise=rhino_bank.noise(),
        bank=rhino_bank.bank(jax.random.key(50)),
        n_basis=N_BASIS,
        support=rhino_bank.support(),
        names=("t21_depth", "index"),
    )
    term = _epoch(partial, 0)
    assert term.latents == ("t21_depth", "index")
    memory = _memory().remember(term)
    _, values = rhino_bank.forward()
    assert float(memory.log_likelihood(values)) == pytest.approx(
        float(term(values)), rel=1e-12
    )


# --------------------------------------------------------------- the shapes --


def _density_state(memory: BayesMemory) -> BayesMemory:
    """The same memory with the archive dropped -- what the density path reads."""
    return BayesMemory(
        memory.factorization,
        memory.accumulated,
        (),
        coefficients=memory.coefficients,
        basis=memory.basis,
    )


def test_the_density_path_treedef_is_fixed_and_the_archive_is_what_grows(basis):
    """Plan A measured 60 traces and 3.66 s for 60 remembers into a jitted
    density over a growing pytree, against 1 trace for a fixed-treedef one.

    What is fixed is the DENSITY PATH -- the two accumulators plus the shared
    dictionary -- and not the memory as a whole: the archive gains one term per
    epoch, so `eqx.filter_jit` over a whole `BayesMemory` still retraces once
    per epoch. That is why `to_numpyro_model` closes over the density path
    alone rather than over `self`, and it is the shape measured here.

    The retrace has a different cause than it used to and the same consequence.
    The archive was a tuple, so it grew the treedef; it is now one opaque leaf,
    which `filter_jit` sorts to the static side because it is not an array, and
    a static value that changes is a cache miss just as a new treedef is. Both
    assertions below hold either way, which is the point of measuring traces
    rather than structures: `_Archive.__eq__` is what makes the first one hold,
    and it compares terms by identity for reasons its own comment gives.
    """
    traces, whole = [], []

    @eqx.filter_jit
    def density(memory, values):
        traces.append(1)
        return memory.log_likelihood(values)

    @eqx.filter_jit
    def over_the_whole_memory(memory, values):
        whole.append(1)
        return memory.log_likelihood(values)

    memory = _memory()
    _, values = rhino_bank.forward()
    for index in range(5):
        memory = memory.remember(_epoch(basis, index))
        density(_density_state(memory), values)
        over_the_whole_memory(memory, values)

    assert len(traces) == 1, (
        f"{len(traces)} traces for 5 epochs: the accumulator's treedef is growing "
        "with the campaign, which is the failure the fixed-treedef design exists "
        "to prevent."
    )
    assert len(whole) == 5, (
        "The archive holds one term per epoch, so jitting the whole memory must "
        "retrace per epoch. If this ever reports 1, the archive stopped being "
        "carried and the diagnostics went with it."
    )
    assert float(density(_density_state(memory), values)) == pytest.approx(
        float(memory.log_likelihood(values)), rel=1e-12
    )


def test_the_accumulators_keep_their_shape_for_the_life_of_a_campaign(basis):
    memory = _memory()
    for index in range(6):
        memory = memory.remember(_epoch(basis, index))
        assert memory.accumulated.factor.shape == (4, 4)
        assert memory.coefficients.factor.shape == (N_BASIS, N_BASIS)


# --------------------------------------------------------------- downstream --


def test_the_numpyro_model_carries_the_reduced_half_too(basis):
    """A memory holding only T1 has an EMPTY theta accumulator.

    Sampling it against `accumulated` alone gives a smooth, finite, perfectly
    well-behaved posterior that is exactly the prior -- no error, no warning,
    and no data. The reduced half has to reach the entry point too, and the
    assertion below is exactly that difference: the theta accumulator's own
    log-probability here is 0.0.
    """
    numpyro = pytest.importorskip("numpyro")
    import numpyro.distributions as dist

    # The same four latents in the same non-alphabetical order, with priors
    # numpyro can actually sample; rhino_bank's are duck-typed for log_prob.
    latents = tuple(
        Latent(
            name,
            init=jnp.asarray(rhino_bank.TRUTH[name]),
            prior=dist.Normal(rhino_bank.TRUTH[name], rhino_bank.PRIOR_STD[name]),
        )
        for name in ("t21_depth", "index", "amplitude", "running")
    )
    space = ParameterSpace(
        latents=latents,
        bindings=tuple(
            Bind(latent.name, into=lambda p, n=latent.name: getattr(p, n))
            for latent in latents
        ),
    )
    memory = BayesMemory(Factorization(space))
    for index in range(3):
        memory = memory.remember(_epoch(basis, index))
    _, values = rhino_bank.forward()

    assert float(memory.accumulated.log_prob(values)) == 0.0
    trace = numpyro.handlers.trace(
        numpyro.handlers.substitute(memory.to_numpyro_model(), values)
    ).get_trace()
    assert float(trace["campaign"]["fn"].log_factor) == pytest.approx(
        float(memory.log_likelihood(values)), rel=1e-9
    )


def test_audit_names_both_tiers_rather_than_reporting_one_for_an_archive_of_both(basis):
    """`fisher_*` describes the theta accumulator ONLY.

    For a memory holding only T1 terms that accumulator is the null factor, so
    `fisher_lambda_min` is 0 and `fisher_condition` is inf -- which reads as a
    degenerate campaign when the campaign is simply not quadratic in theta. The
    coefficient half gets its own pair, and the dictionary it is against is
    named, so the report cannot be read as covering more than it does.
    """
    memory = _memory()
    for index in range(3):
        memory = memory.remember(_epoch(basis, index))
    report = memory.audit()

    assert report["n_epochs"] == 3
    assert report["n_reduced"] == 3
    assert report["all_exact"] is False
    assert report["basis_fingerprint"] == basis.fingerprint()
    assert report["fisher_lambda_min"] == 0.0
    assert report["fisher_condition"] == float("inf")
    assert np.isfinite(report["coefficient_lambda_min"])
    assert report["coefficient_lambda_min"] > 0.0
    assert np.isfinite(report["coefficient_condition"])


def test_audit_reports_no_reduced_half_when_there_is_none():
    memory = _memory().remember(_linear_term())
    report = memory.audit()
    assert report["n_reduced"] == 0
    assert report["basis_fingerprint"] is None
    assert report["coefficient_lambda_min"] is None
    assert report["coefficient_condition"] is None
    assert report["all_exact"] is True
