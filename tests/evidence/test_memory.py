import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.core.errors import StateValidationError
from rheplicant.inference import (
    Bind,
    Latent,
    ParameterSpace,
    parameter_covariance,
    propagate_covariance,
)
from rheplicant.inference.compressed import QuadraticLikelihood
from rheplicant.inference.factorize import Factorization
from rheplicant.inference.memory import BayesMemory
from rheplicant.inference.sqrtinfo import SqrtInfo


class _Normal:
    """Minimal duck-typed prior: only log_prob is used by BayesMemory."""

    def __init__(self, loc, scale):
        self.loc, self.scale = loc, scale

    def log_prob(self, x):
        return -0.5 * (((x - self.loc) / self.scale) ** 2 + jnp.log(2 * jnp.pi * self.scale**2))


def _factorization():
    latents = (
        Latent("depth", init=-0.5, prior=_Normal(0.0, 1.0)),
        Latent("width", init=1.0, prior=_Normal(0.0, 2.0)),
    )
    space = ParameterSpace(
        latents=latents,
        bindings=tuple(
            Bind(latent.name, into=lambda p, n=latent.name: getattr(p, n))
            for latent in latents
        ),
    )
    return Factorization(space)


def _term(epoch_id, key=None, rows=1):
    names, shapes = ("depth", "width"), ((), ())
    if key is None:
        factor, target = jnp.zeros((rows, 2)), jnp.zeros(rows)
    else:
        kf, kt = jax.random.split(key)
        factor = jax.random.normal(kf, (rows, 2))
        target = jax.random.normal(kt, (rows,))
    return QuadraticLikelihood(
        info=SqrtInfo(factor=factor, target=target, offset=jnp.array(0.0),
                      names=names, shapes=shapes),
        epoch_id=epoch_id, n_observed=1024,
    )


def test_an_empty_memory_returns_exactly_the_prior():
    memory = BayesMemory(_factorization())
    values = {"depth": jnp.array(0.3), "width": jnp.array(-1.1)}
    expected = float(_Normal(0.0, 1.0).log_prob(0.3) + _Normal(0.0, 2.0).log_prob(-1.1))
    assert float(memory.log_posterior(values)) == pytest.approx(expected, abs=1e-12)
    assert float(memory.log_likelihood(values)) == 0.0


@pytest.mark.parametrize("n_epochs", [1, 2, 10, 1000])
def test_null_terms_leave_the_posterior_exactly_equal_to_the_prior(n_epochs):
    """The prior-count pin: N null terms must not multiply the prior N times."""
    memory = BayesMemory(_factorization())
    for index in range(n_epochs):
        memory = memory.remember(_term(f"null-{index}"))
    values = {"depth": jnp.array(0.3), "width": jnp.array(-1.1)}
    expected = float(_Normal(0.0, 1.0).log_prob(0.3) + _Normal(0.0, 2.0).log_prob(-1.1))
    assert float(memory.log_posterior(values)) == pytest.approx(expected, abs=1e-10)


def test_remembering_accumulates_the_log_likelihood():
    terms = [_term(f"n{i}", key=jax.random.key(i)) for i in range(5)]
    memory = BayesMemory(_factorization())
    for term in terms:
        memory = memory.remember(term)
    values = {"depth": jnp.array(0.3), "width": jnp.array(-1.1)}
    assert float(memory.log_likelihood(values)) == pytest.approx(
        sum(float(term(values)) for term in terms), abs=1e-10
    )


def test_the_same_night_cannot_be_remembered_twice_by_accident():
    """...and the remedy offered is the **bag's**, which a chain refuses.

    The rule is shared with `ChainMemory.remember` and the last sentence is not:
    a bag can be told to count a night twice, a chain cannot, because its repeat
    would land last and reorder the campaign as well. `reject_bad_term` takes
    that sentence as an argument for exactly this reason, so it is pinned on
    both sides -- swapping the two remedies over must fail a test here as well
    as one in `test_chain_memory.py`, and until this assertion existed it
    failed only there.
    """
    memory = BayesMemory(_factorization()).remember(_term("night-001", jax.random.key(0)))
    with pytest.raises(StateValidationError, match="already") as caught:
        memory.remember(_term("night-001", jax.random.key(1)))
    assert "Pass duplicate=True" in str(caught.value)


def test_a_deliberate_duplicate_is_allowed_and_doubles_the_information():
    term = _term("night-001", jax.random.key(0))
    once = BayesMemory(_factorization()).remember(term)
    twice = once.remember(term, duplicate=True)
    assert float(np.trace(np.asarray(twice.fisher().matrix))) == pytest.approx(
        2 * float(np.trace(np.asarray(once.fisher().matrix))), rel=1e-10
    )


def test_two_estimators_cannot_be_summed():
    memory = BayesMemory(_factorization()).remember(_term("a", jax.random.key(0)))
    gls = QuadraticLikelihood(
        info=_term("b", jax.random.key(1)).info,
        epoch_id="b", n_observed=1024, include_logdet=False,
    )
    with pytest.raises(StateValidationError, match="estimator"):
        memory.remember(gls)


def test_a_term_over_different_latents_is_refused():
    memory = BayesMemory(_factorization())
    other = QuadraticLikelihood(
        info=SqrtInfo(factor=jnp.zeros((1, 1)), target=jnp.zeros(1),
                      offset=jnp.array(0.0), names=("depth",), shapes=((),)),
        epoch_id="x", n_observed=1,
    )
    with pytest.raises(StateValidationError, match="different latents"):
        memory.remember(other)


def test_a_term_carrying_a_prior_share_is_refused_on_the_streaming_path():
    memory = BayesMemory(_factorization())
    tempered = QuadraticLikelihood(
        info=_term("a", jax.random.key(0)).info,
        epoch_id="a", n_observed=1024, prior_share=(1, 300),
    )
    with pytest.raises(StateValidationError, match="prior-free"):
        memory.remember(tempered)


def test_the_treedef_does_not_change_as_epochs_are_remembered():
    """The compile-cost guard: one trace for a whole campaign, not one per epoch."""
    memory = BayesMemory(_factorization())
    first = jax.tree.structure(memory.accumulated)
    for index in range(20):
        memory = memory.remember(_term(f"n{index}", jax.random.key(index)))
    assert jax.tree.structure(memory.accumulated) == first


def test_the_archive_costs_one_pytree_leaf_however_long_the_campaign():
    """The O(N) guard on every method call, pinned structurally rather than by clock.

    Equinox wraps every non-magic bound method as a ``BoundMethod``, which is
    itself a ``Module`` *with* a dataclass ``__init__``, and that constructor
    flattens ``(args, kwargs)`` -- ``self`` included -- purely to check each leaf
    for a jax-transformed function. So the cost of ``memory.anything(...)`` is
    linear in the memory's leaf count, whether or not the method reads the
    archive. Measured before the fix, with three leaves per stored term:
    ``log_likelihood`` took 1.43 / 3.18 / 7.24 ms at 1,000 / 2,000 / 4,000
    epochs, against 0.08 ms for ``accumulated.log_prob`` on the same values.
    That is the sampling path -- a NUTS chain pays it once per leapfrog step.

    Leaf count is the mechanism, so leaf count is what is asserted. A wall-clock
    assertion measures the same defect but fails on a loaded machine for reasons
    that have nothing to do with this package; the scaling test below is
    deliberately given an order of magnitude of slack for that reason, and this
    one is exact.
    """
    memory = BayesMemory(_factorization())
    empty = len(jax.tree_util.tree_leaves(memory))
    for index in range(8):
        memory = memory.remember(_term(f"n{index}", jax.random.key(index)))
    eight = len(jax.tree_util.tree_leaves(memory))
    for index in range(8, 128):
        memory = memory.remember(_term(f"n{index}", jax.random.key(index)))
    # Not "bounded" but literally unchanged: the empty memory already carries the
    # one archive leaf, and 120 more epochs add nothing to it.
    assert len(jax.tree_util.tree_leaves(memory)) == eight == empty
    # The archive is one leaf, and it is still a plain tuple to every reader.
    assert isinstance(memory.archive, tuple)
    assert len(memory.archive) == 128
    assert memory.archive[3].epoch_id == "n3"


def test_remembering_a_late_epoch_costs_what_an_early_one_did():
    """The same claim in seconds, because leaf count is a proxy and cost is the point.

    Deliberately generous: the arithmetic under a ``remember`` is one
    ``SqrtInfo.combine``, flat at 0.405 ms/epoch on this repo, and the defect
    this pins added 0.03 ms per already-stored epoch on top -- 10x the
    arithmetic by epoch 2,000, and nine minutes of pure bookkeeping by 10,000.
    A factor of four therefore separates the two regimes by a wide margin while
    leaving room for a busy machine.
    """
    memory = BayesMemory(_factorization())
    term = _term("warmup", jax.random.key(0))

    def _span(start, stop):
        nonlocal memory
        while len(memory.archive) < start:
            memory = memory.remember(
                _term(f"n{len(memory.archive)}", jax.random.key(0))
            )
        begin = time.perf_counter()
        while len(memory.archive) < stop:
            memory = memory.remember(
                _term(f"n{len(memory.archive)}", jax.random.key(0))
            )
        return (time.perf_counter() - begin) / (stop - start)

    memory.remember(term)  # pay the first-call tracing cost outside the timing
    early = _span(50, 150)
    late = _span(1900, 2000)
    assert late < 4.0 * early, f"{late * 1e3:.3f} ms/epoch late vs {early * 1e3:.3f} early"


def test_fisher_is_named_and_refuses_to_pretend_to_be_a_covariance():
    memory = BayesMemory(_factorization()).remember(_term("a", jax.random.key(0), rows=2))
    fisher = memory.fisher()
    assert fisher.kind == "fisher"
    assert fisher.names == ("depth", "width")
    with pytest.raises(StateValidationError, match="not a standard deviation"):
        fisher.sigma("depth")


def test_fisher_spans_index_the_flat_vector_not_the_latent_list():
    """A (4,) latent covers four columns, not one -- and after the permutation.

    ``spans`` is what ``block`` and ``sigma`` slice with, so an off-by-one here
    reports a neighbouring parameter's number under this parameter's name --
    finite, correctly shaped, and wrong. Pinned against a non-scalar latent
    because every other test in this file uses scalars, where a span-per-latent
    bug and a span-per-element bug agree.

    The latents are declared ``("depth", "alms")`` and flatten as
    ``("alms", "depth")``, so this also pins the two properties together: the
    spans must be measured per ELEMENT and in FLATTEN order. Declared order
    would put ``depth`` first and give ``alms`` the span ``(1, 5)``.
    """
    latents = (
        Latent("depth", init=-0.5, prior=_Normal(0.0, 1.0)),
        Latent("alms", init=jnp.zeros(4), prior=_Normal(0.0, 2.0)),
    )
    space = ParameterSpace(
        latents=latents,
        bindings=tuple(
            Bind(latent.name, into=lambda p, n=latent.name: getattr(p, n))
            for latent in latents
        ),
    )
    fisher = BayesMemory(Factorization(space)).fisher()
    assert fisher.names == ("alms", "depth")
    assert fisher.spans == ((0, 4), (4, 5))
    assert fisher.matrix.shape == (5, 5)
    assert fisher.span("alms") == (0, 4)
    assert fisher.block("alms").shape == (4, 4)


def test_audit_reports_the_epoch_count_and_the_conditioning():
    memory = BayesMemory(_factorization())
    for index in range(4):
        memory = memory.remember(_term(f"n{index}", jax.random.key(index), rows=2))
    report = memory.audit()
    assert report["n_epochs"] == 4
    assert report["estimator"] == ("full", "none")
    assert report["prior_shares_sum"] == 0
    assert np.isfinite(report["fisher_lambda_min"])
    assert np.isfinite(report["fisher_condition"])


def test_fisher_is_permuted_into_flatten_order_not_declared_order():
    """The bug the fixture above cannot see, because its names are alphabetical.

    ``SqrtInfo``'s columns follow DECLARED order; ``FlatMatrix`` carries the
    treedef its rows were flattened against, and jax sorts a dict's keys. When
    a space is declared in non-alphabetical order the two disagree, and an
    unpermuted matrix would describe itself with a ``structure`` its numbers do
    not have. Per-latent information is 9 and 49 here so the two orders are
    told apart by value, not only by position.

    Every other test in this file declares ``("depth", "width")`` -- already
    sorted -- where the permutation is the identity and a missing permutation
    is indistinguishable from a correct one.
    """
    latents = (
        Latent("width", init=1.0, prior=_Normal(0.0, 2.0)),
        Latent("depth", init=-0.5, prior=_Normal(0.0, 1.0)),
    )
    space = ParameterSpace(
        latents=latents,
        bindings=tuple(
            Bind(latent.name, into=lambda p, n=latent.name: getattr(p, n))
            for latent in latents
        ),
    )
    factorization = Factorization(space)
    assert factorization.global_names == ("width", "depth"), "declared order"

    memory = BayesMemory(factorization).remember(
        QuadraticLikelihood(
            info=SqrtInfo(
                factor=jnp.asarray([[3.0, 0.0], [0.0, 7.0]]),
                target=jnp.zeros(2),
                offset=jnp.array(0.0),
                names=factorization.global_names,
                shapes=factorization.global_shapes,
            ),
            epoch_id="e",
            n_observed=1,
        )
    )
    fisher = memory.fisher()

    assert fisher.names == ("depth", "width")
    assert fisher.structure == jax.tree.structure(
        {"width": jnp.zeros(()), "depth": jnp.zeros(())}
    )
    # width carried 3**2, depth 7**2; after the permutation depth leads.
    np.testing.assert_allclose(np.diag(np.asarray(fisher.matrix)), [49.0, 9.0])
    assert fisher.span("width") == (1, 2)


def test_the_permuted_fisher_composes_with_propagate_covariance():
    """What the permutation is actually for, run end to end.

    Unpermuted, this raised StateValidationError reporting
    ``{'width': (), 'depth': ()}`` against ``{'depth': (), 'width': ()}`` -- a
    shape disagreement between two identical shapes, which is not a diagnosis
    anyone can act on.
    """
    latents = (
        Latent("width", init=1.0, prior=_Normal(0.0, 2.0)),
        Latent("depth", init=-0.5, prior=_Normal(0.0, 1.0)),
    )
    space = ParameterSpace(
        latents=latents,
        bindings=tuple(
            Bind(latent.name, into=lambda p, n=latent.name: getattr(p, n))
            for latent in latents
        ),
    )
    factorization = Factorization(space)
    memory = BayesMemory(factorization).remember(
        QuadraticLikelihood(
            info=SqrtInfo(
                factor=jnp.asarray([[3.0, 0.0], [0.0, 7.0]]),
                target=jnp.zeros(2),
                offset=jnp.array(0.0),
                names=factorization.global_names,
                shapes=factorization.global_shapes,
            ),
            epoch_id="e",
            n_observed=1,
        )
    )
    covariance = parameter_covariance(memory.fisher())
    params = {"width": jnp.array(1.0), "depth": jnp.array(-0.5)}

    # A prediction sensitive to `width` alone, so its propagated width is
    # exactly sigma(width) = 1/3 and picking up `depth` instead would give 1/7.
    sigma = propagate_covariance(lambda p: jnp.stack([p["width"]]), params, covariance)
    np.testing.assert_allclose(np.asarray(sigma), [1.0 / 3.0], rtol=1e-12)


class TestATermMissingWhatTheMemoryReadsIsRefusedAtTheDoor:
    """The protocol used to promise less than BayesMemory relies on.

    Declaring only ``latents``/``epoch_id``/``estimator``/``__call__`` made
    ``CompressedLikelihood`` a narrower claim than the code: a term satisfying
    it passed ``isinstance``, passed ``remember``, contributed to the
    accumulated density, and then raised ``AttributeError: 'X' object has no
    attribute 'n_observed'`` out of ``audit()``. Accumulation is a QR, so by
    then the campaign already depended on the term irreversibly.
    """

    class _Minimal:
        """Satisfies the OLD contract exactly, and nothing more."""

        latents = ("depth", "width")
        epoch_id = "minimal"
        estimator = ("full", "none")

        def __init__(self, info):
            self.info = info

        def __call__(self, values):
            return self.info.log_prob(values)

    def test_the_protocol_now_names_every_member_the_memory_reads(self):
        from rheplicant.inference.compressed import (
            REQUIRED_TERM_MEMBERS,
            CompressedLikelihood,
        )

        assert isinstance(_term("a"), CompressedLikelihood)
        # The old-contract term must NOT satisfy the published protocol.
        info = _term("a").info
        assert not isinstance(self._Minimal(info), CompressedLikelihood)
        for member in REQUIRED_TERM_MEMBERS:
            assert hasattr(_term("a"), member), member

    def test_remember_refuses_it_before_it_reaches_the_accumulator(self):
        memory = BayesMemory(_factorization())
        term = self._Minimal(_term("a").info)
        with pytest.raises(StateValidationError, match="n_observed"):
            memory.remember(term)

    def test_and_the_memory_is_unchanged_by_the_refusal(self):
        """A QR is irreversible, so the refusal has to happen before it."""
        memory = BayesMemory(_factorization())
        before = np.asarray(memory.accumulated.factor).copy()
        with pytest.raises(StateValidationError):
            memory.remember(self._Minimal(_term("a").info))
        np.testing.assert_array_equal(np.asarray(memory.accumulated.factor), before)
        assert memory.archive == ()
