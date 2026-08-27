"""Joint priors — a prior over a BLOCK of latents, declared on the space.

``Latent(prior=...)`` says what one quantity is a priori and is the only prior
declaration the rest of the package needs. A Jeffreys prior is not of that
shape: it is a single density over several latents at once, it is a function of
the forward model and the noise rather than of the latent alone, and it moves
when the model does. So it is declared once, on the
:class:`~rheplicant.inference.parameters.ParameterSpace`, where D14's rule
applies — the declaration is what every exit reads::

    ParameterSpace(
        latents=[...],
        bindings=[...],
        joint_prior=JeffreysPrior(over=("fg_log_amp", "fg_beta")),
    )

Default ``None``. Nothing here is on unless it is asked for.
"""

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from rheplicant.core.errors import ParameterSpaceError

#: How many null directions a refusal names before it says "and N more".
_DIRECTIONS_SHOWN: int = 3


def _as_names(names: Any) -> tuple[str, ...]:
    """"One or many" for latent names — careful not to explode a string.

    ``over="fg_beta"`` must mean one latent, not four characters. The same
    convention :class:`~rheplicant.inference.parameters.Bind` uses for its
    ``latents`` and :func:`~rheplicant.inference.identifiability.identifiability`
    for its ``names``; spelled again here rather than imported, because
    ``parameters`` imports this module and the cycle would not resolve.
    """
    if isinstance(names, str):
        return (names,)
    if isinstance(names, Sequence):
        return tuple(names)
    raise ParameterSpaceError(
        f"JeffreysPrior(over={names!r}) — `over` takes a latent name or a sequence of "
        "them, and it is mandatory: there is no default block, because the conditional "
        "Jeffreys prior of one block and the full-space one are different priors."
    )


def _bayesmith_prior(prior: "JeffreysPrior") -> Any:
    """This declaration as the far side's, carrying both static fields.

    Built per call rather than cached on the module: an ``eqx.Module`` is
    frozen, and a lazily attached attribute would be a second piece of state
    with its own staleness question. The object is two static fields.
    """
    import bayesmith

    return bayesmith.JeffreysPrior(over=prior.over, rank_rtol=prior.rank_rtol)


def _rows_in_sorted_order(
    matrix: jax.Array, over: tuple[str, ...], values: Any
) -> jax.Array:
    """Permute a block matrix from ``over``'s order into ``sorted(over)``'s.

    D24's execution. The far side lays the block out in the order ``over``
    names; this package has always laid it out by sorted key, because
    :func:`~rheplicant.inference.uncertainty.fisher_information` flattens that
    way. A symmetric permutation leaves the determinant alone -- so the prior
    is the same prior either way -- and moves every row, so a caller reading
    row 0 gets a different number. On the tour's own block those two readings
    differ by 7.4e+1.

    Each latent occupies a span the size of its own value, so the permutation
    is over spans and not over names: a two-element latent and a scalar do not
    swap as single rows.
    """
    sizes = {name: int(jnp.size(jnp.asarray(values[name]))) for name in over}
    start, spans = 0, {}
    for name in over:
        spans[name] = (start, start + sizes[name])
        start += sizes[name]
    order = jnp.concatenate(
        [jnp.arange(*spans[name]) for name in sorted(over)]
    ) if over else jnp.arange(0)
    return matrix[jnp.ix_(order, order)]


class JeffreysPrior(eqx.Module):
    r"""``p(theta) = sqrt(det I(theta))`` over a named block, conditional on the rest.

    **Read this first: under the package's default noise model this prior is
    the flat prior.** Under
    :class:`~rheplicant.inference.noise.RadiometerNoise`, for a bare power law
    ``mu = A (nu/nu0)^-beta`` over ``(log A, beta)``, the half-log-determinant
    measured on the 8x8 fixture of ``tests/inference/test_jeffreys_prior.py``
    is **+15.80169853 at every one of the nine grid points** ``log A in {6.8, 7.8,
    8.8} x beta in {2.05, 2.55, 3.05}`` — identical to the last printed digit —
    and its gradient there is ``(0.000000e+00, -1.387779e-17)``. Switching it on
    for that model is the same thing as deleting the ``prior=`` keyword, at the
    cost of a Jacobian per leapfrog step. The algebra says why, and says it
    exactly rather than approximately: ``sigma = |mu| f`` gives ``N^-1 = 1 /
    (mu^2 f^2)`` while ``J_{k,i} = mu_k g_i(nu_k)``, so every ``mu`` cancels and
    ``I_ij = (1 + 2 f^2) / f^2 * sum_k g_i(nu_k) g_j(nu_k)`` — a constant matrix
    in ``(log A, beta)``.

    **Under** :class:`~rheplicant.inference.noise.HomoscedasticNoise` **the same
    block gives** ``p(log A) proportional to A^2``: the half-log-determinant is
    exactly linear in ``log A`` with slope **+2.000000** measured over six
    decades, ``log A in [-3, +3]``. That is improper *upward* — the density
    grows without bound as the amplitude does — so it is a prior that needs the
    likelihood to be doing the work, and it is not a neutral choice.

    Those two sentences are the same prior under two noise models, which is the
    thing to take from them: **the noise model chooses the prior's shape.** On
    the same power law with a fixed 300 K floor added, where the radiometer
    variance no longer factorises, ``d(half-logdet)/d beta`` is
    ``-1.366854e-02`` under ``RadiometerNoise`` and ``+8.052944e-03`` under
    ``HomoscedasticNoise`` — opposite signs, on one model, from the noise
    declaration alone. This is also why the prior carries **no noise model of
    its own**: it is evaluated with the noise the exit was given, so a
    likelihood/prior noise mismatch is not something this API can express.

    Attributes:
        over: the latents the prior is over — a name, or a sequence of them.
            Mandatory and explicit. **It is the CONDITIONAL Jeffreys prior of
            that block**, with every other latent held at whatever value the
            exit currently has, and the conditional and the full-space priors
            are different priors. They are not interchangeable and the
            full-space one often does not exist: on the tour's own 34-latent
            space :func:`~rheplicant.inference.identifiability.identifiability`
            reports rank 32 of 34, so ``det I`` is zero over the full space and
            ``sqrt(det I)`` is not a density at all. Naming the block is
            therefore not boilerplate — it is the part of the declaration that
            decides whether the object exists.
        rank_rtol: the relative cut separating a null eigenvalue from a small
            one. ``None`` (the default) means
            :data:`~rheplicant.inference.identifiability.DEFAULT_RANK_RTOL`, read
            from the one place in this package where that number is justified
            against a measured spectrum.

    **Why the determinant comes from** ``eigh`` **and not from the two obvious
    routes.** On the exactly degenerate block of
    ``tests/inference/test_jeffreys_prior.py`` — a 3-latent space whose
    amplitude is ``exp(a + b)``, so ``a`` and ``b`` are the same parameter
    twice — measured:

    =========================  =============================================
    route                      what it returns
    =========================  =============================================
    ``jnp.linalg.slogdet``     sign ``+1.0``, half-logdet ``+6.420496``
    ``jnp.linalg.cholesky``    all-finite, half-logdet ``+6.566517``,
                               smallest pivot ``9.755e-05`` (positive, so
                               it did not fail)
    ``eigvalsh`` + rank floor  ``-338.05`` — effectively zero density
    =========================  =============================================

    Both of the first two came back with a plausible, finite, positive-definite
    answer for a matrix that is singular by construction, because the null
    eigenvalue lands at ``-2.117e-09`` against a largest of ``1.281e+08`` and
    the sign of that roundoff is a coin flip. **A determinant that came back
    finite is not a guard.** So the eigenvalues are taken explicitly, everything
    at or below ``rank_rtol * max`` is floored to the smallest positive number
    the dtype has, and the result is the ``-338`` that says what is true.

    That floor is the *arithmetic*; the *refusal* is :meth:`check_identified`,
    which every exit reading this declaration calls once at build time and which
    delegates its verdict to
    :func:`~rheplicant.inference.identifiability.identifiability` — an SVD of
    the column-normalised Jacobian, which does not square the condition number
    the way ``J^T N^-1 J`` does, and which already knows how to name a
    degeneracy as a combination of latents.

    **Which exits read it.**
    :func:`~rheplicant.inference.numpyro_bridge.to_numpyro_model` evaluates it:
    each covered latent gets an improper flat site and the half-log-determinant
    is added at the ``"joint_prior"`` factor site, with the same noise object
    the likelihood uses.
    :func:`~rheplicant.inference.uncertainty.fisher_information` refuses
    ``space=`` outright while one is declared — a Jeffreys prior defined as
    ``sqrt(det I)`` cannot be added to ``I``.
    :class:`~rheplicant.inference.plan.SamplingPlan` refuses a block partition
    that splits ``over`` across two blocks. That last one is a check on the
    declaration and not an evaluation: a plan's engines build their conditional
    potentials from ``Latent(prior=...)``, so the NUTS route through
    ``to_numpyro_model`` is the route that applies this prior today.
    """

    over: tuple[str, ...] = eqx.field(static=True, converter=_as_names)
    rank_rtol: float | None = eqx.field(static=True, default=None)

    def __check_init__(self):
        if not self.over:
            raise ParameterSpaceError(
                "JeffreysPrior(over=()) is over no latents, so its information matrix is "
                "0x0, its determinant is the empty product 1, and it would contribute a "
                "flat zero to every posterior while reading as a declared prior. Name the "
                "block: over=('fg_log_amp', 'fg_beta')."
            )
        wrong = [name for name in self.over if not isinstance(name, str)]
        if wrong:
            raise ParameterSpaceError(
                f"JeffreysPrior(over=...) takes latent NAMES, got {wrong}. The block is "
                "declared over the names a ParameterSpace uses, not over Latent objects."
            )
        repeated = sorted({name for name in self.over if self.over.count(name) > 1})
        if repeated:
            raise ParameterSpaceError(
                f"JeffreysPrior(over={list(self.over)}) lists {repeated} more than once. "
                "Two copies of one latent are exactly degenerate with each other, so the "
                "information matrix would be singular by construction and this prior "
                "would not exist — for a reason that says nothing about the model."
            )
        if self.rank_rtol is not None and not self.rank_rtol > 0.0:
            raise ParameterSpaceError(
                f"JeffreysPrior(rank_rtol={self.rank_rtol!r}) — the rank tolerance is a "
                "positive relative cut. rank_rtol=0 keeps every eigenvalue including the "
                "roundoff ones, which is exactly the slogdet behaviour this prior does "
                "not use: on an exactly degenerate block it returns a plausible finite "
                "number for a density that does not exist."
            )

    # ------------------------------------------------------------- reading --

    @property
    def rank_tolerance(self) -> float:
        """:attr:`rank_rtol`, or identifiability's default when it is ``None``."""
        if self.rank_rtol is not None:
            return float(self.rank_rtol)
        # Function-local: `identifiability` imports `parameters`, which imports
        # this module, so the constant cannot be read at import time. It is read
        # from there rather than re-declared because that is where the number is
        # justified against a measured spectrum.
        from rheplicant.inference.identifiability import DEFAULT_RANK_RTOL

        return float(DEFAULT_RANK_RTOL)

    def covers(self, name: str) -> bool:
        """Whether this prior is the prior on latent ``name``."""
        return name in self.over

    @property
    def label(self) -> str:
        """How a message names this block."""
        return "(" + ", ".join(repr(name) for name in self.over) + ")"

    # ------------------------------------------------------------ refusing --

    def validate_against(
        self, declared: Sequence[str], with_prior: Sequence[str]
    ) -> None:
        """Check ``over`` against a space's latents. Raises, or returns ``None``.

        Called by :meth:`~rheplicant.inference.parameters.ParameterSpace.__check_init__`
        so the wording lives with the class that owns the concept.

        Args:
            declared: every latent name the space declares.
            with_prior: those of them that also carry ``Latent(prior=...)``.
        """
        unknown = [name for name in self.over if name not in declared]
        if unknown:
            raise ParameterSpaceError(
                f"JeffreysPrior(over={list(self.over)}) names {unknown}, which this space "
                f"does not declare; its latents are {list(declared)}. The block would be "
                "assembled from the names that DO match and the prior would be a "
                "different prior from the one written down — over a smaller block, which "
                "is a different density and not a subset of the same one. Every other "
                "latent would then be sampled flat with nothing saying so."
            )
        doubled = [name for name in self.over if name in with_prior]
        if doubled:
            raise ParameterSpaceError(
                f"Latent(s) {doubled} are covered by JeffreysPrior(over={list(self.over)}) "
                "AND declare their own Latent(prior=...). That is two priors on one "
                "quantity: the posterior would be multiplied by both, which is a proper "
                "density and a plausible chain and not the model either declaration "
                "describes — and no diagnostic reports a prior counted twice, because "
                "each one on its own is correct. Drop the Latent(prior=...) for the "
                "latents the joint prior covers, or take them out of `over`."
            )

    def check_identified(
        self,
        space: Any,
        pipeline: Any,
        state_template: Any,
        *,
        at: dict[str, jax.Array] | None = None,
        caller: str = "This JeffreysPrior",
    ) -> Any:
        """Refuse a block whose information matrix is rank-deficient.

        ``sqrt(det I)`` with ``det I = 0`` is not a density: it is zero
        everywhere the rank is deficient, which is everywhere, so there is no
        prior to normalise and nothing for a sampler to explore. The verdict is
        delegated to
        :func:`~rheplicant.inference.identifiability.identifiability`, which
        takes the rank of the column-normalised Jacobian rather than of
        ``J^T N^-1 J`` — the Jacobian's own condition number, not its square —
        and which reports the degenerate direction as a share of each latent.

        Args:
            space: the declaration (this prior's own space).
            pipeline: the forward model.
            state_template: the state it is evaluated on.
            at: where to ask. Identifiability is a LOCAL property of a nonlinear
                model; ``None`` means the space's declared initial values, which
                is where a build-time check can ask.
            caller: what to name in the message.

        Returns:
            The :class:`~rheplicant.inference.identifiability.IdentifiabilityReport`,
            when the block is identified.

        Raises:
            ParameterSpaceError: naming the nullity and the latents each null
                direction mixes.
        """
        from rheplicant.inference.identifiability import identifiability

        report = identifiability(
            space,
            pipeline,
            state_template,
            names=self.over,
            at=at,
            rtol=self.rank_tolerance,
        )
        if report.nullity == 0:
            return report

        lines = []
        for index in range(min(report.nullity, _DIRECTIONS_SHOWN)):
            share = report.participation(index)
            carried = sorted(share.items(), key=lambda item: -item[1])
            lines.append(
                f"  direction {index}: "
                + ", ".join(f"{name} {value:.2f}" for name, value in carried if value > 1e-3)
            )
        more = report.nullity - len(lines)
        if more > 0:
            lines.append(f"  ... and {more} more")

        raise ParameterSpaceError(
            f"{caller} is declared over {self.label}, whose Jacobian has nullity "
            f"{report.nullity} of {report.n_par} parameters — so det I is zero and "
            "sqrt(det I) is not a density: it is zero everywhere, there is nothing to "
            "normalise, and a sampler would be exploring a potential with an arbitrary "
            "additive constant along the null directions. Nothing downstream would say "
            "so: on a block degenerate by construction, jnp.linalg.slogdet returned "
            "sign +1.0 with a half-log-determinant of +6.420496 and jnp.linalg.cholesky "
            "succeeded with a positive smallest pivot of 9.755e-05. The degenerate "
            "directions, as shares of each latent:\n"
            + "\n".join(lines)
            + "\nRe-parameterize the block, drop one of its latents from over=, or give "
            "the degenerate combination a Latent(prior=...) of its own. "
            f"identifiability(space, pipeline, state, names={list(self.over)}) reports "
            "the same thing in full."
        )

    # ---------------------------------------------------------- evaluating --

    def information(
        self,
        forward: Callable[[dict[str, jax.Array]], jax.Array],
        values: dict[str, jax.Array],
        noise_std: Any,
        flags: jax.Array | None = None,
    ) -> jax.Array:
        """The CONDITIONAL information matrix over ``over``, at ``values``.

        Every latent not in ``over`` is held at its entry in ``values`` — that
        is what makes this the conditional prior and not the full-space one.

        ``space=None`` is passed to
        :func:`~rheplicant.inference.uncertainty.fisher_information` and is
        load-bearing: with ``space=`` that function returns the POSTERIOR
        precision, likelihood Fisher plus declared prior curvature, and a
        Jeffreys prior built from it would sit inside its own definition.
        :func:`~rheplicant.inference.uncertainty.fisher_information` refuses the
        combination outright for the same reason.

        The second Fisher term — ``2 (d log sigma/d theta)^T (d log sigma/d
        theta)``, the information the variance carries when sigma tracks the
        prediction — comes along automatically, because that function already
        adds it whenever the noise model reports ``depends_on_prediction``. It
        is not decoration: it is half of why the radiometer prior for a bare
        power law is exactly flat, the ``(1 + 2 f^2)`` in ``I_ij = (1 + 2 f^2) /
        f^2 sum_k g_i g_j``.

        **Row order is ``sorted(over)``, not ``over``.**
        :func:`~rheplicant.inference.uncertainty.fisher_information` flattens by
        sorted key, so ``over=("b", "a")`` and ``over=("a", "b")`` return the
        IDENTICAL matrix. The determinant does not care — a symmetric
        permutation leaves it alone, which is why the prior itself is unaffected
        — but a caller reading row 0 as "the first name I passed" is wrong by
        7.4e+1 on the tour's own block.

        Args:
            forward: ``f(values) -> prediction`` over the FULL latent dict.
            values: every latent's current value.
            noise_std: whatever the exit was given — a scalar, an array, or a
                :class:`~rheplicant.inference.noise.NoiseModel`. This prior
                carries none of its own.
            flags: optional boolean mask, as elsewhere.
        """
        from rheplicant.inference.graph_bridge import graph_for_information, translate
        from rheplicant.inference.uncertainty import as_noise_model

        missing = [name for name in self.over if name not in values]
        if missing:
            raise ParameterSpaceError(
                f"JeffreysPrior over {self.label} was evaluated at a values dict with no "
                f"entry for {missing}; it has {sorted(values)}. The block would be built "
                "from the names that are present, which is a prior over a different "
                "block."
            )
        graph = graph_for_information(
            forward, values, as_noise_model(noise_std, flags)
        )
        with translate("JeffreysPrior.information"):
            matrix = _bayesmith_prior(self).information(graph, values)
        # Rows come back in sorted(over) order, not over= order -- D24. The far
        # side returns them in over='s order and says in its own docstring that
        # this package's sorted order is a wart that does not port. It is a
        # wart; it is also observable and pinned, so the FACADE permutes and
        # the difference is registered rather than taken silently. The
        # determinant is invariant either way, so the prior is untouched.
        return _rows_in_sorted_order(matrix, self.over, values)

    def log_density(
        self,
        forward: Callable[[dict[str, jax.Array]], jax.Array],
        values: dict[str, jax.Array],
        noise_std: Any,
        flags: jax.Array | None = None,
    ) -> jax.Array:
        """``0.5 * log det I`` over the block — the log prior, up to a constant.

        Jit-safe and differentiable, which is the whole requirement: NUTS
        differentiates it at every leapfrog step. It therefore cannot refuse a
        rank-deficient block, because a rank is a decision and a traced decision
        is one you cannot branch on — :meth:`check_identified` is that refusal
        and the exits call it once, before sampling. What this returns on a
        degenerate block is the floored value, ``-338.05`` on the measured fixture:
        an honest zero rather than the plausible ``+6.42`` ``slogdet`` gives.

        Arguments are :meth:`information`'s.
        """
        return self.half_log_determinant(
            self.information(forward, values, noise_std, flags)
        )

    def half_log_determinant(self, matrix: jax.Array) -> jax.Array:
        """``0.5 * log det`` by eigendecomposition, with the rank floor applied.

        Separate from :meth:`log_density` so a caller holding an information
        matrix already — or a test pinning this against ``slogdet`` and
        ``cholesky`` on the same array — can reach the arithmetic without
        re-differentiating the model.
        """
        return _bayesmith_prior(self).half_log_determinant(matrix)


__all__ = ["JeffreysPrior"]
