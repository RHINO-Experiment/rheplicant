"""Uncertainty propagation through differentiable forward models.

Two complementary routes, both riding on the framework's differentiability:

- **Linear (Fisher / delta-method)** — the domain-standard forecasting tool.
  :func:`fisher_information` builds ``F = J^T N^-1 J`` from the exact
  Jacobian of ``forward`` (``jax.jacfwd`` — no finite differences), so
  parameter forecasts and error bars are one linear solve away
  (:func:`parameter_covariance`), and :func:`propagate_covariance` pushes a
  parameter covariance to a per-sample prediction standard deviation
  (delta method). Exact for models linear in the parameters; a local
  approximation otherwise.

  ``F = J^T N^-1 J`` is the LIKELIHOOD's information and nothing else, which
  is a different quantity from the posterior precision the other exits target.
  Pass ``space=`` a :class:`~rheplicant.inference.parameters.ParameterSpace`
  and the declared Gaussian priors' curvature is added, so a forecast and a
  NUTS run over one declaration answer the same question; leave it out and the
  result says ``kind="fisher"``, meaning exactly what it says.

- **Monte Carlo pushforward** — :func:`push_forward` vmaps ``forward`` over
  a stack of parameter samples (e.g. a NumPyro posterior via
  :func:`~rheplicant.inference.numpyro_bridge.predict_from_samples`), giving the
  full predictive distribution with no linearity assumption.

A Laplace approximation is the composition of the two: MAP-fit with
:class:`~rheplicant.inference.calibrate.GradientCalibrator`, take
:func:`parameter_covariance` at the fit, sample from the Gaussian, and
:func:`push_forward`.

``forward`` here is the ``f(params) -> prediction`` callable produced by
:func:`~rheplicant.inference.forward.build_forward_fn` — uncertainty tooling
connects through the same seam as every other inference engine (D7).
"""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from rheplicant.core.errors import ParameterSpaceError, StateValidationError
from rheplicant.inference.noise import (
    FlaggedNoise,
    HomoscedasticNoise,
    NoiseModel,
    check_noise_std_axis,
)
from rheplicant.inference.parameters import ParameterSpace


def _named_spans(
    params: Any,
) -> tuple[tuple[str, ...] | None, tuple[tuple[int, int], ...] | None, tuple[tuple, ...] | None]:
    """Map each name of a flat parameter dict to its span in the flat vector.

    Derived from the actual flattening rather than from an assumption about
    dict ordering. Returns ``(None, None, None)`` for anything that is not a
    flat ``{name: array}`` dict — the pytrees ``build_forward_fn`` produces,
    which have no names to give.
    """
    if not isinstance(params, dict):
        return None, None, None
    entries = jax.tree_util.tree_flatten_with_path(params)[0]
    names: list[str] = []
    spans: list[tuple[int, int]] = []
    shapes: list[tuple] = []
    offset = 0
    for path, leaf in entries:
        if len(path) != 1 or not isinstance(path[0], jax.tree_util.DictKey):
            return None, None, None  # nested — no flat naming to offer
        size = int(jnp.size(leaf))
        names.append(str(path[0].key))
        spans.append((offset, offset + size))
        shapes.append(jnp.shape(leaf))
        offset += size
    return tuple(names), tuple(spans), tuple(shapes)


# What each kind reports, and what inverting it gives. A precision-like kind is
# one whose sqrt(diag(.)) is not an error bar, which is the only thing `sigma`
# has to branch on; keeping the two facts in one table means a new kind cannot
# be added to one of them and forgotten in the other.
_PRECISION_KINDS: dict[str, str] = {
    "fisher": "a Fisher matrix",
    "posterior_precision": "a posterior precision matrix",
}
_INVERSE_KIND: dict[str, str] = {
    "fisher": "covariance",
    "posterior_precision": "posterior_covariance",
}

#: This package's ``kind`` spelled in the far side's vocabulary.
#:
#: The far side knows three kinds and this package needs four, because
#: "posterior" is a fact about the matrix that :meth:`FlatMatrix.sigma` reports
#: and the far side has no use for: the only distinction it branches on is
#: whether the thing IS a covariance, and both of this package's covariance
#: spellings are one. So this is a translation rather than a widening --
#: nothing sayable here is lost, and nothing the far side can do becomes
#: reachable through it.
#:
#: Anything else -- ``"matrix"``, which is what :class:`FlatMatrix` calls a
#: hand-built one, or a kind from a future version of this module -- crosses as
#: a precision, through the lookup's default rather than through a row of its
#: own. That is what :func:`parameter_covariance` has always done with such a
#: matrix: inverted it and called the result a covariance. Every FlatMatrix
#: this package builds declares ``kind="fisher"`` explicitly (measured:
#: `memory.py`, `reduced_basis.py`, `fisher_information`), so a row for
#: ``"matrix"`` would be a second spelling of the default with nothing able to
#: reach it and nothing able to tell the two apart.
_REMOTE_KIND: dict[str, str] = {
    "fisher": "fisher",
    "posterior_precision": "posterior_precision",
    "covariance": "covariance",
    "posterior_covariance": "covariance",
}


class FlatMatrix(eqx.Module):
    """A matrix over a FLATTENED parameter vector, carrying its provenance.

    ``ravel_pytree`` ordering depends on the parameter pytree's structure, so
    a Fisher/covariance matrix is only meaningful together with the treedef
    it was flattened against. Carrying the structure lets
    :func:`propagate_covariance` reject a covariance built for a different
    parameterization instead of silently returning wrong numbers.

    When the parameters came from a
    :class:`~rheplicant.inference.parameters.ParameterSpace` — a flat
    ``{name: array}`` dict — the rows also carry their names, so error bars can
    be asked for by the name the model was declared in
    (``cov.sigma("fwhm_deg")``) rather than by position.

    Attributes:
        matrix: the ``(n_params, n_params)`` array.
        structure: treedef of the parameter pytree it was computed for.
        kind: which quantity this is. Not decoration, twice over.

            ``"fisher"`` and ``"posterior_precision"`` are precisions:
            ``sqrt(diag(.))`` is not an error bar, so :meth:`sigma` refuses to
            pretend otherwise on either.

            ``"covariance"`` and ``"posterior_covariance"`` are the two things
            :meth:`sigma` can report, and they are different quantities. The
            first is the Cramer-Rao bound — the width the DATA alone supports,
            with no prior in it. The second includes the declared
            ``Latent(prior=...)`` curvature and is what a NUTS chain over the
            same space is a sample from. Comparing the two as if they were one
            quantity is the failure this field exists to make visible: under an
            informative prior they can differ by orders of magnitude, and both
            are finite, correctly shaped and plausible.

            ``"matrix"`` is the default for anything constructed by hand.
        names, spans, shapes: per-parameter name, ``(start, stop)`` span in the
            flat vector, and original shape. ``None`` for unnamed pytrees.
    """

    matrix: jax.Array
    structure: jax.tree_util.PyTreeDef = eqx.field(static=True)
    kind: str = eqx.field(static=True, default="matrix")
    names: tuple[str, ...] | None = eqx.field(static=True, default=None)
    spans: tuple[tuple[int, int], ...] | None = eqx.field(static=True, default=None)
    shapes: tuple[tuple, ...] | None = eqx.field(static=True, default=None)

    def _index(self, name: str) -> int:
        if self.names is None:
            raise StateValidationError(
                "This matrix's parameters are not named — it was built from a plain pytree. "
                "Build the forward function with ParameterSpace.forward_fn to get named rows."
            )
        if name not in self.names:
            raise StateValidationError(
                f"There is no parameter named {name!r}; available: {list(self.names)}."
            )
        return self.names.index(name)

    def span(self, name: str) -> tuple[int, int]:
        """``(start, stop)`` of one parameter within the flattened vector."""
        return self.spans[self._index(name)]

    def sigma(self, name: str) -> jax.Array:
        """Marginal standard deviation(s) of one named parameter.

        Only meaningful on a covariance: raises on either precision kind rather
        than returning ``sqrt(diag(.))``, which looks like an error bar and is
        not one — inverting is exactly the step that couples the parameters.

        What it reports depends on :attr:`kind` and the caller has to know
        which: on a ``"covariance"`` it is the Cramer-Rao bound from the
        likelihood alone, on a ``"posterior_covariance"`` it is the posterior
        width including the declared priors.
        """
        if self.kind in _PRECISION_KINDS:
            raise StateValidationError(
                f"sqrt(diag(.)) of {_PRECISION_KINDS[self.kind]} is not a standard "
                "deviation — it ignores every parameter degeneracy. Invert it with "
                "parameter_covariance() first."
            )
        index = self._index(name)
        start, stop = self.spans[index]
        return jnp.sqrt(jnp.diag(self.matrix)[start:stop]).reshape(self.shapes[index])

    def block(self, name: str, other: str | None = None) -> jax.Array:
        """The sub-matrix for one parameter, or the cross-block of two."""
        start, stop = self.span(name)
        other_start, other_stop = self.span(other) if other is not None else (start, stop)
        return self.matrix[start:stop, other_start:other_stop]


def _flat_forward(
    forward: Callable[[Any], jax.Array], params: Any
) -> tuple[Callable[[jax.Array], jax.Array], jax.Array, jax.Array]:
    """Flatten the parameter pytree: return (f_flat, x0, prediction0)."""
    complex_leaves = [
        path
        for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]
        if jnp.issubdtype(jnp.result_type(leaf), jnp.complexfloating)
    ]
    if complex_leaves:
        where = ", ".join(jax.tree_util.keystr(p) for p in complex_leaves)
        raise StateValidationError(
            f"Complex parameters ({where}) cannot go through jax.jacfwd, which is what the "
            "Fisher and delta-method routines are built on. A real prediction makes the map "
            "R-linear but not C-linear, so the Jacobian is only defined over the real degrees "
            "of freedom: split the latent into real and imaginary parts, or use "
            "rheplicant.inference.linear (linear_operator / wiener_solve / gcr_sample), which "
            "does that split internally."
        )
    x0, unravel = ravel_pytree(params)
    if x0.size == 0:
        raise StateValidationError(
            "params contains no trainable leaves — nothing to propagate."
        )

    def f_flat(x: jax.Array) -> jax.Array:
        return jnp.ravel(forward(unravel(x)))

    return f_flat, x0, forward(params)


def as_noise_model(
    noise_std: Any,
    flags: jax.Array | None = None,
    *,
    prediction_shape: Any | None = None,
    caller: str = "as_noise_model",
) -> NoiseModel:
    """Normalize a ``noise_std`` argument into a :class:`NoiseModel`.

    A bare scalar or array becomes :class:`HomoscedasticNoise`; a noise model
    is passed through; ``flags`` wrap either in :class:`FlaggedNoise`. This is
    what lets every ``noise_std=`` argument in the package accept the seam
    without any of their signatures changing.

    The discrimination is by ``depends_on_prediction``, not by ``std``: jax and
    numpy arrays both *have* a ``.std`` method, so the protocol's data member
    is the only unambiguous marker.

    Args:
        noise_std: a scalar, an array, or a :class:`NoiseModel`.
        flags: optional boolean mask; ``True`` = not observed.
        prediction_shape: the shape the model predicts, when the caller knows
            it. Supplying it turns on
            :func:`~rheplicant.inference.noise.check_noise_std_axis`, which
            refuses a 1-D sigma whose axis the prediction cannot settle.
            Optional because this function is also called from inside a NumPyro
            model body, *before* the prediction exists; omitting it is the old
            behaviour exactly.
        caller: the exit to name if that check refuses.
    """
    if prediction_shape is not None:
        check_noise_std_axis(noise_std, prediction_shape, caller)
    noise = (
        noise_std
        if isinstance(noise_std, NoiseModel)
        else HomoscedasticNoise(jnp.asarray(noise_std))
    )
    return noise if flags is None else FlaggedNoise(noise, flags)


def _declared_gaussian_priors(
    space: ParameterSpace,
    names: tuple[str, ...] | None,
    spans: tuple[tuple[int, int], ...] | None,
    shapes: tuple[tuple, ...] | None,
) -> dict[str, Any]:
    """The declared prior per latent, once every admission check has passed.

    **G15's deferral is discharged, and this is what is left of it.** The
    arithmetic — ``diag(1/sigma^2)`` laid out over the flat vector — is gone.
    The far side computes it from the graph's own nodes, and the two agreed to
    ``0.0e+00``, not merely to a tolerance, on the fixture in
    ``docs/probes/probe_16_g15_discharge.py`` before the old spelling was
    deleted.

    What stays is the ADMISSION, and it stays under a rule the plan states
    generally rather than as an exception carved for this function: **any
    refusal whose evidence the graph seam would erase lives in a
    pre-validation, before the graph is built**
    (``2026-08-26-one-implementation.md`` §三; its first named instance is
    :func:`~rheplicant.inference.noise.check_noise_std_axis`).

    The seam erases exactly this one. :func:`~rheplicant.inference.graph_bridge
    .translate` sorts bayesmith's refusals into three families, and
    ``NotGaussian`` is the blameless third: *caught and not re-raised*, left on
    the yielded ``Seam``, because a caller asking "is there an exact route
    here?" in order to branch should not have to write ``except`` around a
    question. ``fisher_information(space=...)`` is not that caller — for it a
    Uniform prior is an error, and three tests pin it as one, by name.
    **Measured**: with the prior moved onto the graph and the check left to the
    far side, a ``Uniform`` ends the ``with`` block early, and the next line
    reads ``.values`` off a name that was never assigned. The caller is
    promised a ``ParameterSpaceError`` naming the latent and gets an
    ``UnboundLocalError`` naming nothing.

    Two of the five checks below have no counterpart on the far side at all:
    a ``joint_prior`` (bayesmith has never heard of
    :class:`~rheplicant.inference.priors.JeffreysPrior`), and an unnamed params
    pytree (the far side's block always has names). The other three would
    arrive as bayesmith's wording about "the exact linear-Gaussian path" —
    a true sentence, about a different question.

    For a Gaussian ``N(m, s)`` the log-density's curvature is ``1/s^2``,
    independent of where it is evaluated — the one prior family whose
    contribution to the information is a constant matrix and not a function of
    the expansion point. That is exactly the family
    :func:`~rheplicant.inference.linear.wiener_solve` calls conjugate, so
    reading the SAME declaration here is what keeps the Fisher forecast and the
    conjugate solve talking about one posterior.

    Anything else is refused rather than approximated. Substituting a
    Uniform's variance would report a crisp Gaussian posterior for a prior with
    no curvature at all, narrower than the truth wherever the declared prior is
    bounded — the identical failure the conjugate exits already refuse, in the
    identical words.

    **The two sides' Gaussian rules differ on exactly one spelling**, and the
    returned mapping is canonical rather than verbatim because of it. Measured:
    bayesmith's ``check_gaussian`` accepts a ``Normal`` and an ``Independent``
    (``.to_event(1)``) and refuses an ``ExpandedDistribution`` (``.expand([2])``);
    :func:`~rheplicant.inference.linear._gaussian_parameters` unwraps all three,
    because both wrappers only re-shape a base distribution. Handing the graph
    the declaration as written would therefore admit ``.expand([2])`` here and
    have it refused — invisibly, per the paragraph above — one call later. So
    what crosses is ``Normal(loc, scale)`` broadcast to the latent's own shape:
    one form, agreed by both sides, for every spelling that gets this far. The
    ``Latent`` keeps what the user wrote; this is the ``priors=`` entry point
    synthesising a ``Normal``, which is the mechanism the plan's construction
    table already names for a prior that has no graph-side spelling.

    The broadcast is not decoration either. A ``Normal(0.0, 0.5)`` declared for
    a ``(2,)`` latent would reach the far side with batch shape ``()``, and
    ``Latent.__check_init__`` refuses that shape mismatch at construction — so
    it cannot arrive from a ``ParameterSpace``. ``.expand([2])`` is the same
    prior with the shape supplied, and it can.

    Returns:
        ``{name: Normal}``, ready for
        :func:`~rheplicant.inference.graph_bridge.graph_for_information`'s
        ``priors=`` — that function's single entry point for a declared prior.
    """
    # Imported where it is used rather than at module scope: `linear` is the
    # module that owns what "a Gaussian prior ON THE LATENT" means (it refuses
    # LogNormal, which carries .loc/.scale and a Normal .base_dist while being
    # Gaussian in log x), and a function-local import keeps the two modules
    # free to depend on each other in the other direction later.
    from rheplicant.inference.linear import (
        _gaussian_parameters,
        _numpyro_distributions,
    )

    if space.joint_prior is not None:
        raise ParameterSpaceError(
            f"fisher_information was given space= a ParameterSpace declaring "
            f"{type(space.joint_prior).__name__}(over={list(space.joint_prior.over)}), and "
            "`space=` means 'add the declared priors' curvature to this matrix'. That "
            "prior is DEFINED as sqrt(det of this matrix), so adding it would put it "
            "inside its own definition: the result is not the posterior precision it "
            "would be labelled as, and it is finite, symmetric and positive definite, so "
            "nothing downstream would say otherwise. Call this with space=None — which "
            "is the likelihood Fisher the prior is built from, and is what "
            "JeffreysPrior.log_density itself passes — or read the posterior by sampling "
            "the space with to_numpyro_model + NUTS, which is the exit that evaluates a "
            "joint prior."
        )

    if names is None or spans is None or shapes is None:
        raise StateValidationError(
            "fisher_information was given a ParameterSpace, but these params are not "
            "named — they came from a plain pytree, so there is no span to add each "
            "latent's prior curvature at and placing it by position would be a guess. "
            "Build the forward function with ParameterSpace.forward_fn, whose params "
            "are the {name: array} dict this matrix's rows are named from."
        )
    if tuple(sorted(names)) != tuple(sorted(space.names)):
        raise ParameterSpaceError(
            f"fisher_information was given params named {list(names)} but a "
            f"ParameterSpace declaring {list(space.names)} — the two do not match, so "
            "some latent's prior would be added at another latent's span, or dropped. "
            "Pass the params that ParameterSpace.forward_fn returned for this space."
        )

    distributions = _numpyro_distributions()
    declared: dict[str, Any] = {}
    for name, shape in zip(names, shapes, strict=True):
        prior = space.latent(name).prior
        if prior is None:
            raise ParameterSpaceError(
                f"fisher_information was given a ParameterSpace in which latent "
                f"{name!r} declares no prior, so what it returns would be a posterior "
                "precision for every latent but that one — a matrix that is part "
                "likelihood and part posterior, reported under a single name. A "
                "prior-free latent is a free parameter: fine for the calibrators and "
                "for the likelihood Fisher (drop `space=` and you get exactly that), "
                "meaningless in a posterior. Declare Latent(prior=dist.Normal(...)), "
                "or ask for the likelihood matrix and say so."
            )
        gaussian = _gaussian_parameters(prior)
        if gaussian is None:
            raise ParameterSpaceError(
                f"fisher_information was given a ParameterSpace, but latent {name!r} "
                f"declares a {type(prior).__name__} prior, which has no quadratic form. "
                "A Fisher matrix with a prior in it is -d2 log p/dtheta2 summed over "
                "likelihood and prior, and the second derivative of this prior is not a "
                "constant matrix; substituting the distribution's mean and variance "
                "would return a finite, confident posterior precision for a prior you "
                "did not declare — narrower than the truth wherever the declared prior "
                "is skewed or bounded. Sample this space with to_numpyro_model + NUTS "
                "instead, which honours the prior as written, or drop `space=` and read "
                "the result as the likelihood Fisher it then is."
            )
        loc, scale = gaussian
        declared[name] = distributions.Normal(
            jnp.broadcast_to(jnp.asarray(loc), shape),
            jnp.broadcast_to(jnp.asarray(scale), shape),
        )
    return declared


#: The name the flat pytree path gives its single latent. Double-underscored
#: for the reason `graph_bridge`'s own node names are: it is not something a
#: caller declared, and it must not collide with one that was.
_FLAT_LATENT: str = "__flat__"


def _remote_flat(matrix: FlatMatrix) -> Any:
    """This package's :class:`FlatMatrix` as the far side's, over the same rows.

    Three fields cross unchanged and two are translated. ``names`` and
    ``spans`` are optional here and required there, because a matrix on this
    side can have been built over a pytree with no names to give; such a matrix
    crosses as the ONE latent :func:`fisher_information` gives it, over
    ``ravel_pytree``'s own vector -- which is the layout :class:`FlatMatrix`
    documents in either case, so the rows line up without a permutation.

    ``kind`` is the other, through :data:`_REMOTE_KIND`.

    ``shapes`` and ``structure`` do not cross at all: they are how THIS package
    checks a matrix against a pytree, and the far side checks the same thing
    against a graph's own declarations instead.
    """
    from bayesmith.exact.fisher import FlatMatrix as RemoteFlat

    size = matrix.matrix.shape[0]
    return RemoteFlat(
        values=matrix.matrix,
        names=matrix.names if matrix.names is not None else (_FLAT_LATENT,),
        spans=matrix.spans if matrix.spans is not None else ((0, size),),
        kind=_REMOTE_KIND.get(matrix.kind, "fisher"),
    )


def _bayesmith_fisher(graph, block_names, values, noise, *, include_prior):
    """The far side's Fisher over ``block_names``, read from the graph.

    The noise reaches it TWICE and the two are not the same argument -- that
    function says so itself. ``precision`` is the decided operator that weights
    the design; ``precision_of`` is the RULE, whose derivative is the variance's
    own information. An operator has no derivative, so a caller holding only
    the first cannot supply the second, and one holding only the second has not
    decided anything yet.

    Both are read from this graph at these values, so ``centre`` is the same
    point by construction -- and it is passed anyway, because the far side
    cross-checks the two rather than trusting them, on the grounds that an
    unchecked redundancy is how a covariance ends up weighted at one point and
    curved at another.
    """
    from bayesmith.diagnose.local import local_block
    from bayesmith.exact.fisher import fisher_information as remote_fisher
    from bayesmith.exact.gaussian import precision_at

    return remote_fisher(
        # `priors=` is G15's third constructor, and it is a keyword rather
        # than a second function because the default is unchanged. It must
        # track `include_prior`: the prior curvature is read off the BLOCK's
        # `prior_std`, so asking for it with a prior-free block raises
        # `KeyError` on the first member -- measured, not inferred.
        local_block(graph, block_names, values, priors=include_prior),
        precision=precision_at(graph, values),
        include_prior=include_prior,
        depends_on_prediction=bool(noise.depends_on_prediction),
        precision_of=lambda moving: precision_at(graph, {**values, **moving}),
        centre={name: values[name] for name in block_names},
    )


def fisher_information(
    forward: Callable[[Any], jax.Array],
    params: Any,
    noise_std: Any,
    flags: jax.Array | None = None,
    *,
    space: ParameterSpace | None = None,
) -> jax.Array:
    """Fisher information at ``params`` — **likelihood-only unless given a space**.

    With ``space=None`` (the default, and what this function has always done)
    the matrix is ``F = J^T N^-1 J``: the information the DATA carries, and
    nothing else. It is not a posterior precision, and its inverse is a
    Cramer-Rao bound rather than an error bar you could compare with a NUTS
    posterior run under informative priors. That distinction used to be
    invisible, and it mattered: ``Latent(prior=...)`` is the package's one
    statement of what a latent is a priori and every other exit reads it —
    :func:`~rheplicant.inference.linear.wiener_solve` solves with it as ``S``
    and refuses a prior-free linear latent by name — while this function never
    saw the :class:`~rheplicant.inference.parameters.ParameterSpace` at all.
    Tightening a declared prior by a factor of 5,000,000 moved the reported
    error bar by exactly zero.

    Pass ``space=`` and the declared Gaussian priors' own curvature is added at
    each latent's span, giving the **posterior precision** at ``params``; the
    result is tagged ``kind="posterior_precision"`` and its inverse
    ``kind="posterior_covariance"``, so which quantity was reported survives
    into the object rather than living in the caller's memory.

    Args:
        forward: ``f(params) -> prediction``.
        params: where to evaluate.
        noise_std: standard deviation — a scalar, or an array whose axes say
            which axis of the prediction it runs along (``(n, 1)`` / ``(1, n)``;
            see :func:`~rheplicant.inference.noise.check_noise_std_axis`) —
            **or** a :class:`~rheplicant.inference.noise.NoiseModel`.
        flags: optional boolean mask; flagged samples carry zero weight, the
            same convention as
            :class:`~rheplicant.inference.likelihood.MaskedGaussianLikelihood`.
        space: the declaration ``params`` was built from. Optional, and its
            absence is a real answer rather than a missing argument — the
            likelihood Fisher is the standard forecasting quantity. When given,
            every latent must declare a Gaussian prior: a prior-free one, or one
            with no quadratic form (a Uniform, a Half-Normal, a LogNormal),
            raises :class:`~rheplicant.core.errors.ParameterSpaceError` by name
            rather than being approximated away. A space declaring a
            ``joint_prior`` is refused outright: a
            :class:`~rheplicant.inference.priors.JeffreysPrior` is defined as
            ``sqrt(det I)``, so it cannot be a term inside ``I``.

    Returns:
        A :class:`FlatMatrix` — the ``(n_params, n_params)`` matrix
        (``.matrix``) over the flattened parameter vector, tagged with the
        parameter structure it belongs to and with ``kind`` saying which
        quantity it is.

    Note:
        **When the noise depends on the parameters, ``J^T N^-1 J`` is not the
        Fisher matrix.** For ``d ~ N(mu(theta), Sigma(theta))`` the information
        has a second term from the covariance's own parameter dependence::

            F = J^T Sigma^-1 J  +  1/2 tr(Sigma^-1 dSigma Sigma^-1 dSigma)

        which for a diagonal covariance is ``2 (d log sigma/d theta)^T
        (d log sigma/d theta)``. It is included automatically whenever the
        noise model reports ``depends_on_prediction``, and omitted otherwise
        (where it is exactly zero). Under
        :class:`~rheplicant.inference.noise.RadiometerNoise` with fractional
        level ``f`` the correction is a clean factor: ``F = (1 + 2 f^2)
        J^T N^-1 J``. Reporting only the first term would forecast error bars
        that are too wide by ``sqrt(1 + 2 f^2)`` — a plausible number, and the
        wrong one.
    """
    from rheplicant.inference.graph_bridge import graph_for_information, translate

    f_flat, x0, prediction = _flat_forward(forward, params)
    noise = as_noise_model(
        noise_std,
        flags,
        prediction_shape=jnp.shape(prediction),
        caller="fisher_information",
    )
    names, spans, shapes = _named_spans(params)

    # BEFORE the graph, not after: `translate` files bayesmith's `NotGaussian`
    # as a blameless verdict and does NOT re-raise it, so a prior checked on
    # the far side would end the `with` block early instead of refusing. This
    # is P1's general rule -- a refusal whose evidence the seam would erase
    # lives in a pre-validation -- and `_declared_gaussian_priors` is where it
    # lives for this exit.
    declared = (
        None
        if space is None
        else _declared_gaussian_priors(space, names, spans, shapes)
    )

    # A flat `{name: array}` dict becomes one node per latent, which is what
    # lets the declared priors reach the far side as declarations. Anything
    # else -- a bare array, a pipeline pytree -- has no names to give
    # (`_named_spans` says so), so it crosses as ONE latent over
    # `ravel_pytree`'s own vector. That is exactly the ordering `FlatMatrix`
    # documents, so the layout is the same either way; and `space=` cannot
    # reach it, because `_declared_gaussian_priors` just above has already
    # refused an unnamed params by name.
    if names is None:
        block_names = (_FLAT_LATENT,)
        values = {_FLAT_LATENT: x0}
        graph_forward = lambda v: f_flat(v[_FLAT_LATENT])  # noqa: E731
    else:
        # SORTED, because that is this package's flat layout -- `_named_spans`
        # derives it from `ravel_pytree`, which orders a dict by sorted key.
        # Handing the far side the same order means no permutation afterwards:
        # it lays the block out in the order it is given (D24 is the same fact
        # seen from the one call site that cannot choose).
        block_names = tuple(sorted(names))
        values = {name: jnp.asarray(params[name]) for name in block_names}
        graph_forward = forward

    graph = graph_for_information(
        graph_forward, values, noise, priors=declared, caller="fisher_information"
    )
    with translate("fisher_information"):
        found = _bayesmith_fisher(
            graph, block_names, values, noise, include_prior=space is not None
        )
    return FlatMatrix(
        matrix=found.values,
        structure=jax.tree_util.tree_structure(params),
        # Derived rather than re-spelled. The far side already tags which
        # quantity it returned, and it tags it from the same flag that decided
        # it; an `if space is not None` here would be a second copy of that
        # rule, and the one that goes stale.
        kind=found.kind,
        names=names,
        spans=spans,
        shapes=shapes,
    )


def parameter_covariance(fisher: FlatMatrix, jitter: float = 0.0) -> FlatMatrix:
    """Invert a Fisher matrix (or a posterior precision) into a covariance.

    Inversion does not change *which* quantity is being inverted, so ``kind``
    is carried across rather than reset: a likelihood Fisher gives a
    ``"covariance"`` — the Cramer-Rao bound, what the data alone can do — and a
    ``"posterior_precision"`` gives a ``"posterior_covariance"``, whose
    :meth:`FlatMatrix.sigma` is a posterior width comparable with a NUTS chain
    run under the same declaration. The two are different numbers and used to
    come back under the same label.

    Args:
        fisher: output of :func:`fisher_information`.
        jitter: optional Tikhonov term added to the diagonal for
            near-degenerate parameter combinations (prior-like regularizer).
            Note what it is on a likelihood Fisher: an undeclared Gaussian
            prior of width ``1/sqrt(jitter)``, chosen for numerical comfort
            rather than declared. ``fisher_information(..., space=...)`` is the
            same regularization with the prior written down.

    Raises:
        StateValidationError: if the condition number exceeds what the values'
            own dtype can carry, or if this is handed a covariance. Both
            refusals are decided across the seam and re-raised here in this
            package's class; see the notes below.

    Note:
        **This inversion IS gated on conditioning, and the gate is the far
        side's.** ``F = J^T N^-1 J`` SQUARES the design's condition number, so
        an ordinary model reaches the arithmetic's limit: measured at
        ``kappa(J) = 1e3``, the float32 covariance is 2.4% wrong while the
        float64 one is wrong by 1.08e-12, and neither used to say so. The
        ceiling is ``1/sqrt(eps)`` read from the values' own dtype (float32:
        2.90e+03, float64: 6.71e+07) -- the point where inverting has spent
        half the digits available -- and it arrives with this function's
        delegation rather than being written a second time here (D29).

        Note what the remedy is NOT: wrapping this call in ``with
        jax.enable_x64(True):`` recovers nothing. The context does not widen
        an array traced outside it, and even forcing the upcast leaves the
        error at 2.45e-02 against 2.41e-02 for doing nothing, because the
        digits were spent forming ``F``. The arithmetic has to be widened
        around building the model, which is what the refusal's own message
        says.

    Note:
        **Why the refusals are caught by class and not by ``translate``.**
        This is the one delegation in the module that reaches no graph, so
        none of the three families ``translate`` knows about can arise on this
        path -- the far side's only refusals here are its own two plain
        ``ValueError``s. Catching ``ValueError`` and re-raising is therefore
        narrow rather than broad: it names the exit a caller actually called,
        and it keeps the exception class this module promises, which is the
        same reason ``numpyro_bridge`` refuses ahead of a bare
        ``AssertionError`` from the far side.
    """
    from bayesmith.exact.fisher import parameter_covariance as remote

    try:
        found = remote(_remote_flat(fisher), jitter)
    except ValueError as refusal:
        raise StateValidationError(
            f"parameter_covariance() will not invert this {fisher.kind!r}: "
            f"{refusal}"
        ) from refusal
    return FlatMatrix(
        matrix=found.values,
        structure=fisher.structure,
        kind=_INVERSE_KIND.get(fisher.kind, "covariance"),
        names=fisher.names,
        spans=fisher.spans,
        shapes=fisher.shapes,
    )


def propagate_covariance(
    forward: Callable[[Any], jax.Array],
    params: Any,
    param_cov: jax.Array,
) -> jax.Array:
    """Delta-method prediction uncertainty: ``std = sqrt(diag(J Sigma J^T))``.

    Args:
        forward: ``f(params) -> prediction``.
        params: expansion point (pytree, same structure as the covariance's
            flattening).
        param_cov: covariance over the flattened parameter vector — a
            :class:`FlatMatrix` from :func:`parameter_covariance` (structure
            is verified against ``params``), or a raw ``(n_params, n_params)``
            array (external covariances; only the size can be checked — YOU
            must guarantee the flattening order matches).

    Returns:
        Per-sample prediction standard deviation, shaped like the prediction.

    Raises:
        StateValidationError: for a covariance whose provenance does not match
            ``params`` (three checks, all of them ahead of the seam because the
            graph would erase what they are about), or for a PRECISION -- the
            same table :meth:`FlatMatrix.sigma` refuses on, and the same
            remedy. A Fisher matrix and a covariance are the same shape, so
            putting one where the other belongs returns an error bar wrong by
            the square of everything and says nothing.

    Note:
        **The graph this builds synthesises a noise model and data, and
        neither can reach the answer.** The delta method reads the Jacobian of
        the prediction and the covariance it was handed; it does not read the
        residual, and it does not weight anything. So a homoscedastic sigma of
        1.0 and zeros of the prediction's shape are enough to give the far
        side a graph to differentiate -- the same argument D22 makes for the
        rank test, and, like that one, MEASURED rather than asserted:
        ``TestTheSynthesisedGraphCannotReachTheAnswer`` builds it three times
        over different synthetic sigmas and data and compares the reports
        bitwise.
    """
    from rheplicant.inference.graph_bridge import graph_for_information, translate

    f_flat, x0, prediction = _flat_forward(forward, params)
    names, spans, shapes = _named_spans(params)
    if isinstance(param_cov, FlatMatrix):
        expected = jax.tree_util.tree_structure(params)
        if param_cov.structure != expected:
            raise StateValidationError(
                f"param_cov was computed for parameter structure "
                f"{param_cov.structure}, but params has structure {expected} — "
                "the flattened orderings differ and the numbers would be wrong."
            )
        # For a dict-based space the treedef encodes the KEY NAMES only, so two
        # spaces with the same latent names and different per-latent shapes pass
        # the structure check and produce finite, wrong error bars.
        if param_cov.shapes is not None and shapes is not None:
            if param_cov.names != names or param_cov.shapes != shapes:
                was = dict(zip(param_cov.names, param_cov.shapes, strict=True))
                now = dict(zip(names, shapes, strict=True))
                raise StateValidationError(
                    f"param_cov was computed for {was} but params is {now} — the "
                    "flattened orderings differ and the numbers would be wrong."
                )
        if param_cov.kind in _PRECISION_KINDS:
            raise StateValidationError(
                f"propagate_covariance was given {_PRECISION_KINDS[param_cov.kind]}, "
                "not a covariance. The delta method wants Sigma, and a precision "
                "is its INVERSE — the same shape, so the error bar comes back "
                "finite and wrong by the square of everything. Invert it with "
                "parameter_covariance() first."
            )
        param_cov = param_cov.matrix
    if param_cov.shape != (x0.size, x0.size):
        raise StateValidationError(
            f"param_cov shape {param_cov.shape} does not match the flattened "
            f"parameter size {x0.size}."
        )

    # One node per named latent, or ONE node over `ravel_pytree`'s own vector
    # when the pytree has no names to give -- the same two routes, and the same
    # layout, as `fisher_information`. The matrix crosses with the layout the
    # PARAMS decide, not the one it arrived carrying: the guards above have
    # already refused any disagreement between them, and the graph is built
    # from the params, so taking both from one place is what keeps the rows
    # lined up with the columns.
    if names is None:
        values = {_FLAT_LATENT: x0}
        graph_forward = lambda v: f_flat(v[_FLAT_LATENT])  # noqa: E731
    else:
        values = {name: jnp.asarray(params[name]) for name in names}
        graph_forward = forward
    crossing = FlatMatrix(
        matrix=param_cov,
        structure=jax.tree_util.tree_structure(params),
        kind="covariance",
        names=names,
        spans=spans,
        shapes=shapes,
    )
    graph = graph_for_information(
        graph_forward,
        values,
        HomoscedasticNoise(jnp.ones((), jnp.result_type(float))),
        caller="propagate_covariance",
    )
    from bayesmith.exact.fisher import propagate_covariance as remote

    with translate("propagate_covariance"):
        return remote(graph, _remote_flat(crossing), values)


def push_forward(
    forward: Callable[[Any], jax.Array],
    param_samples: Any,
) -> jax.Array:
    """Monte Carlo pushforward: run ``forward`` over stacked parameter samples.

    Args:
        forward: ``f(params) -> prediction``.
        param_samples: a params pytree whose every array leaf carries a
            leading sample axis of common length ``n_samples``.

    Returns:
        ``(n_samples, *prediction.shape)`` stacked predictions — summarize
        with e.g. ``mean(0)`` / ``std(0)`` / quantiles.
    """
    return jax.vmap(forward)(param_samples)
