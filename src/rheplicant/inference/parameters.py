"""Parameter spaces: what gets inferred, and how it enters the forward model.

Two words carry the whole design, so they are worth defining precisely.

**Latent** — *a named quantity you infer.* It is the thing a sampler draws or
an optimizer steps: it has a name, an initial value (which fixes its shape and
dtype), optionally a prior, and optionally a declaration that it enters the
model linearly. A latent knows **nothing** about the pipeline. ``log_gain`` is
a latent; so is a 10⁴-element vector of sky ``alm`` coefficients.

**Bind** — *a rule turning latents into pipeline leaf values.* It names the
latents it consumes, the pipeline leaves it writes (``eqx.tree_at`` selectors),
and optionally the function between them. A bind knows nothing about priors.

The split is the point. A pipeline leaf is *what the instrument model holds*;
a latent is *what you chose to infer*. They are usually not the same object:
two scalars can determine a beam's whole harmonic expansion, one scalar can
drive several stages at once, and a positive quantity is best sampled in its
logarithm. Keeping the two apart means re-parameterizing never requires
editing the instrument description — which is the promise of D7.

Three shapes cover essentially everything:

.. code-block:: python

    ParameterSpace(
        latents=[
            Latent("fwhm_deg", init=12.0, prior=dist.Uniform(5.0, 30.0)),
            Latent("log_e",    init=0.0,  prior=dist.Normal(0.0, 0.3)),
            Latent("log_gain", init=0.0,  prior=dist.Normal(0.0, 0.1)),
            Latent("sky_alms", init=alms0, prior=..., linear=True),
        ],
        bindings=[
            # derived: two scalars -> one high-dimensional leaf
            Bind(("fwhm_deg", "log_e"),
                 into=lambda p: p["t_ant"]["sky"].projector.beam_alms,
                 fn=lambda f, e: gaussian_beam_alms(f, jnp.exp(e), lmax=LMAX)),
            # tied: one latent -> several leaves, through a positivity transform
            Bind("log_gain",
                 into=(lambda p: p["gain"].gain, lambda p: p["ref_gain"].gain),
                 fn=jnp.exp),
            # direct: straight into one leaf
            Bind("sky_alms", into=lambda p: p["t_ant"]["sky"].sky_model.alms),
        ],
    )

Anything the blocks cannot express goes through :meth:`ParameterSpace.raw`,
which takes a bind function outright.

Binding never changes the pipeline's pytree *structure* — only leaf values.
Every downstream consumer depends on that: ``eqx.filter_vmap`` over posterior
samples, ``ravel_pytree`` for Fisher matrices, and ``jit`` all assume a fixed
treedef. :meth:`ParameterSpace.validate` checks it.
"""

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from rheplicant.core.errors import ParameterSpaceError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State


def _as_tuple(x: Any) -> tuple:
    """Normalize "one or many" arguments — careful not to explode strings."""
    return tuple(x) if isinstance(x, (tuple, list)) else (x,)


def _dtype_kind(dtype: Any) -> str:
    """Coarse dtype class — strict enough to catch complex-into-real, loose
    enough to survive JAX's float32/float64 promotion rules."""
    if jnp.issubdtype(dtype, jnp.complexfloating):
        return "complex"
    if jnp.issubdtype(dtype, jnp.floating):
        return "float"
    if jnp.issubdtype(dtype, jnp.integer):
        return "integer"
    if jnp.issubdtype(dtype, jnp.bool_):
        return "boolean"
    return str(dtype)  # pragma: no cover - defensive


def _tag_leaves_with_paths(pipeline: AbstractOperator) -> AbstractOperator:
    """A copy of ``pipeline`` in which every leaf holds its own tree path.

    Running a selector against this tells us *which* leaf the selector picked,
    by identity of position rather than of value — so two stages sharing the
    same array object are still recognised as two distinct targets.
    """
    return jax.tree_util.tree_map_with_path(lambda path, _: path, pipeline)


class Latent(eqx.Module):
    """A named quantity to infer — the unit a sampler or optimizer works on.

    Deliberately ignorant of the pipeline: a latent says *what* is inferred,
    a :class:`Bind` says *where it goes*.

    Attributes:
        name: identifier. Doubles as the NumPyro sample-site name and the key
            in the ``dict`` a forward function consumes, so it should read
            like a physical quantity (``"fwhm_deg"``, ``"sky_alms"``).
        init: initial value — also the authority on shape and dtype. Plain
            Python numbers are converted to arrays.
        prior: a NumPyro distribution, or ``None``. ``None`` means a *free*
            parameter: usable by the optimizers, rejected by the Bayesian
            bridge (a parameter with no prior has no place in a posterior).
            Read by every inference exit, not only the sampler: a Gaussian
            declared here is the ``S`` that
            :func:`~rheplicant.inference.linear.wiener_solve` and
            :func:`~rheplicant.inference.linear.gcr_sample` solve with, so the
            two routes to a posterior cannot drift apart. A prior with no
            conjugate Gaussian form is fine — it is simply an error at those
            exits rather than silently ignored there.
        linear: assert that the prediction is an **affine** function of this
            latent, holding the others fixed. Unlocks
            :func:`~rheplicant.inference.linear.linear_operator` and the
            conjugate-Gaussian machinery built on it. The claim is checkable —
            see :func:`~rheplicant.inference.linear.check_linearity` — and is
            checked before it is exploited.
    """

    name: str = eqx.field(static=True)
    init: jax.Array = eqx.field(converter=jnp.asarray)
    prior: Any = None
    linear: bool = eqx.field(static=True, default=False)

    def __check_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ParameterSpaceError(f"Latent name must be a non-empty string, got {self.name!r}.")
        # Duck-typed so that declaring a space costs no numpyro import.
        prior_shape = getattr(self.prior, "shape", None) if self.prior is not None else None
        if callable(prior_shape):
            declared = tuple(prior_shape())
            if declared != self.init.shape:
                raise ParameterSpaceError(
                    f"Latent {self.name!r}: prior has shape {declared} but init has shape "
                    f"{self.init.shape}. The prior describes the latent, so the two must agree."
                )


class Bind(eqx.Module):
    """A rule turning latent values into pipeline leaf values.

    Attributes:
        latents: name, or tuple of names, of the latents this rule consumes.
            They are passed to ``fn`` positionally, in this order.
        into: an ``eqx.tree_at`` selector (``lambda p: p["gain"].gain``), or a
            tuple of them. Several selectors is how one latent drives several
            leaves.
        fn: ``latent values -> leaf value(s)``. ``None`` means identity, which
            requires exactly one latent. If ``fn`` returns a single array it is
            written to **every** selector in ``into`` (this is parameter
            tying); if it returns a tuple, its length must match ``into``.
    """

    latents: tuple[str, ...] = eqx.field(static=True, converter=_as_tuple)
    into: tuple[Callable, ...] = eqx.field(static=True, converter=_as_tuple)
    fn: Callable | None = eqx.field(static=True, default=None)

    def __check_init__(self):
        if not self.latents:
            raise ParameterSpaceError("Bind needs at least one latent name.")
        for name in self.latents:
            if not isinstance(name, str):
                raise ParameterSpaceError(f"Bind latent names must be strings, got {name!r}.")
        if not self.into:
            raise ParameterSpaceError("Bind needs at least one `into` selector.")
        for selector in self.into:
            if not callable(selector):
                raise ParameterSpaceError(
                    f"Bind `into` must hold callables (eqx.tree_at selectors), got {selector!r}."
                )
        if self.fn is None and len(self.latents) != 1:
            raise ParameterSpaceError(
                f"Bind for {self.latents} has no `fn`, so it is the identity — which takes "
                "exactly one latent. Supply `fn` to combine several."
            )

    def evaluate(self, values: dict[str, jax.Array]) -> tuple[jax.Array, ...]:
        """Produce one value per selector in ``into``."""
        args = [values[name] for name in self.latents]
        produced = self.fn(*args) if self.fn is not None else args[0]
        if isinstance(produced, (tuple, list)):
            if len(produced) != len(self.into):
                raise ParameterSpaceError(
                    f"Bind for {self.latents} returned {len(produced)} values but has "
                    f"{len(self.into)} `into` selectors."
                )
            return tuple(produced)
        return (produced,) * len(self.into)


class ParameterSpace(eqx.Module):
    """The declaration inference engines read: latents plus how they bind.

    Attributes:
        latents: the quantities to infer, in declaration order.
        bindings: how they reach the pipeline. Empty only when ``raw_bind``
            is supplied.
        raw_bind: escape hatch — ``(pipeline, values) -> pipeline``, used
            instead of compiling ``bindings``. Build one with
            :meth:`raw`.
    """

    latents: tuple[Latent, ...] = eqx.field(converter=tuple)
    bindings: tuple[Bind, ...] = eqx.field(static=True, converter=tuple, default=())
    raw_bind: Callable | None = eqx.field(static=True, default=None)

    def __check_init__(self):
        if not self.latents:
            raise ParameterSpaceError("A ParameterSpace needs at least one Latent.")
        names = [latent.name for latent in self.latents]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ParameterSpaceError(
                f"Latent names must be unique, repeated: {sorted(duplicates)}."
            )
        if self.raw_bind is None and not self.bindings:
            raise ParameterSpaceError(
                "A ParameterSpace needs bindings — otherwise its latents reach nothing. "
                "Use ParameterSpace.raw(...) to supply a bind function instead."
            )
        if self.raw_bind is not None and self.bindings:
            raise ParameterSpaceError(
                "A ParameterSpace takes a raw bind function INSTEAD of bindings, not as well: "
                "bind() would call raw_bind and never apply the declared Bind(s), so any latent "
                "only those bindings reach would be sampled without entering the model. Pass one "
                "or the other."
            )
        declared = set(names)
        for binding in self.bindings:
            unknown = [name for name in binding.latents if name not in declared]
            if unknown:
                raise ParameterSpaceError(
                    f"Bind references undeclared latent(s) {unknown}; declared: {sorted(declared)}."
                )
        bound = {name for binding in self.bindings for name in binding.latents}
        if self.raw_bind is None:
            dead = declared - bound
            if dead:
                raise ParameterSpaceError(
                    f"Latent(s) {sorted(dead)} are declared but never bound, so they would be "
                    "sampled without reaching the model — the posterior would just return the "
                    "prior. Bind them, or drop them."
                )

    # ------------------------------------------------------------ builders --

    @classmethod
    def raw(
        cls,
        latents: Sequence[Latent],
        bind: Callable[[AbstractOperator, dict[str, jax.Array]], AbstractOperator],
        bindings: Sequence[Bind] | None = None,
    ) -> "ParameterSpace":
        """Build a space from a bind function instead of declarative blocks.

        The escape hatch for parameterizations the blocks cannot express. The
        structural checks in :meth:`validate` still apply, so a bind function
        that quietly changes the pipeline's pytree structure is still caught.
        """
        if bindings:
            raise ParameterSpaceError(
                "ParameterSpace.raw takes a bind function INSTEAD of bindings, not as well. "
                "Pass one or the other."
            )
        return cls(latents=tuple(latents), bindings=(), raw_bind=bind)

    @classmethod
    def direct(
        cls,
        name: str,
        init: Any,
        into: Callable | tuple[Callable, ...],
        prior: Any = None,
        fn: Callable | None = None,
        linear: bool = False,
    ) -> "ParameterSpace":
        """One latent, one binding — the common case, in one call."""
        return cls(
            latents=(Latent(name, init=init, prior=prior, linear=linear),),
            bindings=(Bind(name, into=into, fn=fn),),
        )

    # ------------------------------------------------------------- reading --

    @property
    def names(self) -> tuple[str, ...]:
        """Latent names, in declaration order."""
        return tuple(latent.name for latent in self.latents)

    def latent(self, name: str) -> Latent:
        """Look a latent up by name."""
        for latent in self.latents:
            if latent.name == name:
                return latent
        raise ParameterSpaceError(f"No latent named {name!r}; declared: {list(self.names)}.")

    def initial_values(self) -> dict[str, jax.Array]:
        """The starting point: ``{name: init}``, ready for an optimizer."""
        return {latent.name: latent.init for latent in self.latents}

    # ------------------------------------------------------------ applying --

    def _abstract_values(self) -> dict[str, jax.ShapeDtypeStruct]:
        """The latents as shapes only — enough for every check, costs no compute."""
        return {
            latent.name: jax.ShapeDtypeStruct(latent.init.shape, latent.init.dtype)
            for latent in self.latents
        }

    def _resolve_targets(self, pipeline: AbstractOperator) -> list[tuple[Bind, int, tuple, Any]]:
        """Resolve every ``into`` selector to (binding, slot, tree path, leaf)."""
        tagged = _tag_leaves_with_paths(pipeline)
        leaves = dict(jax.tree_util.tree_flatten_with_path(pipeline)[0])
        resolved = []
        for binding in self.bindings:
            for slot, selector in enumerate(binding.into):
                try:
                    path = selector(tagged)
                except Exception as exc:
                    raise ParameterSpaceError(
                        f"Bind for {binding.latents}: `into` selector {slot} failed against the "
                        f"pipeline ({exc}). Selectors may only walk attributes and indices."
                    ) from exc
                if not isinstance(path, tuple) or path not in leaves:
                    raise ParameterSpaceError(
                        f"Bind for {binding.latents}: `into` selector {slot} does not reach an "
                        "array leaf of the pipeline. It landed on static configuration "
                        f"({path!r}), which inference cannot touch."
                    )
                if not eqx.is_array(leaves[path]):
                    raise ParameterSpaceError(
                        f"Bind for {binding.latents}: `into` selector {slot} reaches "
                        f"{type(leaves[path]).__name__}, not an array leaf."
                    )
                resolved.append((binding, slot, path, leaves[path]))
        return resolved

    def validate(self, pipeline: AbstractOperator) -> None:
        """Check this space against a pipeline. Raises, or returns ``None``.

        Every check runs on shapes alone (``jax.eval_shape``): no array is
        ever computed. It still *traces* the bindings, so a derived ``fn``
        doing real work costs one trace — negligible against a fit, but not
        literally zero. Called for you by :meth:`forward_fn` and by the
        Bayesian bridge, once per build rather than per evaluation.

        The failure modes it exists to prevent all share a shape: they produce a
        finite, correctly-shaped, **wrong** inference rather than an exception.
        """
        abstract = self._abstract_values()

        if self.raw_bind is None:
            resolved = self._resolve_targets(pipeline)

            seen: dict[tuple, Bind] = {}
            for binding, _, path, _ in resolved:
                if path in seen:
                    raise ParameterSpaceError(
                        f"Pipeline leaf {jax.tree_util.keystr(path)} is written by more than one "
                        f"binding ({seen[path].latents} and {binding.latents}); one would "
                        "silently win. Give each leaf a single binding."
                    )
                seen[path] = binding

            for binding in self.bindings:
                produced = jax.eval_shape(binding.evaluate, abstract)
                for slot, value in enumerate(produced):
                    target = next(
                        leaf for b, s, _, leaf in resolved if b is binding and s == slot
                    )
                    if value.shape != target.shape:
                        raise ParameterSpaceError(
                            f"Bind for {binding.latents} produces shape {value.shape} for "
                            f"`into` selector {slot}, but that leaf has shape {target.shape}."
                        )
                    if _dtype_kind(value.dtype) != _dtype_kind(target.dtype):
                        raise ParameterSpaceError(
                            f"Bind for {binding.latents} produces "
                            f"{_dtype_kind(value.dtype)} values for `into` selector {slot}, "
                            f"but that leaf is {_dtype_kind(target.dtype)}."
                        )

        bound = jax.eval_shape(lambda values: self.bind(pipeline, values), abstract)
        if jax.tree_util.tree_structure(bound) != jax.tree_util.tree_structure(pipeline):
            raise ParameterSpaceError(
                "Binding changed the pipeline's pytree structure. Bindings may only replace "
                "leaf VALUES — posterior-predictive vmapping, Fisher flattening and jit all "
                "assume a fixed treedef."
            )

        # A treedef encodes neither shape nor dtype, so the structure check above
        # lets a scalar written into an (n_time,) leaf — or a complex value into a
        # real one — through to be broadcast by the operator. Compare the leaves.
        for (path, before), after in zip(
            jax.tree_util.tree_flatten_with_path(pipeline)[0],
            jax.tree_util.tree_leaves(bound),
            strict=True,
        ):
            if not eqx.is_array(before):
                continue
            where = jax.tree_util.keystr(path)
            if jnp.shape(after) != jnp.shape(before):
                raise ParameterSpaceError(
                    f"Binding changed the shape of leaf {where} from {jnp.shape(before)} to "
                    f"{jnp.shape(after)}. The operator would broadcast it into a finite, "
                    "correctly-shaped, wrong model."
                )
            if _dtype_kind(jnp.result_type(after)) != _dtype_kind(jnp.result_type(before)):
                raise ParameterSpaceError(
                    f"Binding changed leaf {where} from {_dtype_kind(jnp.result_type(before))} "
                    f"to {_dtype_kind(jnp.result_type(after))}."
                )

        if self.raw_bind is not None:
            self._reject_latents_the_raw_bind_ignores(pipeline)

    def _reject_latents_the_raw_bind_ignores(self, pipeline: AbstractOperator) -> None:
        """The dead-latent check, for spaces whose bind function is opaque.

        The declarative path can see that a latent is named by no binding. A raw
        bind function cannot be read, so probe it: perturb one latent at a time
        and look for any change in the bound pipeline. Two perturbations, because
        a single one could land on a genuine null of a legitimate binding.
        """
        base = self.initial_values()
        reference = jax.tree.leaves(self.bind(pipeline, base))
        for latent in self.latents:
            scale = jnp.maximum(jnp.max(jnp.abs(latent.init)), 1.0)
            moved = False
            for step in (0.37, -1.61):
                probe = {**base, latent.name: latent.init + step * scale}
                candidate = jax.tree.leaves(self.bind(pipeline, probe))
                if any(
                    not jnp.array_equal(a, b)
                    for a, b in zip(reference, candidate, strict=True)
                    if eqx.is_array(a)
                ):
                    moved = True
                    break
            if not moved:
                raise ParameterSpaceError(
                    f"Latent {latent.name!r} does not reach the pipeline: perturbing it leaves "
                    "the bound model bitwise unchanged. It would be sampled without entering "
                    "the model, and its posterior would just be its prior. Check the bind "
                    "function passed to ParameterSpace.raw."
                )

    def bind(
        self, pipeline: AbstractOperator, values: dict[str, jax.Array]
    ) -> AbstractOperator:
        """Return a copy of ``pipeline`` carrying ``values``.

        Pure: ``pipeline`` is untouched. All bindings are applied in a single
        ``eqx.tree_at`` call.

        Note what that does NOT give you: ``eqx.tree_at`` resolves replacements
        by leaf identity and lets the LAST write to a leaf win, silently. The
        guarantee that no leaf is written twice comes from :meth:`validate`,
        which every entry point (:meth:`forward_fn`, the Bayesian bridge) runs
        first. Calling ``bind`` directly on an unvalidated space skips it.
        """
        if self.raw_bind is not None:
            return self.raw_bind(pipeline, values)
        selectors: list[Callable] = []
        produced: list[jax.Array] = []
        for binding in self.bindings:
            selectors.extend(binding.into)
            produced.extend(binding.evaluate(values))
        return eqx.tree_at(
            lambda p: tuple(selector(p) for selector in selectors), pipeline, tuple(produced)
        )

    def forward_fn(
        self, pipeline: AbstractOperator, state_template: State
    ) -> tuple[Callable[[dict[str, jax.Array]], jax.Array], dict[str, jax.Array]]:
        """Build ``forward(values) -> prediction`` and the starting values.

        The D7 seam, re-expressed over *named* parameters:
        :func:`~rheplicant.inference.forward.build_forward_fn` hands back a
        ghost pipeline whose leaves are the trainables, which is right when
        the answer is "train this whole subtree"; this hands back a plain
        ``dict`` of named arrays, which is right when the parameters are
        chosen, transformed, or shared. Both feed the same calibrators, Fisher
        tooling and posterior-predictive machinery — a dict is a pytree.

        The space is validated against the pipeline first. Validation reads
        shapes only and happens once per build, not per evaluation, so there
        is no reason to make it skippable.

        Args:
            pipeline: the forward model.
            state_template: the state it is evaluated on. Closed over, fixed.

        Returns:
            ``(forward, values0)``. ``values0`` is
            :meth:`initial_values`, so ``forward(values0)`` is the model at its
            declared starting point.
        """
        self.validate(pipeline)

        def forward(values: dict[str, jax.Array]) -> jax.Array:
            return self.bind(pipeline, values)(state_template).data

        return forward, self.initial_values()
